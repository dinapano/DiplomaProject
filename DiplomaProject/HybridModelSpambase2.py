import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pyswarms as ps


# Load the Spambase dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
data = np.loadtxt(url, delimiter=',')

def preprocess_data(data, num_classes=2):
    """Preprocess the data by shuffling, splitting, and normalizing."""
    features = data[:, :-1]
    labels = data[:, -1]

    # Shuffle data
    indices = np.random.permutation(len(labels))
    shuffled_features = features[indices]
    shuffled_labels = labels[indices]

    # Convert labels to categorical
    categorical_labels = to_categorical(shuffled_labels, num_classes=num_classes)

    # Split data
    X_train, X_temp, Y_train, Y_temp = train_test_split(shuffled_features, categorical_labels, test_size=0.60,
                                                        random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=42)

    # Normalize data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler

# Define the objective function for PSO (negative accuracy for minimization)
def fitness_function(weights, X_train, Y_train, X_val, Y_val):
    n_particles = weights.shape[0]
    accuracy = np.zeros(n_particles)

    for i, w in enumerate(weights):
        # Set weights and intercept for logistic regression
        model = LogisticRegression(max_iter=1000)
        model.coef_ = np.array([w[:-1]])  # Use all but last as coefficients
        model.intercept_ = np.array([w[-1]])  # Last element as intercept

        # Convert Y_train to class indices (use np.argmax to get integer labels)
        model.fit(X_train, np.argmax(Y_train, axis=1))  # Convert one-hot to integers

        # Make predictions on the validation set
        Y_pred = model.predict(X_val)

        # Calculate accuracy
        accuracy[i] = accuracy_score(np.argmax(Y_val, axis=1), Y_pred)  # Convert Y_val to integers

    # We return negative accuracy because we want to minimize the function
    return -accuracy


# Define the PSO-based model
def pso_model(X_train, Y_train, X_val, X_test):
    # Define the PSO hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # Cognitive and social parameters
    n_dim = X_train.shape[1] + 1  # Number of features + 1 for intercept

    # Create a particle swarm optimizer
    optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=n_dim, options=options)

    # Run the optimizer
    best_cost, best_pos = optimizer.optimize(fitness_function, iters=100, X_train=X_train, Y_train=Y_train, X_val=X_val,
                                             Y_val=Y_val)

    # Create the logistic regression model with the best weights
    model = LogisticRegression(max_iter=1000)
    model.coef_ = np.array([best_pos[:-1]])  # Best position found as coefficients
    model.intercept_ = np.array([best_pos[-1]])  # Last value is the intercept
    model.fit(X_train, np.argmax(Y_train, axis=1))  # Fit with class indices

    # Predict probabilities (not labels)
    Yp_val = model.predict_proba(X_val)  # Get class probabilities
    Yp_test = model.predict_proba(X_test)  # Get class probabilities
    return Yp_val, Yp_test


# SVM Model
def svm_model(X_train, Y_train, X_val, X_test):
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, np.argmax(Y_train, axis=1))  # Convert Y_train to integer labels
    Yp_val = model.predict_proba(X_val)  # Get class probabilities
    Yp_test = model.predict_proba(X_test)  # Get class probabilities
    return Yp_val, Yp_test


# MLP Model
def mlp_model(X_train, Y_train, X_val, Y_val):
    num_classes = Y_train.shape[1]
    model = Sequential([
        Dense(128, activation='relu', input_dim=X_train.shape[1]),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
    history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_val, Y_val),
              callbacks=[early_stopping, model_checkpoint], verbose=0)
    Yp_val = model.predict(X_val)
    Yp_test = model.predict(X_test)
    return Yp_val, Yp_test, history


# Ensemble Voting
def ensemble_voting(Yp_svm, Yp_lr, Yp_mlp):
    combined_probs = (Yp_svm + Yp_lr + Yp_mlp) / 3  # Average probabilities across models
    return np.argmax(combined_probs, axis=1)  # Return the class with the highest probability


# Evaluate function
def evaluate(Yp_val, Y_val, Yp_test, Y_test):
    def calculate_confusion(predictions, true_labels):
        accuracy = 100 * np.mean(predictions == true_labels)
        tn = 100 * np.mean((predictions == 0) & (true_labels == 0))
        tp = 100 * np.mean((predictions == 1) & (true_labels == 1))
        fn = 100 * np.mean((predictions == 0) & (true_labels == 1))
        fp = 100 * np.mean((predictions == 1) & (true_labels == 0))
        return accuracy, tn, tp, fn, fp

    Y_val_labels = np.argmax(Y_val, axis=1)
    Y_test_labels = np.argmax(Y_test, axis=1)

    accval, tnval, tpval, fnval, fpval = calculate_confusion(Yp_val, Y_val_labels)
    acctest, tntest, tptest, fntest, fptest = calculate_confusion(Yp_test, Y_test_labels)

    print("\nValidation Metrics:")
    print("\nValidation Confusion Matrix:")
    print(f"Accuracy: {accval:.2f}% | TN: {tnval:.2f}% | TP: {tpval:.2f}% | FN: {fnval:.2f}% | FP: {fpval:.2f}%")
    print("\nValidation Classification Report:")
    print(classification_report(Y_val_labels, Yp_val, target_names=["Not Spam", "Spam"]))

    print("\nTest Metrics:")
    print("\nTest Confusion Matrix:")
    print(f"Accuracy: {acctest:.2f}% | TN: {tntest:.2f}% | TP: {tptest:.2f}% | FN: {fntest:.2f}% | FP: {fptest:.2f}%")
    print("\nTest Classification Report:")
    print(classification_report(Y_test_labels, Yp_test, target_names=["Not Spam", "Spam"]))

# Run the data preprocessing
X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler = preprocess_data(data)

# Train and Evaluate
Yp_svm_val, Yp_svm_test = svm_model(X_train, Y_train, X_val, X_test)
Yp_lr_val, Yp_lr_test = pso_model(X_train, Y_train, X_val, X_test)
Yp_mlp_val, Yp_mlp_test, history = mlp_model(X_train, Y_train, X_val, Y_val)

# Ensemble Predictions
Yp_val = ensemble_voting(Yp_svm_val, Yp_lr_val, Yp_mlp_val)
Yp_test = ensemble_voting(Yp_svm_test, Yp_lr_test, Yp_mlp_test)

# Evaluate Ensemble Model
evaluate(Yp_val, Y_val, Yp_test, Y_test)

f1_scores = [
    f1_score(np.argmax(Y_val, axis=1),
             ensemble_voting(Yp_svm_val, Yp_lr_val, Yp_mlp_val))
    for _ in range(len(history.epoch))  # Avoid using slicing on predictions
]

plt.scatter(history.epoch, f1_scores, color='blue', label='F1 Score per Epoch')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('MLP-PSO-SVM Model using Spambase Dataset')
plt.grid(True)
plt.legend()
plt.show()