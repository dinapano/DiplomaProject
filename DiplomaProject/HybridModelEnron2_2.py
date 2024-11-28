import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pyswarms as ps


# Function to read raw email files and extract content and label (spam or ham)
def load_enron2_data(spam_dir, ham_dir):
    emails = []
    labels = []

    for filename in os.listdir(spam_dir):
        with open(os.path.join(spam_dir, filename), 'r', encoding='latin-1') as file:
            content = file.read()
            emails.append(content)
            labels.append(1)  # Spam

    for filename in os.listdir(ham_dir):
        with open(os.path.join(ham_dir, filename), 'r', encoding='latin-1') as file:
            content = file.read()
            emails.append(content)
            labels.append(0)  # Ham

    return emails, np.array(labels)


# Preprocess the emails and labels (vectorize emails, split, and normalize)
def preprocess_data(emails, labels, num_classes=2):
    # Convert emails to a matrix of token counts (features)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(emails).toarray()

    # Convert labels to categorical
    categorical_labels = to_categorical(labels, num_classes=num_classes)

    # Split data
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, categorical_labels, test_size=0.60, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=42)

    # Normalize data (optional, depending on model choice)
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
    model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_data=(X_val, Y_val),
              callbacks=[early_stopping, model_checkpoint], verbose=0)

    # Get class probabilities
    Yp_val = model.predict(X_val)  # This returns probabilities by default
    Yp_test = model.predict(X_test)  # This returns probabilities by default
    return Yp_val, Yp_test


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

# Preprocess the emails (you need to provide the correct paths to spam and ham directories)
spam_dir = 'enron2/spam/'
ham_dir = 'enron2/ham/'
emails, labels = load_enron2_data(spam_dir, ham_dir)

X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler = preprocess_data(emails, labels)

# Train and Evaluate
Yp_svm_val, Yp_svm_test = svm_model(X_train, Y_train, X_val, X_test)
Yp_lr_val, Yp_lr_test = pso_model(X_train, Y_train, X_val, X_test)
Yp_mlp_val, Yp_mlp_test = mlp_model(X_train, Y_train, X_val, Y_val)

# Ensemble Predictions
Yp_val = ensemble_voting(Yp_svm_val, Yp_lr_val, Yp_mlp_val)
Yp_test = ensemble_voting(Yp_svm_test, Yp_lr_test, Yp_mlp_test)

# Evaluate Ensemble Model
evaluate(Yp_val, Y_val, Yp_test, Y_test)