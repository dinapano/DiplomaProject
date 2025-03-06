import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.special import softmax

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


def svm_model(X_train, Y_train, X_val, X_test):
    """Train an SVM model."""
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, np.argmax(Y_train, axis=1))

    # Get probability predictions for ensemble voting
    Yp_val = model.predict_proba(X_val)
    Yp_test = model.predict_proba(X_test)

    return Yp_val, Yp_test


def logistic_regression_sca(X_train, Y_train, X_val, X_test, scaler):
    """Optimize logistic regression weights with SCA."""

    def fitness_function(weights, X_train, Y_train, X_val, Y_val):
        accuracy = []
        for w in weights:
            model = LogisticRegression(max_iter=1000)
            model.coef_ = np.array([w[:-1]])
            model.intercept_ = np.array([w[-1]])
            model.fit(X_train, Y_train)

            Yp = model.predict(X_val)
            accuracy.append(accuracy_score(Y_val, Yp))

        return -np.array(accuracy)

    def sca_optimizer(dim, bounds, pop_size, max_iter, X_train, Y_train, X_val, Y_val):
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
        best_sol = population[0]
        best_fit = float('inf')

        for iteration in range(max_iter):
            r1 = 2 - iteration * (2 / max_iter)
            for i in range(pop_size):
                r2, r3, r4 = np.random.rand(3)
                if r4 < 0.5:
                    population[i] += r1 * np.sin(r2) * abs(r3 * best_sol - population[i])
                else:
                    population[i] += r1 * np.cos(r2) * abs(r3 * best_sol - population[i])
                population[i] = np.clip(population[i], bounds[0], bounds[1])

            fitness = fitness_function(population, X_train, Y_train, X_val, Y_val)
            if np.min(fitness) < best_fit:
                best_fit = np.min(fitness)
                best_sol = population[np.argmin(fitness)]

        return best_sol

    dim = X_train.shape[1] + 1
    pop_size, bounds, max_iter = 50, [-1, 1], 100
    best_weights = sca_optimizer(dim, bounds, pop_size, max_iter, X_train, np.argmax(Y_train, axis=1), X_val,
                                 np.argmax(Y_val, axis=1))

    model = LogisticRegression(max_iter=1000)
    model.coef_ = np.array([best_weights[:-1]])
    model.intercept_ = np.array([best_weights[-1]])
    model.fit(X_train, np.argmax(Y_train, axis=1))

    Yp_val = model.predict_proba(X_val)
    Yp_test = model.predict_proba(X_test)

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



def ensemble_voting(Yp_svm, Yp_lr, Yp_mlp):
    """Combine the predictions from each model using ensemble voting."""
    return np.argmax(softmax(Yp_svm + Yp_lr + Yp_mlp, axis=1), axis=1)


def evaluate(Yp_val, Y_val, Yp_test, Y_test):
    """Evaluate model performance."""
    def confusion_matrix(predictions, true_labels):
        accuracy = 100 * np.mean(predictions == true_labels)
        tn = 100 * np.mean((predictions == 0) & (true_labels == 0))
        tp = 100 * np.mean((predictions == 1) & (true_labels == 1))
        fn = 100 * np.mean((predictions == 0) & (true_labels == 1))
        fp = 100 * np.mean((predictions == 1) & (true_labels == 0))
        return accuracy, tn, tp, fn, fp

    # Convert one-hot labels to single class labels
    Y_val_labels = np.argmax(Y_val, axis=1)
    Y_test_labels = np.argmax(Y_test, axis=1)

    accval, tnval, tpval, fnval, fpval = confusion_matrix(Yp_val, Y_val_labels)
    acctest, tntest, tptest, fntest, fptest = confusion_matrix(Yp_test, Y_test_labels)

    # Print classification reports and confusion matrices
    print("Validation Confusion Matrix:")
    print(f"Accuracy: {accval:.2f}% | TN: {tnval:.2f}% | TP: {tpval:.2f}% | FN: {fnval:.2f}% | FP: {fpval:.2f}%")
    print("\nValidation Classification Report:")
    print(classification_report(Y_val_labels, Yp_val, target_names=["Not Spam", "Spam"]))
    print("\nTest Confusion Matrix:")
    print(f"Accuracy: {acctest:.2f}% | TN: {tntest:.2f}% | TP: {tptest:.2f}% | FN: {fntest:.2f}% | FP: {fptest:.2f}%")
    print("\nTest Classification Report:")
    print(classification_report(Y_test_labels, Yp_test, target_names=["Not Spam", "Spam"]))


# Run the data preprocessing
X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler = preprocess_data(data)

# Train and Evaluate
Yp_svm_val, Yp_svm_test = svm_model(X_train, Y_train, X_val, X_test)
Yp_lr_val, Yp_lr_test = logistic_regression_sca(X_train, Y_train, X_val, X_test, scaler)
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
plt.title('MLP-SCA-SVM Model using Spambase Dataset')
plt.grid(True)
plt.legend()
plt.show()



