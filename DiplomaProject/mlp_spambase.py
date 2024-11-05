import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def load_data(url):
    """Load the dataset from the provided URL."""
    return np.loadtxt(url, delimiter=',')

def preprocess_data(data, num_classes=2):
    """Preprocess the data by shuffling, splitting, and normalizing."""
    features = data[:, :57]
    labels = data[:, 57]

    # Shuffle data
    indices = np.random.permutation(len(labels))
    shuffled_features = features[indices]
    shuffled_labels = labels[indices]

    # Convert labels to categorical
    categorical_labels = to_categorical(shuffled_labels, num_classes=num_classes)

    # Split data
    X_train, X_temp, Y_train, Y_temp = train_test_split(shuffled_features, categorical_labels, test_size=0.60, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=42)

    # Normalize data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def build_model(input_dim):
    """Build and compile the MLP model."""
    model = Sequential([
        Dense(20, input_dim=input_dim, activation='relu'),
        Dense(10, activation='relu'),
        Dense(2, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, Y_train, X_val, Y_val, epochs=50, batch_size=10):
    """Train the model with early stopping and model checkpointing."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

    history = model.fit(X_train, Y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_val, Y_val),
                        callbacks=[early_stopping, model_checkpoint],
                        verbose=0)
    return history

def evaluate_model(model, X_val, Y_val, X_test, Y_test):
    """Evaluate the model on validation and test data."""
    def calculate_metrics(predictions, true_labels):
        accuracy = 100 * np.mean(predictions == true_labels)
        tn = 100 * np.mean((predictions == 0) & (true_labels == 0))
        tp = 100 * np.mean((predictions == 1) & (true_labels == 1))
        fn = 100 * np.mean((predictions == 0) & (true_labels == 1))
        fp = 100 * np.mean((predictions == 1) & (true_labels == 0))
        return accuracy, tn, tp, fn, fp

    Yp_val = np.argmax(model.predict(X_val), axis=1)
    Yp_test = np.argmax(model.predict(X_test), axis=1)
    Y_val_labels = np.argmax(Y_val, axis=1)
    Y_test_labels = np.argmax(Y_test, axis=1)

    accval, tnval, tpval, fnval, fpval = calculate_metrics(Yp_val, Y_val_labels)
    acctest, tntest, tptest, fntest, fptest = calculate_metrics(Yp_test, Y_test_labels)

    print("Validation Metrics:")
    print(f"Accuracy: {accval:.2f}% | TN: {tnval:.2f}% | TP: {tpval:.2f}% | FN: {fnval:.2f}% | FP: {fpval:.2f}%")
    print("\nTest Metrics:")
    print(f"Accuracy: {acctest:.2f}% | TN: {tntest:.2f}% | TP: {tptest:.2f}% | FN: {fntest:.2f}% | FP: {fptest:.2f}%")

    print("\nClassification Report (Validation):")
    print(classification_report(Y_val_labels, Yp_val, target_names=["Not Spam", "Spam"]))

    print("\nClassification Report (Test):")
    print(classification_report(Y_test_labels, Yp_test, target_names=["Not Spam", "Spam"]))

# calling the functions in one place
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
data = load_data(url)
X_train, X_val, X_test, Y_train, Y_val, Y_test = preprocess_data(data)
model = build_model(input_dim=57)
train_model(model, X_train, Y_train, X_val, Y_val)
evaluate_model(model, X_val, Y_val, X_test, Y_test)
