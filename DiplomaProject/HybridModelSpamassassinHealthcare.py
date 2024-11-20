import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from scipy.special import softmax


# Function to read raw email files and extract content and label (spam or ham) with healthcare filter
def load_spamassassin_data(spam_dir, ham_dir):
    if not os.path.exists(spam_dir) or not os.path.exists(ham_dir):
        raise FileNotFoundError("Spam or Ham directory does not exist!")

    emails = []
    labels = []

    healthcare_keywords = set(["medicine", "healthcare", "doctor", "prescription",
                                "pharmacy", "treatment", "hospital", "health", "clinic"])

    def contains_healthcare_content(email_content):
        email_words = set(email_content.lower().split())
        return bool(healthcare_keywords & email_words)

    for label, directory in [(1, spam_dir), (0, ham_dir)]:
        for filename in os.listdir(directory):
            try:
                with open(os.path.join(directory, filename), 'r', encoding='latin-1') as file:
                    content = file.read()
                    if contains_healthcare_content(content):
                        emails.append(content)
                        labels.append(label)
            except Exception as e:
                print(f"Skipping file {filename} due to error: {e}")

    return emails, np.array(labels)


# Preprocess the emails and labels (vectorize emails, split, and normalize)
def preprocess_data(emails, labels, num_classes=2):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(emails).toarray()

    categorical_labels = to_categorical(labels, num_classes=num_classes)

    X_train, X_temp, Y_train, Y_temp = train_test_split(X, categorical_labels, test_size=0.60, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler, vectorizer


# Train and evaluate SVM model
def svm_model(X_train, Y_train, X_val, X_test):
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, np.argmax(Y_train, axis=1))

    Yp_val = model.predict_proba(X_val)
    Yp_test = model.predict_proba(X_test)

    return Yp_val, Yp_test


# Train Logistic Regression model
def logistic_regression(X_train, Y_train, X_val, X_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, np.argmax(Y_train, axis=1))

    Yp_val = model.predict_proba(X_val)
    Yp_test = model.predict_proba(X_test)

    return Yp_val, Yp_test


# Train MLP (Neural Network) model
def mlp_model(X_train, Y_train, X_val, X_test):
    model = Sequential([
        Dense(20, input_dim=X_train.shape[1], activation='relu'),
        Dense(10, activation='relu'),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')
    model.fit(X_train, Y_train, epochs=50, batch_size=10, validation_data=(X_val, Y_val),
              callbacks=[early_stopping, model_checkpoint], verbose=0)

    Yp_val = model.predict(X_val)
    Yp_test = model.predict(X_test)

    return Yp_val, Yp_test


# Ensemble Voting
def ensemble_voting(Yp_svm, Yp_lr, Yp_mlp):
    return np.argmax(softmax(Yp_svm + Yp_lr + Yp_mlp, axis=1), axis=1)


# Evaluate models
def evaluate(Yp_val, Y_val, Yp_test, Y_test):
    def confusion_matrix(predictions, true_labels):
        # Predictions are already class labels
        accuracy = 100 * np.mean(predictions == true_labels)
        tn = 100 * np.mean((predictions == 0) & (true_labels == 0))
        tp = 100 * np.mean((predictions == 1) & (true_labels == 1))
        fn = 100 * np.mean((predictions == 0) & (true_labels == 1))
        fp = 100 * np.mean((predictions == 1) & (true_labels == 0))
        return accuracy, tn, tp, fn, fp

    # Convert validation and test labels to their class indices
    Y_val_labels = np.argmax(Y_val, axis=1)
    Y_test_labels = np.argmax(Y_test, axis=1)

    # Use the predictions directly (already 1D arrays)
    accval, tnval, tpval, fnval, fpval = confusion_matrix(Yp_val, Y_val_labels)
    acctest, tntest, tptest, fntest, fptest = confusion_matrix(Yp_test, Y_test_labels)

    print("\nValidation Confusion Matrix:")
    print(f"Accuracy: {accval:.2f}% | TN: {tnval:.2f}% | TP: {tpval:.2f}% | FN: {fnval:.2f}% | FP: {fpval:.2f}%")
    print("\nValidation Classification Report:")
    print(classification_report(Y_val_labels, Yp_val, target_names=["Not Spam", "Spam"]))
    print("\nTest Confusion Matrix:")
    print(f"Accuracy: {acctest:.2f}% | TN: {tntest:.2f}% | TP: {tptest:.2f}% | FN: {fntest:.2f}% | FP: {fptest:.2f}%")
    print("\nTest Classification Report:")
    print(classification_report(Y_test_labels, Yp_test, target_names=["Not Spam", "Spam"]))



# Load the dataset and preprocess
spam_dir = 'spamassassin/spam_2/'
ham_dir = 'spamassassin/easy_ham/'
emails, labels = load_spamassassin_data(spam_dir, ham_dir)

X_train, X_val, X_test, Y_train, Y_val, Y_test, scaler, vectorizer = preprocess_data(emails, labels)

# Train individual models
Yp_svm_val, Yp_svm_test = svm_model(X_train, Y_train, X_val, X_test)
Yp_lr_val, Yp_lr_test = logistic_regression(X_train, Y_train, X_val, X_test)
Yp_mlp_val, Yp_mlp_test = mlp_model(X_train, Y_train, X_val, X_test)

# Combine predictions using ensemble voting
Yp_val = ensemble_voting(Yp_svm_val, Yp_lr_val, Yp_mlp_val)
Yp_test = ensemble_voting(Yp_svm_test, Yp_lr_test, Yp_mlp_test)

# Evaluate the ensemble model
evaluate(Yp_val, Y_val, Yp_test, Y_test)
