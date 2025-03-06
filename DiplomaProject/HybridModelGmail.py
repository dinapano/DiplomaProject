import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


# Define Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Authenticate and connect to the Gmail API
def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    service = build('gmail', 'v1', credentials=creds)
    return service

# Fetch messages by label (INBOX or SPAM)
def fetch_messages(service, label_id, max_results=100):
    results = service.users().messages().list(userId='me', labelIds=[label_id], maxResults=max_results).execute()
    messages = results.get('messages', [])
    email_data = []

    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id']).execute()
        subject = next((header['value'] for header in msg['payload']['headers'] if header['name'] == 'Subject'), 'No Subject')
        sender = next((header['value'] for header in msg['payload']['headers'] if header['name'] == 'From'), 'No Sender')
        body = msg.get('snippet', 'No Body')
        email_data.append([subject, sender, body])

    df = pd.DataFrame(email_data, columns=['Subject', 'Sender', 'Body'])
    return df

# Main function to fetch and preprocess emails
def main():
    service = authenticate_gmail()

    # Retrieve emails after Gmail has filtered them, but use them as labeled data to train your own model.
    # Retrieve INBOX emails
    inbox_emails = fetch_messages(service, label_id='INBOX', max_results=10000)
    inbox_emails['Label'] = 'Inbox'

    # Retrieve SPAM emails
    spam_emails = fetch_messages(service, label_id='SPAM', max_results=10000)
    spam_emails['Label'] = 'Spam'

    # Combine datasets
    all_emails = pd.concat([inbox_emails, spam_emails], ignore_index=True)
    all_emails.to_csv('emails.csv', index=False)
    return all_emails

# Load Gmail data
dat = main()

# Preprocess text
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(dat['Body'].fillna('')).toarray()

# Encode labels
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(dat['Label'])
Y = to_categorical(Y)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, Y, test_size=0.5, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.6, random_state=42)

# SVM Model
def svm_model(X_train, Y_train, X_val, X_test):
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, np.argmax(Y_train, axis=1))
    Yp_val = model.predict_proba(X_val)
    Yp_test = model.predict_proba(X_test)
    return Yp_val, Yp_test

# Logistic Regression Model
def logistic_regression_sca(X_train, Y_train, X_val, X_test):
    model = LogisticRegression(max_iter=1000)
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

# Ensemble Voting
def ensemble_voting(Yp_svm, Yp_lr, Yp_mlp):
    # Ensure all probability arrays are NumPy arrays
    Yp_svm = np.array(Yp_svm)
    Yp_lr = np.array(Yp_lr)
    Yp_mlp = np.array(Yp_mlp)

    # Check and reshape arrays if needed
    min_samples = min(Yp_svm.shape[0], Yp_lr.shape[0], Yp_mlp.shape[0])

    Yp_svm = Yp_svm[:min_samples]
    Yp_lr = Yp_lr[:min_samples]
    Yp_mlp = Yp_mlp[:min_samples]

    combined_probs = (Yp_svm + Yp_lr + Yp_mlp) / 3
    return np.argmax(combined_probs, axis=1)


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


# Train and Evaluate
Yp_svm_val, Yp_svm_test = svm_model(X_train, Y_train, X_val, X_test)
Yp_lr_val, Yp_lr_test = logistic_regression_sca(X_train, Y_train, X_val, X_test)
Yp_mlp_val, Yp_mlp_test, history = mlp_model(X_train, Y_train, X_val, Y_val)

# Ensemble Predictions
Yp_val = ensemble_voting(Yp_svm_val, Yp_lr_val, Yp_mlp_val)
Yp_test = ensemble_voting(Yp_svm_test, Yp_lr_test, Yp_mlp_test)

# Evaluate Ensemble Model
evaluate(Yp_val, Y_val, Yp_test, Y_test)

#Calculating f1-score for every epoch
f1_scores = [
    f1_score(np.argmax(Y_val, axis=1),
             ensemble_voting(Yp_svm_val, Yp_lr_val, Yp_mlp_val))
    for _ in range(len(history.epoch))  # Avoid using slicing on predictions
]

plt.scatter(history.epoch, f1_scores, color='blue', label='F1 Score per Epoch')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('MLP-SCA-SVM Model using Gmail Dataset')
plt.grid(True)
plt.legend()
plt.show()

