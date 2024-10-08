from __future__ import print_function
import os.path
import base64
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.utils import to_categorical


# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def main():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels and emails."""
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    # Call the Gmail API
    service = build('gmail', 'v1', credentials=creds)

    # Retrieve emails
    results = service.users().messages().list(userId='me', maxResults=100000).execute()
    messages = results.get('messages', [])

    data = []

    if not messages:
        print('No messages found.')
    else:
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()

            # Get email details
            subject = next(header['value'] for header in msg['payload']['headers'] if header['name'] == 'Subject')
            sender = next(header['value'] for header in msg['payload']['headers'] if header['name'] == 'From')
            recipient = next((header['value'] for header in msg['payload']['headers'] if header['name'] == 'To'), 'No recipient found')
            body = msg.get('snippet', 'No body message found')

            data.append([subject, sender, recipient, body])

    # Convert to DataFrame and save as CSV
    df = pd.DataFrame(data, columns=['Subject', 'Sender', 'Recipient', 'Body'])
    df.to_csv('emails.csv', index=False)

if __name__ == '__main__':
    main()


# Function to load Gmail dataset
def load_gmail_data():
    url = 'emails.csv'
    try:
        dat = pd.read_csv(url, delimiter=',')
        print(f"Dataset loaded successfully with shape: {dat.shape}")
        print(dat.head())
        return dat
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None


# Load the dataset
dat = load_gmail_data()


def cnn_model(dat):
    # Filter columns and handle missing data
    dat.fillna('', inplace=True)

    # Classify emails as spam (1) or not spam (0) based on the subject line
    conditions = dat['Subject'].apply(lambda x: 1 if 'spam' in x.lower() else 0)
    y = conditions.values

    # Convert labels to categorical (one-hot encoding)
    ynew = to_categorical(y, num_classes=2)

    # Use the 'Subject', 'Sender', 'Recipient', 'Body' columns for features
    X = dat[['Subject', 'Sender', 'Recipient', 'Body']]

    # Split the data into training (80%), validation (10%), and test sets (10%)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, ynew, test_size=0.6, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

    # Vectorize text features using TF-IDF
    vectorizer_subject = TfidfVectorizer()
    X_train_subject = vectorizer_subject.fit_transform(X_train['Subject']).toarray()
    X_val_subject = vectorizer_subject.transform(X_val['Subject']).toarray()
    X_test_subject = vectorizer_subject.transform(X_test['Subject']).toarray()

    vectorizer_sender = TfidfVectorizer()
    X_train_sender = vectorizer_sender.fit_transform(X_train['Sender']).toarray()
    X_val_sender = vectorizer_sender.transform(X_val['Sender']).toarray()
    X_test_sender = vectorizer_sender.transform(X_test['Sender']).toarray()

    vectorizer_body = TfidfVectorizer()
    X_train_body = vectorizer_body.fit_transform(X_train['Body']).toarray()
    X_val_body = vectorizer_body.transform(X_val['Body']).toarray()
    X_test_body = vectorizer_body.transform(X_test['Body']).toarray()

    vectorizer_recipient = TfidfVectorizer()
    X_train_recipient = vectorizer_recipient.fit_transform(X_train['Recipient']).toarray()
    X_val_recipient = vectorizer_recipient.transform(X_val['Recipient']).toarray()
    X_test_recipient = vectorizer_recipient.transform(X_test['Recipient']).toarray()

    # Combine the features
    X_train_combined = np.hstack((X_train_subject, X_train_sender, X_train_recipient, X_train_body))
    X_val_combined = np.hstack((X_val_subject, X_val_sender, X_val_recipient, X_val_body))
    X_test_combined = np.hstack((X_test_subject, X_test_sender, X_test_recipient, X_test_body))

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_val_scaled = scaler.transform(X_val_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    # Reshape data for Conv1D layer (input for Conv1D: (samples, timesteps, features))
    X_train_reshape = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_val_reshape = np.reshape(X_val_scaled, (X_val_scaled.shape[0], X_val_scaled.shape[1], 1))
    X_test_reshape = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    # Build the CNN model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_reshape.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train_reshape, Y_train, epochs=50, batch_size=10, validation_data=(X_val_reshape, Y_val), verbose=1)

    # Predict and evaluate the model
    Yp_val = np.argmax(model.predict(X_val_reshape), axis=1)
    Yp_test = np.argmax(model.predict(X_test_reshape), axis=1)
    Y_val_labels = np.argmax(Y_val, axis=1)
    Y_test_labels = np.argmax(Y_test, axis=1)

    # Compute validation metrics
    accval = 100 * np.mean(Yp_val == Y_val_labels)
    tnval = 100 * np.mean((Yp_val == 0) & (Y_val_labels == 0))
    tpval = 100 * np.mean((Yp_val == 1) & (Y_val_labels == 1))
    fnval = 100 * np.mean((Yp_val == 0) & (Y_val_labels == 1))
    fpval = 100 * np.mean((Yp_val == 1) & (Y_val_labels == 0))

    # Compute test metrics
    acctest = 100 * np.mean(Yp_test == Y_test_labels)
    tntest = 100 * np.mean((Yp_test == 0) & (Y_test_labels == 0))
    tptest = 100 * np.mean((Yp_test == 1) & (Y_test_labels == 1))
    fntest = 100 * np.mean((Yp_test == 0) & (Y_test_labels == 1))
    fptest = 100 * np.mean((Yp_test == 1) & (Y_test_labels == 0))

    return model, accval, acctest, tnval, tpval, fnval, fpval, tntest, tptest, fntest, fptest


# Run the model
model, accval, acctest, tnval, tpval, fnval, fpval, tntest, tptest, fntest, fptest = cnn_model(dat)

# Print results
print(f"Validation Accuracy: {accval:.2f}%")
print(f"Test Accuracy: {acctest:.2f}%")
print(f"Validation Confusion Matrix: TN={tnval:.2f}%, FP={fpval:.2f}%, FN={fnval:.2f}%, TP={tpval:.2f}%")
print(f"Test Confusion Matrix: TN={tntest:.2f}%, FP={fptest:.2f}%, FN={fntest:.2f}%, TP={tptest:.2f}%")
print("Model training completed.")
