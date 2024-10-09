from __future__ import print_function
import os
import pandas as pd
import numpy as np
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix

# Define Gmail API scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def main():
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
    results = service.users().messages().list(userId='me', maxResults=100).execute()
    messages = results.get('messages', [])
    data = []

    if messages:
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            subject = next(header['value'] for header in msg['payload']['headers'] if header['name'] == 'Subject')
            sender = next(header['value'] for header in msg['payload']['headers'] if header['name'] == 'From')
            recipient = next((header['value'] for header in msg['payload']['headers'] if header['name'] == 'To'), 'No recipient found')
            body = msg.get('snippet', 'No body message found')
            data.append([subject, sender, recipient, body])

    df = pd.DataFrame(data, columns=['Subject', 'Sender', 'Recipient', 'Body'])
    df.to_csv('emails.csv', index=False)

if __name__ == '__main__':
    main()

# Load the dataset
def load_gmail_data():
    try:
        dat = pd.read_csv('emails.csv', delimiter=',')
        print(f"Dataset loaded successfully with shape: {dat.shape}")
        return dat
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

dat = load_gmail_data()

def lstm_model(dat):
    if dat is None:
        print("No data to process.")
        return

    # Extract input and output data
    x = dat[['Subject', 'Sender', 'Body']]
    y = dat['Recipient']
    x = x.fillna('')  # Handle missing values by replacing NaN values with empty strings

    # Convert text data to numerical features using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(x['Subject'] + ' ' + x['Sender'] + ' ' + x['Body']).toarray()   # Combine subject, sender, body into one text

    # Normalize the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # detects the number of classes dynamically
    num_classes = len(y.unique())
    y = pd.factorize(y)[0]

    # Convert labels to categorical (one-hot encoding)
    y = to_categorical(y, num_classes=num_classes)

    # Split data into training, validation, and test sets (40%, 30%, 30%)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, y, test_size=0.50, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.4, random_state=42)

    # Create LSTM model
    model = Sequential([
        LSTM(units=128, input_shape=(X_train.shape[1], 1), return_sequences=True),  # First LSTM layer
        Dropout(0.2),
        LSTM(units=64),  # Second LSTM layer
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Reshape X for LSTM input compatibility
    X_train = np.expand_dims(X_train, -1)
    X_val = np.expand_dims(X_val, -1)
    X_test = np.expand_dims(X_test, -1)

    # Train the LSTM
    model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_data=(X_val, Y_val), verbose=1)

    # Predict outputs
    Yp_val = np.argmax(model.predict(X_val), axis=1)
    Yp_test = np.argmax(model.predict(X_test), axis=1)
    Y_val_labels = np.argmax(Y_val, axis=1)
    Y_test_labels = np.argmax(Y_test, axis=1)

    # Evaluate the model
    accval = 100 * accuracy_score(Y_val_labels, Yp_val)
    acctest = 100 * accuracy_score(Y_test_labels, Yp_test)

    # Confusion matrix output
    val_cm = confusion_matrix(Y_val_labels, Yp_val)
    test_cm = confusion_matrix(Y_test_labels, Yp_test)
    print(f"Validation Confusion Matrix:\n{val_cm}")
    print(f"Test Confusion Matrix:\n{test_cm}")

    return model, accval, acctest

# Run the LSTM model and print results
try:
    model, accval, acctest = lstm_model(dat)
    print(f"Validation Accuracy: {accval:.2f}%")
    print(f"Test Accuracy: {acctest:.2f}%")
except Exception as e:
    print(f"Error during model training or evaluation: {e}")

