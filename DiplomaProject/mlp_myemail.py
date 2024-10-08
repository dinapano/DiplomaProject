from __future__ import print_function
import os
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def main():
    """Shows basic usage of the Gmail API. Lists the user's Gmail labels."""
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

    # Retrieve emails
    results = service.users().messages().list(userId='me', maxResults=1000000).execute()
    messages = results.get('messages', [])

    data = []
    if not messages:
        print('No messages found.')
    else:
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            subject = next(header['value'] for header in msg['payload']['headers'] if header['name'] == 'Subject')
            sender = next(header['value'] for header in msg['payload']['headers'] if header['name'] == 'From')
            date = next(header['value'] for header in msg['payload']['headers'] if header['name'] == 'Date')
            recipient = next((header['value'] for header in msg['payload']['headers'] if header['name'] == 'To'), 'No recipient found')
            body = msg.get('snippet', 'No body message found')
            data.append([subject, sender, recipient, body])
            print(subject)
            print(body)

    df = pd.DataFrame(data, columns=['Subject', 'Sender', 'Recipient', 'Body'])
    df.to_csv('emails.csv', index=False)

if __name__ == '__main__':
    main()

# Define the URL for the gmail dataset
url = 'emails.csv'

# Load the gmail dataset
def load_gmail_data():
    try:
        dat = pd.read_csv(url, delimiter=',')
        print(f"Dataset loaded successfully with shape: {dat.shape}")
        return dat
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

# Load the dataset
dat = load_gmail_data()

# Apply TfidfVectorizer on the "Body" column
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dat['Body'].fillna(''))

# Convert to dense matrix if necessary
X = X.toarray()

# Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Dummy target: Here, I'm using the 'Recipient' column as a dummy multi-output target
# In reality, you need meaningful target labels for your classification task.
Y = pd.get_dummies(dat['Recipient'])

# Split data into training, validation, and test sets (40%, 30%, 30%)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.30, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.20, random_state=42)

# Create and train the MLP model
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)

# Wrap the MLPClassifier with MultiOutputClassifier to support multi-output data
model = MultiOutputClassifier(mlp, n_jobs=-1)

# Fit the model
model.fit(X_train, Y_train)

# Predict outputs
Yp_val = model.predict(X_val)
Yp_test = model.predict(X_test)

# Calculate accuracy for validation and test sets
average_accuracy_val = accuracy_score(Y_val, Yp_val)
average_accuracy_test = accuracy_score(Y_test, Yp_test)

# Confusion matrix for validation
conf_matrix_val = confusion_matrix(Y_val.values.argmax(axis=1), Yp_val.argmax(axis=1))

# Confusion matrix for test
conf_matrix_test = confusion_matrix(Y_test.values.argmax(axis=1), Yp_test.argmax(axis=1))

# Output the results
print(f"Average Validation Accuracy: {average_accuracy_val * 100:.2f}%")
print(f"Average Test Accuracy: {average_accuracy_test * 100:.2f}%")
print(f"Validation Confusion Matrix:\n{conf_matrix_val}")
print(f"Test Confusion Matrix:\n{conf_matrix_test}")
