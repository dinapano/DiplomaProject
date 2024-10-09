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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

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

    # Retrieve emails
    results = service.users().messages().list(userId='me', maxResults=100).execute()
    messages = results.get('messages', [])

    data = []
    if messages:
        for message in messages:
            msg = service.users().messages().get(userId='me', id=message['id']).execute()
            subject = next(header['value'] for header in msg['payload']['headers'] if header['name'] == 'Subject')
            sender = next(header['value'] for header in msg['payload']['headers'] if header['name'] == 'From')
            date = next(header['value'] for header in msg['payload']['headers'] if header['name'] == 'Date')
            recipient = next((header['value'] for header in msg['payload']['headers'] if header['name'] == 'To'), 'No recipient found')
            body = msg.get('snippet', 'No body message found')
            data.append([subject, sender, recipient, body])

    df = pd.DataFrame(data, columns=['Subject', 'Sender', 'Recipient', 'Body'])
    df.to_csv('emails.csv', index=False)

if __name__ == '__main__':
    main()

url = 'emails.csv'
def load_gmail_data():
    try:
        dat = pd.read_csv(url)
        print(f"Dataset loaded successfully with shape: {dat.shape}")
        return dat
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

dat = load_gmail_data()

def svm_model(dat):
    if dat is None:
        print("No data to process.")
        return

    # Check the shape of the dataset
    print(f"Initial dataset shape: {dat.shape}")

    # Extract input and output data
    x = dat[['Subject', 'Sender', 'Body']]
    y = dat['Recipient']

    # Handle missing values
    x = x.fillna('')  # Replace NaN values with empty strings

    # Convert text data to numerical features using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(x['Subject'] + ' ' + x['Sender'] + ' ' + x['Body']).toarray()  # Combine subject, sender, body into one text

    # Normalize the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split data into training, validation, and test sets (40%, 30%, 30%)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, y, test_size=0.60, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=42)

    # Create and train the SVM model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, Y_train)

    # Predict outputs
    Yp_val = model.predict(X_val)
    Yp_test = model.predict(X_test)

    # Evaluate the model
    accval = 100 * accuracy_score(Y_val, Yp_val)
    acctest = 100 * accuracy_score(Y_test, Yp_test)

    # Confusion matrix output
    cm_val = confusion_matrix(Y_val, Yp_val)
    cm_test = confusion_matrix(Y_test, Yp_test)

    # Print the results of the model evaluation and confusion matrix
    print(f"Validation Accuracy: {accval:.2f}%")
    print(f"Test Accuracy: {acctest:.2f}%")
    print("Validation Confusion Matrix:\n", cm_val)
    print("Test Confusion Matrix:\n", cm_test)

# Call the svm_model function with the loaded dataset
svm_model(dat)
