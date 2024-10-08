from __future__ import print_function
import os
import numpy as np
import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pyswarms as ps


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

# Combine features and target into a single dataset
dat = np.column_stack((X, Y))

# Define the objective function for PSO (negative accuracy for minimization)
def fitness_function(weights, X_train, Y_train, X_val, Y_val):
    n_particles = weights.shape[0]
    accuracy = np.zeros(n_particles)

    for i, w in enumerate(weights):
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, Y_train)

        Y_pred = model.predict(X_val)

        accuracy[i] = accuracy_score(Y_val, Y_pred)

    return -accuracy


# Define the PSO-based model
def pso_model(dat):
    x = dat[:, :-1]  # Features
    y = dat[:, -1]  # Target

    # Shuffle and split the data
    randomnum = np.random.permutation(len(y))
    xrand = x[randomnum]
    yrand = y[randomnum]

    X_train, X_temp, Y_train, Y_temp = train_test_split(xrand, yrand, test_size=0.60, random_state=42, stratify=yrand)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=42, stratify=Y_temp)

    # Normalize
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Define the PSO hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    n_dim = X_train.shape[1]

    optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=n_dim, options=options)

    best_cost, best_pos = optimizer.optimize(fitness_function, iters=100, X_train=X_train, Y_train=Y_train, X_val=X_val,
                                             Y_val=Y_val)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)

    Yp_val = model.predict(X_val)
    Yp_test = model.predict(X_test)

    accval = 100 * accuracy_score(Y_val, Yp_val)
    acctest = 100 * accuracy_score(Y_test, Yp_test)

    tnval, fpval, fnval, tpval = confusion_matrix(Y_val, Yp_val).ravel()
    tntest, fptest, fntest, tptest = confusion_matrix(Y_test, Yp_test).ravel()

    return model, accval, acctest, tnval, tpval, fnval, fpval, tntest, tptest, fntest, fptest


model, accval, acctest, tnval, tpval, fnval, fpval, tntest, tptest, fntest, fptest = pso_model(dat)

print(f"Validation Accuracy: {accval:.2f}%")
print(f"Test Accuracy: {acctest:.2f}%")
print(f"Validation Confusion Matrix: TN={tnval:.2f}%, FP={fpval:.2f}%, FN={fnval:.2f}%, TP={tpval:.2f}%")
print(f"Test Confusion Matrix: TN={tntest:.2f}%, FP={fptest:.2f}%, FN={fntest:.2f}%, TP={tptest:.2f}%")
