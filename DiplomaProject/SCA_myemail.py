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

# Define the objective function for SCA (negative accuracy for minimization)
def fitness_function(weights, X_train, Y_train, X_val, Y_val):
    n_particles = weights.shape[0]
    accuracy = np.zeros(n_particles)

    for i, w in enumerate(weights):
        # Set weights and intercept for logistic regression
        model = LogisticRegression(max_iter=1000)
        model.coef_ = np.array([w[:-1]])  # Use all but last as coefficients
        model.intercept_ = np.array([w[-1]])  # Last as intercept
        model.fit(X_train, Y_train)

        # Make predictions on the validation set
        Y_pred = model.predict(X_val)

        # Calculate accuracy
        accuracy[i] = accuracy_score(Y_val, Y_pred)

    # We return negative accuracy because we want to minimize the function
    return -accuracy

# Sine Cosine Algorithm (SCA) for logistic regression weights optimization
def sca_optimizer(dim, bounds, pop_size, max_iter, X_train, Y_train, X_val, Y_val):
    # Initialize population
    population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
    best_sol = population[0]
    best_fit = float('inf')

    for iteration in range(max_iter):
        r1 = 2 - iteration * (2 / max_iter)  # Linearly decreasing r1

        for i in range(pop_size):
            r2, r3, r4 = np.random.rand(3)
            if r4 < 0.5:
                population[i] += r1 * np.sin(r2) * abs(r3 * best_sol - population[i])
            else:
                population[i] += r1 * np.cos(r2) * abs(r3 * best_sol - population[i])

            # Ensure the solutions are within the bounds
            population[i] = np.clip(population[i], bounds[0], bounds[1])

        # Evaluate fitness of population
        fitness = fitness_function(population, X_train, Y_train, X_val, Y_val)

        # Update the best solution found so far
        if np.min(fitness) < best_fit:
            best_fit = np.min(fitness)
            best_sol = population[np.argmin(fitness)]

        print(f"Iteration {iteration + 1}: Best Fitness = {-best_fit:.5f}")

    return best_sol, best_fit

# Define the SCA-based model
def sca_model(dat):
    # Extract input and output data
    x = dat[:, :-1]  # Features are all columns except the last
    y = dat[:, -1]  # Target is the last column

    # Shuffle the data
    randomnum = np.random.permutation(len(y))
    xrand = x[randomnum]
    yrand = y[randomnum]

    # Split data into training, validation, and test sets (40%, 30%, 30%)
    X_train, X_temp, Y_train, Y_temp = train_test_split(xrand, yrand, test_size=0.60, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=42)

    # Normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Define the SCA hyperparameters
    pop_size = 50  # Population size
    dim = X_train.shape[1] + 1  # Number of features + 1 for intercept
    bounds = [-1, 1]  # Lower and upper bounds of the weights
    max_iter = 100  # Maximum number of iterations

    # Run the optimizer
    best_pos, best_fit = sca_optimizer(dim, bounds, pop_size, max_iter, X_train, Y_train, X_val, Y_val)

    # Create the logistic regression model with the best weights
    model = LogisticRegression(max_iter=1000)
    model.coef_ = np.array([best_pos[:-1]])  # Best position found as coefficients
    model.intercept_ = np.array([best_pos[-1]])  # Last value is the intercept
    model.fit(X_train, Y_train)  # Fit to train data

    # Predict outputs
    Yp_val = model.predict(X_val)
    Yp_test = model.predict(X_test)

    # Evaluate the model
    accval = 100 * accuracy_score(Y_val, Yp_val)
    acctest = 100 * accuracy_score(Y_test, Yp_test)

    # Confusion matrix for validation data
    tnval, fpval, fnval, tpval = confusion_matrix(Y_val, Yp_val).ravel()
    tnval = 100 * tnval / len(Y_val)
    tpval = 100 * tpval / len(Y_val)
    fnval = 100 * fnval / len(Y_val)
    fpval = 100 * fpval / len(Y_val)

    # Confusion matrix for test data
    tntest, fptest, fntest, tptest = confusion_matrix(Y_test, Yp_test).ravel()
    tntest = 100 * tntest / len(Y_test)
    tptest = 100 * tptest / len(Y_test)
    fntest = 100 * fntest / len(Y_test)
    fptest = 100 * fptest / len(Y_test)

    return model, accval, acctest, tnval, tpval, fnval, fpval, tntest, tptest, fntest, fptest

# Now run the sca_model function on the loaded dataset
model, accval, acctest, tnval, tpval, fnval, fpval, tntest, tptest, fntest, fptest = sca_model(dat)

# Print the results
print(f"Validation Accuracy: {accval:.2f}%")
print(f"Test Accuracy: {acctest:.2f}%")
print(f"Validation Confusion Matrix: TN={tnval:.2f}%, FP={fpval:.2f}%, FN={fnval:.2f}%, TP={tpval:.2f}%")
print(f"Test Confusion Matrix: TN={tntest:.2f}%, FP={fptest:.2f}%, FN={fntest:.2f}%, TP={tptest:.2f}%")