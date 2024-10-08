import os
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import pyswarms as ps
import pandas as pd

# Path to the enron2 dataset
spam_dir = 'enron2/spam/'
ham_dir = 'enron2/ham/'

def load_emails(directory):
    emails = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r', encoding='latin-1') as file:
            emails.append(file.read())
    return emails

def preprocess_text(text):
    # Remove email headers, punctuation, and convert to lowercase
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\W+', ' ', text)
    text = text.lower()
    return text

# Load and preprocess the dataset
spam_emails = load_emails(spam_dir)
ham_emails = load_emails(ham_dir)

all_emails = spam_emails + ham_emails
all_labels = [1] * len(spam_emails) + [0] * len(ham_emails)  # 1 for spam, 0 for ham

# Preprocess the emails
all_emails = [preprocess_text(email) for email in all_emails]

# Convert text to a numerical feature matrix
vectorizer = CountVectorizer(max_features=1000)  # Use 1000 most frequent words as features
x = vectorizer.fit_transform(all_emails).toarray()
y = np.array(all_labels)

# Define the objective function for PSO (negative accuracy for minimization)
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


# Define the PSO-based model
def pso_model(x, y):
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

    # Define the PSO hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}  # Cognitive and social parameters
    n_dim = X_train.shape[1] + 1  # Number of features + 1 for intercept

    # Create a particle swarm optimizer
    optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=n_dim, options=options)

    # Run the optimizer
    best_cost, best_pos = optimizer.optimize(fitness_function, iters=100, X_train=X_train, Y_train=Y_train, X_val=X_val,
                                             Y_val=Y_val)

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


# Run the PSO model on the Enron dataset
model, accval, acctest, tnval, tpval, fnval, fpval, tntest, tptest, fntest, fptest = pso_model(x, y)

# Print the results
print(f"Validation Accuracy: {accval:.2f}%")
print(f"Test Accuracy: {acctest:.2f}%")
print(f"Validation Confusion Matrix: TN={tnval:.2f}%, FP={fpval:.2f}%, FN={fnval:.2f}%, TP={tpval:.2f}%")
print(f"Test Confusion Matrix: TN={tntest:.2f}%, FP={fptest:.2f}%, FN={fntest:.2f}%, TP={tptest:.2f}%")