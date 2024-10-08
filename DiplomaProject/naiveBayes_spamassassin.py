import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load and preprocess the SpamAssassin corpus
def load_spamassassin_data(spam_dir, ham_dir):
    emails = []
    labels = []

    for filename in os.listdir(spam_dir):
        with open(os.path.join(spam_dir, filename), 'r', encoding='latin1') as f:
            emails.append(f.read())
            labels.append(1)  # Spam label

    for filename in os.listdir(ham_dir):
        with open(os.path.join(ham_dir, filename), 'r', encoding='latin1') as f:
            emails.append(f.read())
            labels.append(0)  # Ham label

    return emails, labels

# Specify paths to spam and ham directories
spam_dir = 'spamassassin/spam_2/'
ham_dir = 'spamassassin/easy_ham/'

emails, labels = load_spamassassin_data(spam_dir, ham_dir)

# Step 2: Convert emails to numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)
y = np.array(labels)

# Step 3: Run the naive_bayes_model function on the processed data
def naive_bayes_model(X, y):
    # Shuffle the data
    randomnum = np.random.permutation(len(y))
    Xrand = X[randomnum]
    yrand = y[randomnum]

    # Split data into training, validation, and test sets (40%, 30%, 30%)
    X_train, X_temp, Y_train, Y_temp = train_test_split(Xrand, yrand, test_size=0.60, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=42)

    # Normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train.toarray())
    X_val = scaler.transform(X_val.toarray())
    X_test = scaler.transform(X_test.toarray())

    # Create and train the Naive Bayes model
    model = GaussianNB()
    model.fit(X_train, Y_train)

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

# Step 4: Call the function with the processed data
model, accval, acctest, tnval, tpval, fnval, fpval, tntest, tptest, fntest, fptest = naive_bayes_model(X, y)

# Print the results
print(f"Validation Accuracy: {accval:.2f}%")
print(f"Test Accuracy: {acctest:.2f}%")
print(f"Validation Confusion Matrix: TN={tnval:.2f}%, FP={fpval:.2f}%, FN={fnval:.2f}%, TP={tpval:.2f}%")
print(f"Test Confusion Matrix: TN={tntest:.2f}%, FP={fptest:.2f}%, FN={fntest:.2f}%, TP={tptest:.2f}%")
