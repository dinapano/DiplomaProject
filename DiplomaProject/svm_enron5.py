import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from email.parser import Parser
from sklearn.metrics import classification_report

# Function to read raw email files and extract content and label (spam or ham)
def load_enron5_data(spam_dir, ham_dir):
    emails = []
    labels = []

    for filename in os.listdir(spam_dir):
        with open(os.path.join(spam_dir, filename), 'r', encoding='latin-1') as file:
            content = file.read()
            emails.append(content)
            labels.append(1)  # Spam

    for filename in os.listdir(ham_dir):
        with open(os.path.join(ham_dir, filename), 'r', encoding='latin-1') as file:
            content = file.read()
            emails.append(content)
            labels.append(0)  # Ham

    return emails, np.array(labels)

# Preprocess the emails (you need to provide the correct paths to spam and ham directories)
spam_dir = 'enron5/spam/'
ham_dir = 'enron5/ham/'
emails, labels = load_enron5_data(spam_dir, ham_dir)

# Convert emails to a matrix of token counts (features)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

# Convert the sparse matrix to a dense one if necessary
X = X.toarray()

# Normalize the data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split data into training, validation, and test sets (40%, 30%, 30%)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, labels, test_size=0.60, random_state=42)
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

# Output the results
print(f"Validation Accuracy: {accval:.2f}%")
print(f"Test Accuracy: {acctest:.2f}%")
print(f"Validation Confusion Matrix: TN={tnval:.2f}%, FP={fpval:.2f}%, FN={fnval:.2f}%, TP={tpval:.2f}%")
print(f"Test Confusion Matrix: TN={tntest:.2f}%, FP={fptest:.2f}%, FN={fntest:.2f}%, TP={tptest:.2f}%")
print("\nClassification Report (Validation):")
print(classification_report(Y_val, Yp_val, target_names=["Not Spam", "Spam"]))
print("\nClassification Report (Test):")
print(classification_report(Y_test, Yp_test, target_names=["Not Spam", "Spam"]))
