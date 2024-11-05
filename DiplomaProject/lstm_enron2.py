import os
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

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
X = vectorizer.fit_transform(all_emails).toarray()
y = np.array(all_labels)

# Shuffle the data
randomnum = np.random.permutation(len(y))
xrand = X[randomnum, :]
yrand = y[randomnum]

# Convert labels to categorical (one-hot encoding)
ynew = to_categorical(yrand, num_classes=2)

# Split data into training, validation, and test sets (40%, 30%, 30%)
X_train, X_temp, Y_train, Y_temp = train_test_split(xrand, ynew, test_size=0.60, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=42)

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Reshape data for LSTM layer (assuming each input is a sequence of length 1000)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Create LSTM model
model = Sequential()
model.add(LSTM(units=64, input_shape=(1000, 1)))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the LSTM
model.fit(X_train, Y_train, epochs=50, batch_size=10, validation_data=(X_val, Y_val), verbose=1)

# Predict outputs
Yp_val = np.argmax(model.predict(X_val), axis=1)
Yp_test = np.argmax(model.predict(X_test), axis=1)
Y_val_labels = np.argmax(Y_val, axis=1)
Y_test_labels = np.argmax(Y_test, axis=1)

# Confusion matrix for validation data
accval = 100 * np.mean(Yp_val == Y_val_labels)
tnval = 100 * np.mean((Yp_val == 0) & (Y_val_labels == 0))
tpval = 100 * np.mean((Yp_val == 1) & (Y_val_labels == 1))
fnval = 100 * np.mean((Yp_val == 0) & (Y_val_labels == 1))
fpval = 100 * np.mean((Yp_val == 1) & (Y_val_labels == 0))

# Confusion matrix for test data
acctest = 100 * np.mean(Yp_test == Y_test_labels)
tntest = 100 * np.mean((Yp_test == 0) & (Y_test_labels == 0))
tptest = 100 * np.mean((Yp_test == 1) & (Y_test_labels == 1))
fntest = 100 * np.mean((Yp_test == 0) & (Y_test_labels == 1))
fptest = 100 * np.mean((Yp_test == 1) & (Y_test_labels == 0))

# Output the results
print(f"Validation Accuracy: {accval:.2f}%")
print(f"Test Accuracy: {acctest:.2f}%")
print(f"Validation Confusion Matrix: TN={tnval:.2f}%, FP={fpval:.2f}%, FN={fnval:.2f}%, TP={tpval:.2f}%")
print(f"Test Confusion Matrix: TN={tntest:.2f}%, FP={fptest:.2f}%, FN={fntest:.2f}%, TP={tptest:.2f}%")
print("\nClassification Report (Validation):")
print(classification_report(Y_val_labels, Yp_val, target_names=["Not Spam", "Spam"], zero_division=1))
print("\nClassification Report (Test):")
print(classification_report(Y_test_labels, Yp_test, target_names=["Not Spam", "Spam"], zero_division=1))
