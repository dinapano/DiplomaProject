import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix

# Define the URL for the Lingspam dataset
url = 'lingspam/messages.csv'

# Load the Lingspam dataset
def load_lingspam_data():
    try:
        dat = pd.read_csv(url, delimiter=',')
        print(f"Dataset loaded successfully with shape: {dat.shape}")
        print(dat.head())  # Print the first few rows
        return dat
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

# Load the dataset
dat = load_lingspam_data()

def cnn_model(dat):
    if dat is None:
        print("No data to process.")
        return

    # Extract input and output data
    x = dat[['subject', 'message']]  # Features are 'subject' and 'message'
    y = dat['label']  # Target is 'label'
    print(f"Feature matrix shape: {x.shape}, Target vector shape: {y.shape}")

    # Handle missing values
    x = x.fillna('')  # Replace NaN values with empty strings

    # Convert text data to numerical features using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features
    X = vectorizer.fit_transform(x['subject'] + ' ' + x['message'])  # Combine subject and message into one text

    # Convert the sparse matrix to a dense one
    X = X.toarray()

    # Normalize the data
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Pad sequences to ensure uniform input size
    max_length = X.shape[1]
    X = pad_sequences(X, maxlen=max_length)

    # Convert labels to categorical (one-hot encoding)
    y = to_categorical(y, num_classes=2)

    # Split data into training, validation, and test sets (40%, 30%, 30%)
    X_train, X_temp, Y_train, Y_temp = train_test_split(X, y, test_size=0.60, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.50, random_state=42)
    print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}, Test set shape: {X_test.shape}")

    # Create CNN model
    model = Sequential()
    model.add(Embedding(input_dim=X.shape[1], output_dim=50, input_length=max_length))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the CNN
    model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_data=(X_val, Y_val), verbose=1)

    # Predict outputs
    Yp_val = np.argmax(model.predict(X_val), axis=1)
    Yp_test = np.argmax(model.predict(X_test), axis=1)
    Y_val_labels = np.argmax(Y_val, axis=1)
    Y_test_labels = np.argmax(Y_test, axis=1)

    # Evaluate the model
    accval = 100 * accuracy_score(Y_val_labels, Yp_val)
    acctest = 100 * accuracy_score(Y_test_labels, Yp_test)

    # Check confusion matrix dimensions
    try:
        tnval, fpval, fnval, tpval = confusion_matrix(Y_val_labels, Yp_val).ravel()
        tnval = 100 * tnval / len(Y_val_labels)
        tpval = 100 * tpval / len(Y_val_labels)
        fnval = 100 * fnval / len(Y_val_labels)
        fpval = 100 * fpval / len(Y_val_labels)

        tntest, fptest, fntest, tptest = confusion_matrix(Y_test_labels, Yp_test).ravel()
        tntest = 100 * tntest / len(Y_test_labels)
        tptest = 100 * tptest / len(Y_test_labels)
        fntest = 100 * fntest / len(Y_test_labels)
        fptest = 100 * fptest / len(Y_test_labels)
    except ValueError as e:
        print(f"Confusion matrix error: {e}")
        return

    return model, accval, acctest, tnval, tpval, fnval, fpval, tntest, tptest, fntest, fptest

# Call the cnn_model function with the loaded dataset
try:
    model, accval, acctest, tnval, tpval, fnval, fpval, tntest, tptest, fntest, fptest = cnn_model(dat)
    print(f"Validation Accuracy: {accval:.2f}%")
    print(f"Test Accuracy: {acctest:.2f}%")
    print(f"Validation Confusion Matrix: TN={tnval:.2f}%, FP={fpval:.2f}%, FN={fnval:.2f}%, TP={tpval:.2f}%")
    print(f"Test Confusion Matrix: TN={tntest:.2f}%, FP={fptest:.2f}%, FN={fntest:.2f}%, TP={tptest:.2f}%")
except Exception as e:
    print(f"Error during model training or evaluation: {e}")
