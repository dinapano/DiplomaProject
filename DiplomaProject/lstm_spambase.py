import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Load the Spambase dataset from the UCI repository
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
dat = np.loadtxt(url, delimiter=',')

def lstm_model(dat):
    # Extract input and output data
    x = dat[:, :57]  # Features: columns 0 to 56
    y = dat[:, 57]  # Labels: last column (column 57)

    # Shuffle the data
    randomnum = np.random.permutation(len(y))
    xrand = x[randomnum, :]
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

    # Reshape data for LSTM layer
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # (samples, timesteps, features)
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # Create LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(1, 57), return_sequences=False))  # LSTM layer
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

    return model, accval, acctest, tnval, tpval, fnval, fpval, tntest, tptest, fntest, fptest

# Run the model
model, accval, acctest, tnval, tpval, fnval, fpval, tntest, tptest, fntest, fptest = lstm_model(dat)

# Output the results
print(f"Validation Accuracy: {accval:.2f}%")
print(f"Test Accuracy: {acctest:.2f}%")
print(f"Validation Confusion Matrix: TN={tnval:.2f}%, FP={fpval:.2f}%, FN={fnval:.2f}%, TP={tpval:.2f}%")
print(f"Test Confusion Matrix: TN={tntest:.2f}%, FP={fptest:.2f}%, FN={fntest:.2f}%, TP={tptest:.2f}%")
