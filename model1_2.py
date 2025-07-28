import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Step 1: Load Dataset from KDD Folder
def load_dataset():
    train_path = "KDD/KDDTrain+.txt"
    test_path = "KDD/KDDTest+.txt"

    # Load dataset (using "," as delimiter for KDD)
    df_train = pd.read_csv(train_path, header=None, delimiter=",", low_memory=False)
    df_test = pd.read_csv(test_path, header=None, delimiter=",", low_memory=False)

    # Remove label column (last column)
    df_train = df_train.iloc[:, :-1]
    df_test = df_test.iloc[:, :-1]

    return df_train, df_test

# Step 2: Data Preprocessing (Fixing NaN and Categorical Data)
def preprocess_data(df_train, df_test):
    categorical_columns = [1, 2, 3]  # Indices for 'protocol_type', 'service', 'flag'

    # Convert categorical features using Label Encoding
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df_train.iloc[:, col] = le.fit_transform(df_train.iloc[:, col].astype(str))
        df_test.iloc[:, col] = le.transform(df_test.iloc[:, col].astype(str))  # Ensure test set uses same encoding
        label_encoders[col] = le  # Store encoders

    # Replace NaN and infinite values
    df_train.fillna(0, inplace=True)
    df_test.fillna(0, inplace=True)
    df_train.replace([np.inf, -np.inf], 0, inplace=True)
    df_test.replace([np.inf, -np.inf], 0, inplace=True)

    # Convert all data to numeric type
    df_train = df_train.apply(pd.to_numeric, errors="coerce")
    df_test = df_test.apply(pd.to_numeric, errors="coerce")

    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    df_train_scaled = scaler.fit_transform(df_train)
    df_test_scaled = scaler.transform(df_test)  # Use same scaling for test set

    return df_train_scaled, df_test_scaled, scaler

# Step 3: Define Sparse Deep Denoising Autoencoder (SDDA)
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))

    # Encoder with sparsity constraint and dropout for denoising
    encoded = Dense(128, activation="relu", activity_regularizer=l1(0.001))(input_layer)
    encoded = BatchNormalization()(encoded)  # Batch Normalization for stability
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(64, activation="relu")(encoded)
    encoded = Dense(32, activation="relu")(encoded)

    # Decoder
    decoded = Dense(64, activation="relu")(encoded)
    decoded = Dense(128, activation="relu")(decoded)
    decoded = Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mse")  # Lower learning rate

    return autoencoder

# Step 4: Train Model
def train_autoencoder(autoencoder, X_train):
    history = autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )
    return history

# Step 5: Detect Anomalies and Handle NaN Scores
def detect_anomalies(autoencoder, X_test):
    reconstructions = autoencoder.predict(X_test)
    errors = np.mean(np.abs(reconstructions - X_test), axis=1)

    errors = np.nan_to_num(errors)  # Replace NaN values with 0
    return errors

# Step 6: Main Execution
if __name__ == "__main__":
    print("Loading dataset...")
    df_train, df_test = load_dataset()

    print("Preprocessing dataset...")
    X_train, X_test, scaler = preprocess_data(df_train, df_test)

    print("Building autoencoder model...")
    autoencoder = build_autoencoder(input_dim=X_train.shape[1])

    print("Training model...")
    history = train_autoencoder(autoencoder, X_train)

    print("Detecting anomalies...")
    anomaly_scores = detect_anomalies(autoencoder, X_test)

    print("Plotting anomaly score distribution...")
    plt.hist(anomaly_scores, bins=50)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Anomaly Scores Distribution")
    plt.show()

    print("Execution completed successfully!")
