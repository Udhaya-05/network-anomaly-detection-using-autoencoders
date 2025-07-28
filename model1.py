import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.regularizers import l1
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

# Step 1: Load Dataset
def load_dataset():
    # Using a sample NSL-KDD dataset (or replace with your network dataset)
    df = pd.read_csv("MU-IoT.csv", header=None)
    df = df.iloc[:, :-1]  # Removing labels column if present
    return df

# Step 2: Data Preprocessing
def preprocess_data(df):
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    return df_scaled, scaler

# Step 3: Define Sparse Deep Denoising Autoencoder (SDDA)
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    
    # Encoder with sparsity constraint and dropout for denoising
    encoded = Dense(128, activation="relu", activity_regularizer=l1(0.001))(input_layer)
    encoded = Dropout(0.2)(encoded)  # Adding dropout to prevent overfitting
    encoded = Dense(64, activation="relu")(encoded)
    encoded = Dense(32, activation="relu")(encoded)

    # Decoder
    decoded = Dense(64, activation="relu")(encoded)
    decoded = Dense(128, activation="relu")(decoded)
    decoded = Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    return autoencoder

# Step 4: Train Model
def train_autoencoder(autoencoder, X_train):
    history = autoencoder.fit(X_train, X_train, 
                              epochs=50, 
                              batch_size=64, 
                              validation_split=0.1, 
                              verbose=1)
    return history

# Step 5: Evaluate Anomaly Detection
def detect_anomalies(autoencoder, X_test):
    reconstructions = autoencoder.predict(X_test)
    errors = np.mean(np.abs(reconstructions - X_test), axis=1)
    return errors

# Step 6: Main Execution
if __name__ == "__main__":
    df = load_dataset()
    X, scaler = preprocess_data(df)
    
    autoencoder = build_autoencoder(input_dim=X.shape[1])
    train_autoencoder(autoencoder, X)
    
    # Detect anomalies
    anomaly_scores = detect_anomalies(autoencoder, X)
    
    # Plot anomaly scores
    plt.hist(anomaly_scores, bins=50)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Anomaly Scores Distribution")
    plt.show()
