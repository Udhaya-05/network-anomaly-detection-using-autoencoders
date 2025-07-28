import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- 1️⃣ Load MU-IoT Dataset --------------------
filename = "D:/MU-IoT.csv"  # 🔁 Update with actual path

# Load a small part of the dataset to inspect structure
sample_df = pd.read_csv(filename, nrows=100)

# Display the first few rows to inspect the data
print(sample_df.head())

# -------------------- 2️⃣ Clean and Preprocess Data --------------------
# Clean column names (strip spaces, remove special characters if needed)
sample_df.columns = sample_df.columns.str.strip()

# Print the column names to check if the 'label' column exists
print("Columns in the dataset:", sample_df.columns)

# Identify non-numeric columns
non_numeric_cols = sample_df.select_dtypes(exclude=[np.number]).columns.tolist()

# Convert non-numeric columns to NaN or handle them accordingly
# For now, we'll drop non-numeric columns to focus on numeric data
df = pd.read_csv(filename, low_memory=False)
df = df.drop(columns=non_numeric_cols)

# Handle missing data (e.g., fill with mean, median, or drop)
df = df.fillna(df.mean())  # You can also use df.dropna() to drop rows with NaN values

# -------------------- 3️⃣ Preprocessing --------------------

# Verify the label column name
# Print the columns again and check the actual name for the 'label' column
print("Columns in the full dataset:", df.columns)

# Assuming 'label' column is present, if not, update with correct column name.
y = df['label'].values  # Ensure that 'label' is the correct column name
X = df.drop(columns=['label'])

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)

# Balance training data (equal normal & anomaly)
normal_idx = np.where(y_train == 0)[0]
anomaly_idx = np.where(y_train == 1)[0]
np.random.shuffle(normal_idx)
normal_idx = normal_idx[:len(anomaly_idx)]
balanced_idx = np.concatenate([normal_idx, anomaly_idx])
np.random.shuffle(balanced_idx)
X_train = X_train[balanced_idx]
y_train = y_train[balanced_idx]

# Add Gaussian noise
noise_factor = 0.1
X_train_noisy = np.clip(X_train + noise_factor * np.random.normal(0, 1, X_train.shape), 0., 1.)

# -------------------- 4️⃣ Build Autoencoder --------------------
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))
x = Dense(128, activation="selu")(input_layer)
x = LeakyReLU(0.1)(x)
x = Dense(64, activation="selu")(x)
x = LeakyReLU(0.1)(x)
x = Dense(32, activation="selu")(x)
x = LeakyReLU(0.1)(x)
x = Dense(16, activation="selu")(x)
x = LeakyReLU(0.1)(x)
encoded = Dense(8, activation="selu")(x)
x = Dense(16, activation="relu")(encoded)
x = Dense(32, activation="relu")(x)
x = Dense(64, activation="relu")(x)
decoded = Dense(input_dim, activation="sigmoid")(x)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# -------------------- 5️⃣ Train --------------------
history = autoencoder.fit(X_train_noisy, X_train, epochs=25, batch_size=64, validation_data=(X_test, X_test), verbose=1)

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title("Loss Curve"); plt.show()

# -------------------- 6️⃣ Anomaly Detection --------------------
X_train_pred = autoencoder.predict(X_train)
X_test_pred = autoencoder.predict(X_test)
train_errors = np.mean((X_train - X_train_pred) ** 2, axis=1)
test_errors = np.mean((X_test - X_test_pred) ** 2, axis=1)

# Threshold using IQR
Q1, Q3 = np.percentile(train_errors, 25), np.percentile(train_errors, 75)
IQR = Q3 - Q1
threshold = Q3 + 2.0 * IQR

# Predict
y_train_pred = (train_errors > threshold).astype(int)
y_test_pred = (test_errors > threshold).astype(int)

# -------------------- 7️⃣ Evaluation --------------------
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

cm = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.show()

# -------------------- 8️⃣ Hybrid Detection (Isolation Forest) --------------------
iso = IsolationForest(contamination=0.01, random_state=42)
iso.fit(X_train)
iso_preds = iso.predict(X_test)
iso_preds = np.where(iso_preds == -1, 1, 0)

print("\nIsolation Forest Accuracy:", accuracy_score(y_test, iso_preds))
print("Isolation Forest Report:\n", classification_report(y_test, iso_preds))
