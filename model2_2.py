import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# -------------------- 1️⃣ Load Data --------------------
def load_kdd_data(file_path):
    """Manually read ARFF file and convert to pandas DataFrame."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    # 🔄 Extract only the data section (skip metadata)
    data_start = False
    data = []
    for line in lines:
        line = line.strip()
        if line.lower() == "@data":
            data_start = True
            continue
        if data_start:
            data.append(line)

    # 🔄 Convert data into a Pandas DataFrame
    df = pd.DataFrame([x.split(",") for x in data])

    return df

# ✅ Explicit File Paths
train_file = "C:/Users/91914/Desktop/Siddhesh/College/YR 3/SEM 6/ML/Project/KDD/KDDTrain+.arff"
test_file = "C:/Users/91914/Desktop/Siddhesh/College/YR 3/SEM 6/ML/Project/KDD/KDDTest+.arff"

df_train = load_kdd_data(train_file)
df_test = load_kdd_data(test_file)

# -------------------- 2️⃣ Data Preprocessing --------------------

# Set column names (42 features + 1 label)
column_names = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", 
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations", 
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", 
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", 
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", 
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", 
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"
]

df_train.columns = column_names
df_test.columns = column_names

# Define categorical columns
categorical_cols = ["protocol_type", "service", "flag"]

# Encode categorical variables
encoder = LabelEncoder()
for col in categorical_cols:
    df_train[col] = encoder.fit_transform(df_train[col])
    df_test[col] = encoder.transform(df_test[col])

# Get the labels (1 = anomaly, 0 = normal)
y_train = np.where(df_train["label"] == "normal", 0, 1)
y_test = np.where(df_test["label"] == "normal", 0, 1)

# Normalize data (Min-Max Scaling)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(df_train.drop(columns=["label"]))
X_test = scaler.transform(df_test.drop(columns=["label"]))

# Balance the dataset by undersampling normal data
normal_indices = np.where(y_train == 0)[0]
anomaly_indices = np.where(y_train == 1)[0]
np.random.shuffle(normal_indices)
normal_indices = normal_indices[:len(anomaly_indices)]  # Match anomaly count
balanced_indices = np.concatenate([normal_indices, anomaly_indices])
np.random.shuffle(balanced_indices)
X_train = X_train[balanced_indices]
y_train = y_train[balanced_indices]

# Add Gaussian noise to training data
noise_factor = 0.1
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)  # Ensure values stay in [0,1]

# -------------------- 3️⃣ Build Autoencoder Model --------------------
input_dim = X_train.shape[1]

input_layer = Input(shape=(input_dim,))
from tensorflow.keras.layers import LeakyReLU

encoded = Dense(128, activation="selu")(input_layer)
encoded = LeakyReLU(negative_slope=0.1)(encoded)
encoded = Dense(64, activation="selu")(encoded)
encoded = LeakyReLU(negative_slope=0.1)(encoded)
encoded = LeakyReLU(negative_slope=0.1)(encoded)
encoded = Dense(32, activation="selu")(encoded)
encoded = LeakyReLU(negative_slope=0.1)(encoded)
encoded = Dense(16, activation="selu")(encoded)
encoded = LeakyReLU(negative_slope=0.1)(encoded)
encoded = Dense(8, activation="selu")(encoded)
encoded = LeakyReLU(negative_slope=0.1)(encoded)

decoded = Dense(16, activation="relu")(encoded)
decoded = Dense(32, activation="relu")(decoded)
decoded = Dense(64, activation="relu")(decoded)
decoded = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# -------------------- 4️⃣ Train the Model (75 epochs) & Track Loss --------------------
import matplotlib.pyplot as plt

history = autoencoder.fit(X_train_noisy, X_train, epochs=75, batch_size=64, validation_data=(X_test, X_test), verbose=1)

# Plot Training & Validation Loss
plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.show()
# -------------------- 5️⃣ Detect Anomalies & Visualizations --------------------
import seaborn as sns

X_train_pred = autoencoder.predict(X_train)
X_test_pred = autoencoder.predict(X_test)

# Compute reconstruction error (MSE)
train_errors = np.mean(np.square(X_train - X_train_pred), axis=1)
test_errors = np.mean(np.square(X_test - X_test_pred), axis=1)

# Plot Reconstruction Error Distribution
plt.figure(figsize=(8,5))
sns.histplot(test_errors, bins=50, kde=True)
plt.axvline(threshold, color='r', linestyle='dashed', linewidth=2, label='Threshold')
plt.xlabel('Reconstruction Error')
plt.title('Reconstruction Error Distribution')
plt.legend()
plt.show()
X_train_pred = autoencoder.predict(X_train)
X_test_pred = autoencoder.predict(X_test)

# Compute reconstruction error (MSE)
train_errors = np.mean(np.square(X_train - X_train_pred), axis=1)
test_errors = np.mean(np.square(X_test - X_test_pred), axis=1)

# Set anomaly threshold (90th percentile)
Q1 = np.percentile(train_errors, 25)
Q3 = np.percentile(train_errors, 75)
IQR = Q3 - Q1
threshold = Q3 + (2.0 * IQR)

# Predict anomalies
y_train_pred = (train_errors > threshold).astype(int)
y_test_pred = (test_errors > threshold).astype(int)

# -------------------- 5️⃣b Hybrid Anomaly Detection (Isolation Forest) --------------------
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X_train)
iso_preds = iso_forest.predict(X_test)
iso_preds = np.where(iso_preds == -1, 1, 0)  # Convert to 1 (anomaly) or 0 (normal)

# -------------------- 6️⃣ Model Evaluation & Confusion Matrix --------------------
from sklearn.metrics import confusion_matrix
import seaborn as sns

print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Classification Report:", classification_report(y_test, y_test_pred))

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
from sklearn.metrics import classification_report, accuracy_score

print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))
