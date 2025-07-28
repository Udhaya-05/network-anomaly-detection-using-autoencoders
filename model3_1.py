import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def load_kdd_data(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    data_start = False
    data = []
    for line in lines:
        line = line.strip()
        if line.lower() == "@data":
            data_start = True
            continue
        if data_start:
            data.append(line)

    df = pd.DataFrame([x.split(",") for x in data])
    return df

train_file = "C:/Users/slk11/Desktop/College/Semester 6/ML/Project sid/Project/KDD/KDDTrain+.arff"
test_file = "C:/Users/slk11/Desktop/College/Semester 6/ML/Project sid/Project/KDD/KDDTest+.arff"

df_train = load_kdd_data(train_file)
df_test = load_kdd_data(test_file)

columns = [
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
df_train.columns = columns
df_test.columns = columns

for col in ["protocol_type", "service", "flag"]:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    df_test[col] = le.transform(df_test[col])

y_train = np.where(df_train["label"] == "normal", 0, 1)
y_test = np.where(df_test["label"] == "normal", 0, 1)
X_train = df_train.drop(columns=["label"])
X_test = df_test.drop(columns=["label"])

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

input_dim = X_train_scaled.shape[1]
input_layer = Input(shape=(input_dim,))

encoded = Dense(64, activation="relu")(input_layer)
encoded = Dense(32, activation="relu")(encoded)
encoded = Dense(16, activation="relu")(encoded)

decoded = Dense(32, activation="relu")(encoded)
decoded = Dense(64, activation="relu")(decoded)
decoded_output = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded_output)
autoencoder.compile(optimizer="adam", loss="mse")

autoencoder.fit(X_train_scaled, X_train_scaled,
                epochs=25, batch_size=64,
                validation_data=(X_test_scaled, X_test_scaled),
                verbose=1)

encoded_input = Input(shape=(16,))
classifier_output = Dense(16, activation="relu")(encoded_input)
classifier_output = Dense(8, activation="relu")(classifier_output)
classifier_output = Dense(1, activation="sigmoid")(classifier_output)

classifier = Model(inputs=encoded_input, outputs=classifier_output)
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

encoder_model = Model(inputs=input_layer, outputs=encoded)
X_train_encoded = encoder_model.predict(X_train_scaled)
X_test_encoded = encoder_model.predict(X_test_scaled)

classifier.fit(X_train_encoded, y_train, epochs=25, batch_size=64, validation_data=(X_test_encoded, y_test), verbose=1)
y_pred = (classifier.predict(X_test_encoded) > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
