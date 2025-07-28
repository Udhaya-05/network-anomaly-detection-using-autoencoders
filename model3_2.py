import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE

# 1️⃣ Randomly sample 250,000 rows from file
filename = "D:/MU-IoT.csv"
with open(filename, 'r') as f:
    total_lines = sum(1 for line in f) - 1  # exclude header

sample_size = 250000
if total_lines < sample_size:
    raise ValueError(f"CSV has only {total_lines} rows, less than requested {sample_size}")

skip_lines = sorted(random.sample(range(1, total_lines + 1), total_lines - sample_size))
df = pd.read_csv(filename, skiprows=skip_lines)

# 2️⃣ Clean and preprocess
df.dropna(axis=1, how='all', inplace=True)
df = df[df['label'].notna()]

# Encode labels: all labels to numeric (multi-class handling)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'].str.lower().astype(str))

print("Label distribution:\n", df['label'].value_counts())

y = df['label'].astype(float)
X = df.drop(columns=['label'])

# Encode categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

# Fill and normalize
X.fillna(0, inplace=True)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 3️⃣ Train/test split + SMOTE
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Optional class weights (useful for multi-class without SMOTE)
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(zip(np.unique(y_train), class_weights))

# 4️⃣ Autoencoder
input_dim = X_train_res.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(128, activation="relu")(input_layer)
encoded = Dense(64, activation="relu")(encoded)
encoded = Dense(32, activation="relu")(encoded)

decoded = Dense(64, activation="relu")(encoded)
decoded = Dense(128, activation="relu")(decoded)
decoded_output = Dense(input_dim, activation="sigmoid")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded_output)
autoencoder.compile(optimizer="adam", loss="mse")

early_stop_ae = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

autoencoder.fit(
    X_train_res, X_train_res,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, X_test),
    callbacks=[early_stop_ae],
    verbose=1
)

# 5️⃣ Encoded features
encoder_model = Model(inputs=input_layer, outputs=encoded)
X_train_encoded = encoder_model.predict(X_train_res)
X_test_encoded = encoder_model.predict(X_test)

# 6️⃣ Classifier
num_classes = len(np.unique(y))  # multi-class support

encoded_input = Input(shape=(32,))
x = Dense(16, activation="relu")(encoded_input)
x = Dropout(0.3)(x)
x = Dense(8, activation="relu")(x)
classifier_output = Dense(num_classes, activation="softmax")(x)

classifier = Model(inputs=encoded_input, outputs=classifier_output)
classifier.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

early_stop_clf = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

classifier.fit(
    X_train_encoded, y_train_res,
    epochs=50,
    batch_size=64,
    validation_data=(X_test_encoded, y_test),
    callbacks=[early_stop_clf],
    verbose=1
    # If not using SMOTE, uncomment below:
    # class_weight=class_weights_dict
)

# 7️⃣ Evaluation
y_pred = np.argmax(classifier.predict(X_test_encoded), axis=1)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 8️⃣ Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
