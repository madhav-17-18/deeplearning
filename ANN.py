import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
file_path = "C:/Users/saima/OneDrive/Desktop/dl-project/ANN/Skin_text_classifier.csv"  # Update as needed
ann_df = pd.read_csv(file_path)

# Correct column name
ann_df.rename(columns={'Disease name': 'disease'}, inplace=True)

# Encode categorical labels
label_encoder = LabelEncoder()
ann_df['disease'] = label_encoder.fit_transform(ann_df['disease'])

# Convert text data to numerical features
vectorizer = TfidfVectorizer(max_features=500)
X_transformed = vectorizer.fit_transform(ann_df['Text']).toarray()

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_transformed)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, ann_df['disease'], test_size=0.2, random_state=42)

# ANN Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# Save model
model.save("ann_model.h5")
print("✅ ANN Model saved as ann_model.h5")

# Save the vectorizer
with open("ann_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save the label encoder
with open("ann_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Vectorizer and Label Encoder saved!")







