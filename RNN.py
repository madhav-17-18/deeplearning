import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# 📂 Load the merged dataset
file_path = "C:/Users/saima/OneDrive/Desktop/dl-project/RNNF1/cleaned_dataset.csv"  # Update with the correct path if needed
df = pd.read_csv(file_path)

# 🔄 Normalize column names
df.columns = df.columns.str.strip().str.lower()

# 🗑 Handle missing values
df.fillna("unknown", inplace=True)

# ✅ Identify relevant columns (medical codes)
code_columns = [col for col in df.columns if col.startswith("code_")]

# 📌 Combine all medical codes into a single text feature
df["combined_text"] = df[code_columns].astype(str).agg(" ".join, axis=1)

# 🎯 Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df["combined_text"])
sequences = tokenizer.texts_to_sequences(df["combined_text"])
X = pad_sequences(sequences, padding='post')

# 🔄 Encode Target Labels (Using first code column as target)
y = df[code_columns[0]].astype(str)  # Ensure string format
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ✅ Save Tokenizer & Label Encoder
with open("rnn_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("rnn_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("✅ Tokenizer and Label Encoder saved!")

# 📊 Define RNN Model
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
max_length = X.shape[1]

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# ⚙️ Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 🎯 Train Model
model.fit(X, y, epochs=10, batch_size=32)

# 💾 Save Model
model.save("rnn_model.h5")
print("✅ RNN Model saved as rnn_model.h5")



