import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer

# Load ANN (Text Model)
ann_model = load_model("ann_model.h5")
with open("ann_vectorizer.pkl", "rb") as f:
    ann_vectorizer = pickle.load(f)
with open("ann_label_encoder.pkl", "rb") as f:
    ann_label_encoder = pickle.load(f)

# Load CNN (Image Model)
cnn_model = load_model("cnn_model.h5")
cnn_labels = [
    "Actinic Keratosis", "Atopic Dermatitis", "Benign Keratosis",
    "Dermatofibroma", "Melanocytic Nevus", "Melanoma",
    "Squamous Cell Carcinoma", "Tinea Ringworm Candidiasis",
    "Vascular Lesion"
]

# Load RNN (Code Model)
rnn_model = load_model("rnn_model.h5")
with open("rnn_tokenizer.pkl", "rb") as f:
    rnn_tokenizer = pickle.load(f)
with open("rnn_label_encoder.pkl", "rb") as f:
    rnn_label_encoder = pickle.load(f)

# Function for ANN (Text Input)
def predict_text():
    user_input = input("Enter symptoms description: ")
    X_input = ann_vectorizer.transform([user_input]).toarray()
    pred = np.argmax(ann_model.predict(X_input), axis=-1)
    predicted_disease = ann_label_encoder.inverse_transform(pred)[0]
    print(f"🩺 Predicted Disease: {predicted_disease}")

# Function for CNN (Image Input)
def predict_image():
    img_path = input("Enter image file path: ")
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = np.argmax(cnn_model.predict(img_array), axis=-1)[0]
    predicted_label = cnn_labels[pred]
    print(f"📸 Predicted Skin Condition: {predicted_label}")

# Function for RNN (Medical Code Input)
def predict_code():
    user_input = input("Enter medical codes separated by spaces: ")
    X_input = rnn_tokenizer.texts_to_sequences([user_input])
    X_padded = pad_sequences(X_input, maxlen=50)
    pred = np.argmax(rnn_model.predict(X_padded), axis=-1)
    predicted_disease = rnn_label_encoder.inverse_transform(pred)[0]
    print(f"🧬 Predicted Condition: {predicted_disease}")

# Main Function
def main():
    print("\nSelect Input Type:")
    print("1️⃣ Text (Symptoms Description)")
    print("2️⃣ Image (Skin Disease Detection)")
    print("3️⃣ Code (Medical Codes)")
    
    choice = input("Enter choice (1/2/3): ").strip()
    
    if choice == "1":
        predict_text()
    elif choice == "2":
        predict_image()
    elif choice == "3":
        predict_code()
    else:
        print("❌ Invalid choice. Please enter 1, 2, or 3.")

# Run the program
if __name__ == "__main__":
    main()

