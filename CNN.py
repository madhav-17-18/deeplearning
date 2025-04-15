#Image-Based Classification
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# Dataset path (update based on your dataset)
dataset_path = "C:/Users/saima/OneDrive/Desktop/dl-project/CNN/val/"

# Preprocessing images
image_size = (128, 128)
batch_size = 32

datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

train_data = datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='training')
val_data = datagen.flow_from_directory(dataset_path, target_size=image_size, batch_size=batch_size, class_mode='categorical', subset='validation')

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
model.save("cnn_model.h5")
print("✅ CNN Model saved as cnn_model.h5")


