import numpy as np
import os
import cv2
import random
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# --- Define Constants ---
DATADIR = os.path.join(os.getcwd(), 'data', 'train')
CATEGORIES = ["cats", "dogs"]
IMG_SIZE = 224 # VGG16 expects 224x224 images
print("Constants defined.")

# --- Load Pre-trained VGG16 Model ---
# We load VGG16 but remove its final classification layer (include_top=False)
# This turns the model into a powerful feature extractor.
print("Loading VGG16 model...")
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
print("✅ VGG16 model loaded.")

# --- Create the Training Data with VGG16 Features ---
training_data = []

def create_training_data():
    print("Starting to create training data with VGG16 features...")
    # Add a counter to show progress
    processed_images = 0
    total_images = sum([len(files) for r, d, files in os.walk(DATADIR)])
    
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category) # 0 for cats, 1 for dogs
        
        if not os.path.exists(path):
            print(f"Warning: Directory not found at {path}. Skipping.")
            continue
            
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                # Load the image IN COLOR, required by VGG16
                img_array = cv2.imread(img_path)
                # Resize to 224x224
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                
                # Prepare image for VGG16
                img_tensor = img_to_array(new_array)
                img_tensor = np.expand_dims(img_tensor, axis=0)
                img_tensor = preprocess_input(img_tensor)
                
                # Extract features
                vgg16_features = model.predict(img_tensor, verbose=0)
                
                training_data.append([vgg16_features.flatten(), class_num])
                
                processed_images += 1
                if processed_images % 50 == 0:
                    print(f"  Processed {processed_images}/{total_images} images...")

            except Exception as e:
                print(f"Warning: Could not process image {img_path}. Error: {e}")
                pass
    print("✅ Training data created successfully.")

create_training_data()

# --- Shuffle and Prepare Data for the Model ---
print("Shuffling data...")
random.shuffle(training_data)

X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)
print("Data prepared for model.")

# --- Save the Processed Data ---
np.save('features.npy', X)
np.save('labels.npy', y)

print("\n✅ Processed VGG16 features saved successfully!")
print("Files 'features.npy' and 'labels.npy' have been updated.")