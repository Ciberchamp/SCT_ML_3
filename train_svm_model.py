import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# --- Load the Preprocessed Data ---
print("Loading preprocessed data...")
# Load the feature and label arrays we created in the previous step
X = np.load('features.npy')
y = np.load('labels.npy')
print("✅ Data loaded successfully.")

# --- Split Data for Training and Testing ---
# We'll use 75% of the data for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("✅ Data split into training and testing sets.")

# --- Initialize and Train the SVM Model ---
# We are using a Support Vector Classifier (SVC)
print("\nInitializing SVM model...")
model = SVC(kernel='rbf') # 'linear' is a good starting point for the kernel

print("Training the model... This may take a few minutes.")
# The model learns the patterns from the training data
model.fit(X_train, y_train)
print("✅ Model training complete.")

# --- Evaluate the Model ---
print("\nMaking predictions on the test set...")
predictions = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, target_names=['Cat', 'Dog'])

print("\n--- Model Evaluation Results ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(report)
print("\nSaving the trained model...")
joblib.dump(model, 'svm_model.pkl')
print("✅ Model saved as 'svm_model.pkl'")