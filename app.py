import streamlit as st
from PIL import Image
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# --- Page Config ---
st.set_page_config(page_title="Cat & Dog Classifier", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #ff6b35 0%, #f7931e 50%, #ff9068 100%);
    }
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 1rem;
        max-height: 80vh;
    }
    .stApp > header {
        background-color: transparent;
    }
    h1 {
        margin-top: -2rem !important;
        padding-top: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Model Loading ---
@st.cache_resource
def load_models():
    """Loads the VGG16 feature extractor and the saved SVM model."""
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
    svm_model = joblib.load('svm_model.pkl')
    return feature_extractor, svm_model

feature_extractor, svm_model = load_models()

# --- Image Preprocessing and Prediction Function ---
def predict(image_to_process):
    """Preprocesses an image and returns the prediction."""
    img = image_to_process.resize((224, 224))
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor = preprocess_input(img_tensor)
    
    vgg16_features = feature_extractor.predict(img_tensor, verbose=0)
    prediction = svm_model.predict(vgg16_features.flatten().reshape(1, -1))
    
    return prediction[0]

# --- UI Layout ---
st.title("🐱 Cat vs. Dog Image Classifier 🐶")
st.markdown("---")
st.markdown("""
**This app uses a Support Vector Machine (SVM) to classify images of cats and dogs. 
The features for the SVM are extracted using the powerful **VGG16** deep learning model.**
""")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload an Image")
    uploaded_file = st.file_uploader("Choose an image of a cat or a dog...", type=["jpg", "jpeg", "png"])

with col2:
    st.header("Prediction")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Your Uploaded Image.', width=300)
        
        with st.spinner('Analyzing the image...'):
            prediction_result = predict(image)
        
        st.markdown("---")
        if prediction_result == 0:
            st.markdown("""
            <div style='background: rgba(255, 120, 0, 0.9); padding: 1rem; border-radius: 10px; margin-top: -5rem; border: 2px solid rgba(255, 140, 0, 1);'>
                <h3 style='color: white; margin: 0; text-align: center;'>Prediction: It's a Cat! 🐱</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background: rgba(255, 120, 0, 0.9); padding: 1rem; border-radius: 10px; margin-top: -5rem; border: 2px solid rgba(255, 140, 0, 1);'>
                <h3 style='color: white; margin: 0; text-align: center;'>Prediction: It's a Dog! 🐶</h3>
            </div>
            """, unsafe_allow_html=True)