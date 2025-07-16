!pip install tensorflow
!pip install PIL
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page title and favicon
st.set_page_config(
    page_title="Brain Tumor MRI Image Classification",
    page_icon="🧠",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = 'my_trained_effnet_model.keras' # Assuming EfficientNet is the best model
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the model is saved correctly.")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Define image size and class names
IMG_HEIGHT = 224
IMG_WIDTH = 224
CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary'] # Ensure this matches your training

# Custom normalization function (if used during training)
def medical_normalization(img):
    """Custom normalization for MRI images"""
    img = img / 255.
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip(img, p1, p99)
    mean = np.mean(img)
    std = np.std(img)
    if std > 0:
      img = (img - mean) / std
    else:
      img = img - mean
    return img

# Prediction function
def predict(image):
    if model is None:
        return None

    # Preprocess the image
    img = Image.open(image).convert('RGB') # Ensure image is RGB
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img)
    # Apply normalization if needed - adjust based on your training preprocessing
    # img_array = medical_normalization(img_array) # Uncomment if you used this
    img_array = img_array / 255.0  # Basic rescaling as in ImageDataGenerator
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension

    # Make prediction
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return {
        'class': CLASS_NAMES[np.argmax(score)],
        'confidence': np.max(score)
    }

# Streamlit UI
st.title("🧠 Brain Tumor MRI Image Classification")

st.write("Upload an MRI image to classify it into one of four categories: Glioma, Meningioma, Pituitary, or No Tumor.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded MRI Image', use_column_width=True)

    if st.button('Classify'):
        with st.spinner('Classifying...'):
            prediction_result = predict(uploaded_file)
            if prediction_result:
                st.subheader("Prediction:")
                st.write(f"**Class:** {prediction_result['class']}")
                st.write(f"**Confidence:** {prediction_result['confidence']:.2%}")
                st.success("Classification Complete!")
            else:
                st.error("Could not perform classification. Please check the model and try again.")

st.markdown("---")
st.write("This is a demo application. For clinical use, consult with a medical professional.")
