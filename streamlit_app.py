import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Function to load and process the image
def preprocess_image(image):
    image = image.resize((28, 28))  # Resize to the input size your model expects
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image) / 255.0  # Normalize the image
    image = image.flatten()  # Flatten the image to match the model's expected input shape
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to apply the model to the image
def apply_model(image, model):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0]

# Load your pre-trained .h5 model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('./task1.h5')  # Replace with your model path
    return model

# Streamlit application
st.title("Image Processing Application")

# Upload PNG file
uploaded_file = st.file_uploader("Choose a PNG file", type="png")

if uploaded_file is not None:
    # Display the input image
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Input Image", use_column_width=True)

    # Load the model
    model = load_model()

    # Button to run the backend image processing
    if st.button("Process Image"):
        # Apply the model to the input image
        output_image = apply_model(input_image, model)
        
        # Assuming your model outputs an image, convert the prediction to an image
        output_image = Image.fromarray((output_image * 255).astype('uint8'))

        # Display both input and output images side by side
        col1, col2 = st.columns(2)

        with col1:
            st.image(input_image, caption="Input Image", use_column_width=True)
        
        with col2:
            st.image(output_image, caption="Output Image", use_column_width=True)
