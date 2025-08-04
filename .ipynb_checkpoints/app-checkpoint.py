import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

# Page title
st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0–9)")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = ImageOps.invert(image)  # MNIST is white digit on black bg
    image = image.resize((28, 28))
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for model

    # Predict
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    # Display prediction
    st.subheader(f"Predicted Digit: {predicted_digit}")
