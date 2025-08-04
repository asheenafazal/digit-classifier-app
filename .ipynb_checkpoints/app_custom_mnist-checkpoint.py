import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model("custom_mnist_model.h5")

# Set page title
st.set_page_config(page_title="Custom Digit Classifier", page_icon="✍️")

st.title("🧠 Handwritten Digit Classifier (Real Images)")
st.markdown("Upload a **handwritten digit image** (pen on paper), and the model will predict it!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = ImageOps.invert(image)  # Invert: black background, white digit
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img)
    predicted_digit = np.argmax(prediction)
    confidence = prediction[0][predicted_digit]

    st.success(f"**Predicted Digit: {predicted_digit}**")
    st.info(f"Confidence: {confidence:.2%}")

    # Show all class probabilities
    st.subheader("📊 All Class Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"Digit {i}: {prob:.2%}")
