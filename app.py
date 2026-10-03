import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load the trained model
model = load_model("brain_tumor_model.h5")

# Define class labels
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

st.title("Brain Tumor Classifier")
st.write("Upload a brain scan image to predict the type of tumor.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image, (224, 224))  # VGG16 input size
    image_array = np.expand_dims(image_resized, axis=0)
    image_preprocessed = preprocess_input(image_array)

    # Predict
    prediction = model.predict(image_preprocessed)
    predicted_class = class_labels[np.argmax(prediction)]

    st.image(
    image,
    channels="BGR",
    caption="Uploaded Image",
    use_container_width=True
    )
    st.success(f"Predicted Tumor Type: **{predicted_class}**")
