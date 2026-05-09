import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("covid_model.h5")

classes = ['Covid', 'Normal', 'Viral Pneumonia']

st.title("COVID-19 Detection App")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, 1)

    img = cv2.resize(img, (128,128))

    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    predicted_class = classes[np.argmax(prediction)]

    st.image(uploaded_file)

    st.success(f"Prediction: {predicted_class}")