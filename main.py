import streamlit as st
from model_helper import predict
from PIL import Image


st.title("Fruit Freshness Detection")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded File", width=300)

    # Save temporarily if needed
    image_path = "temp_file.jpg"
    image.save(image_path)

    prediction = predict(image_path)
    st.info(f"Predicted Class: {prediction}")
