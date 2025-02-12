import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

url = "https://drive.google.com/uc?id=1-bHVMdUQDmPVLcd8-eQlSJMF6-zUPTFU"
model_path = "model.h5"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)
    return load_model(model_path)

model = load_trained_model()

class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

def predict_image(image):
    image = image.resize((299, 299)) 
    image = np.array(image) / 255.0   
    image = np.expand_dims(image, axis=0) 

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  
    confidence = np.max(prediction) 

    return class_names[predicted_class], confidence

st.title("Klasifikasi Tumor Otak dengan InceptionV3")
st.write("Upload citra MRI untuk diklasifikasikan oleh model.")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", width=300)

    if st.button("Prediksi"):
        label, conf = predict_image(image)
        st.success(f"Hasil Prediksi: **{label}** dengan confidence **{conf:.2f}**")
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: grey;
        text-align: left;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        @ <b>Frederick Huisand S</b>
    </div>
    """,
    unsafe_allow_html=True
)
