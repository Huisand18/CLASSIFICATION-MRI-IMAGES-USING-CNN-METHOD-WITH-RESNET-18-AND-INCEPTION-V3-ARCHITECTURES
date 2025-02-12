import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# URL Google Drive dari model (ganti dengan ID model yang benar)
url = "https://drive.google.com/uc?id=1-bHVMdUQDmPVLcd8-eQlSJMF6-zUPTFU"
model_path = "model.h5"

# Fungsi untuk mendownload model jika belum ada
@st.cache_resource
def load_trained_model():
    if not os.path.exists(model_path):
        gdown.download(url, model_path, quiet=False)
    return load_model(model_path)

# Load Model
model = load_trained_model()

# Label Kelas
class_names = ['Glioma_tumor', 'Meningioma_tumor', 'No_tumor', 'Pituitary_tumor']

# Fungsi Prediksi
def predict_image(image):
    image = image.resize((299, 299))  # Sesuaikan dengan ukuran input model
    image = np.array(image) / 255.0   # Normalisasi
    image = np.expand_dims(image, axis=0)  # Tambah dimensi batch

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Ambil indeks kelas tertinggi
    confidence = np.max(prediction)  # Probabilitas tertinggi

    return class_names[predicted_class], confidence

# Streamlit UI
st.title("Klasifikasi Gambar dengan InceptionV3")
st.write("Upload gambar untuk diklasifikasikan oleh model.")

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    if st.button("Prediksi"):
        label, conf = predict_image(image)
        st.success(f"Hasil Prediksi: **{label}** dengan confidence **{conf:.2f}**")
