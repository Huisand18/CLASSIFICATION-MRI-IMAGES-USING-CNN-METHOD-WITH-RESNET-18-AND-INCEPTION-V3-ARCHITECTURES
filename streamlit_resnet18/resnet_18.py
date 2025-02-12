import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import gdown

url = "https://drive.google.com/uc?id=1J4Vsp_s9sg9Ii3_ioDExDo_ST3LjVYeZ"
model_path = "model_resnet18_state_dict.pth"

def download_model():
    if not os.path.exists(model_path):
        st.info("🔄 Mengunduh model, harap tunggu...")
        gdown.download(url, model_path, quiet=False)

class CustomResNet18(nn.Module):
    def __init__(self, num_classes=4):  
        super(CustomResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

def load_model():
    download_model()
    model = CustomResNet18(num_classes=4)  
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval() 
    return model

st.write("🔍 Memuat model...")
model = load_model()

class_names = ['Glioma_tumor', 'Meningioma_tumor', 'No_tumor', 'Pituitary_tumor']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
    image = transform(image).unsqueeze(0)  
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return class_names[predicted.item()], confidence.item()

st.title("🧠 Klasifikasi MRI Tumor Otak dengan ResNet18")
st.write("Unggah gambar MRI untuk diklasifikasikan.")

uploaded_file = st.file_uploader("📤 Upload Gambar", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Gambar yang diunggah", width=300)

    if st.button("Prediksi 🔍"):
        label, conf = predict_image(image)
        st.success(f"✅ Hasil Prediksi: **{label}**")
        st.info(f"📊 Confidence: **{conf:.2f}**")

st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 10px;
            left: 10px;
            font-size: 12px;
            color: gray;
        </style>
    <div class="footer">@ <b>Frederick Huisand S</b>
    </div>""",
    unsafe_allow_html=True
)
