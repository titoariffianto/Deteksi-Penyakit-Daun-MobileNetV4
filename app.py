import streamlit as st
from PIL import Image
import torch
import timm
import torch.nn as nn
import torchvision.transforms as transforms


model_path = 'mobilenetv4_plant_disease_model.pth'

class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]
num_classes = len(class_names)
image_size = 224

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@st.cache_resource
def load_model(path, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = timm.create_model('mobilenetv4_conv_small.e2400_r224_in1k', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)

model = load_model(model_path, num_classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict_image(image, model, transform, class_names, device):
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_idx].item()

    predicted_class = class_names[predicted_idx]
    return predicted_class, confidence


st.title("Aplikasi Deteksi Penyakit Tanaman")

st.write("Unggah gambar daun tanaman untuk mendeteksi potensi penyakit.")

uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    if st.button("Deteksi Penyakit"):
        with st.spinner('Mendeteksi...'):
            predicted_class, confidence = predict_image(image, model, val_transform, class_names, device)
            st.success("Deteksi Selesai!")
            st.write(f"**Penyakit yang Diprediksi:** {predicted_class}")
            st.write(f"**Tingkat Kepercayaan:** {confidence:.2%}")

else:
    st.info("Silakan unggah gambar untuk memulai.")