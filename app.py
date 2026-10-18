import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 X-Ray Based Pneumonia Detection")
st.write("Upload a chest X-ray image to predict NORMAL or PNEUMONIA.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model/xray_classifier.pth"


class XRayClassificationModel(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


@st.cache_resource
def load_model():
    model = XRayClassificationModel().to(DEVICE)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

uploaded_file = st.file_uploader(
    "Choose a chest X-ray image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-Ray", width=400)

    if st.button("Predict", type="primary", use_container_width=True):
        model = load_model()
        input_tensor = transform(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_index = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, predicted_index].item() * 100

        predicted_class = CLASS_NAMES[predicted_index]

        st.subheader("Prediction Result")

        if predicted_class == "PNEUMONIA":
            st.error("PNEUMONIA")
        else:
            st.success("NORMAL")

        st.write(f"Confidence: {confidence:.2f}%")
