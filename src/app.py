import streamlit as st
import torch
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image
import os

# Define class names
CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']

# Define transforms
def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

# Load trained model
@st.cache_resource
def load_model(model_path):
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Predict function
def predict_image(image, model):
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    predicted_class = CLASS_NAMES[predicted.item()]
    confidence = probabilities[predicted.item()].item() * 100

    top3_prob, top3_catid = torch.topk(probabilities, 3)
    top_predictions = [
        {
            'class': CLASS_NAMES[top3_catid[i].item()],
            'probability': top3_prob[i].item() * 100
        }
        for i in range(3)
    ]

    return predicted_class, confidence, top_predictions

# Streamlit UI
st.title("🌿 Plant Disease Detection")

model_path = st.sidebar.text_input("Model Path", "models/best_model.pth")

uploaded_file = st.file_uploader("Upload a leaf image", type=['jpg', 'jpeg', 'png'])

if uploaded_file and os.path.exists(model_path):
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model(model_path)
    predicted_class, confidence, top_preds = predict_image(image, model)

    st.subheader("Prediction")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.subheader("Top 3 Predictions")
    for pred in top_preds:
        st.write(f"{pred['class']} - {pred['probability']:.2f}%")

elif uploaded_file:
    st.error("Model path is invalid. Please enter a valid path.")
