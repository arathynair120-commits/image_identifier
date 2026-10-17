
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.cnn import CNN

st.set_page_config(
    page_title="Image Uncertainty Detector",
    page_icon="🔍",
    layout="wide"
)

CLASSES = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer"
]
THRESHOLD = 0.82

@st.cache_resource
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = CNN(num_classes=5).to(device)
    model.load_state_dict(
        torch.load("models/cnn.pth", map_location=device)
    )
    model.eval()
    return model, device

model, device = load_model()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

st.title("Image Classification & OOD Detection")
st.caption(
    "CNN-based image recognition with confidence-based "
    "uncertainty detection"
)

st.info(
    "Known classes: Airplane, Automobile, Bird, Cat and Deer. "
    "Other CIFAR-10 categories are treated as unknown during evaluation."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    left, right = st.columns(2)

    with left:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_image)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(dim=1)

    class_name = CLASSES[predicted.item()]
    confidence_value = confidence.item()
    is_known = confidence_value >= THRESHOLD

    with right:
        st.subheader("Prediction Results")

        st.metric("Predicted Class", class_name)
        st.metric("Confidence", f"{confidence_value * 100:.2f}%")

        if is_known:
            st.success("Result: KNOWN CLASS")
            st.write(
                "The model confidence is at or above the "
                "selected threshold."
            )
        else:
            st.warning("Result: POSSIBLY UNKNOWN")
            st.write(
                "The model confidence is below the selected "
                "threshold. The image may be unfamiliar."
            )

        st.progress(confidence_value)

    st.subheader("Class Confidence Scores")

    scores = probabilities[0].cpu().tolist()
    score_data = {
        CLASSES[i]: scores[i] for i in range(len(CLASSES))
    }
    st.bar_chart(score_data)

    st.caption(
        f"Decision threshold: {THRESHOLD:.2f}. "
        "This is a confidence-based heuristic, not a guarantee "
        "that an image belongs to a known or unknown category."
    )
else:
    st.write(
        "Upload an image above to view its prediction and "
        "uncertainty result."
    )

st.divider()
st.caption(
    "Model: CNN | Dataset: CIFAR-10 | "
    "OOD method: Maximum Softmax Probability (MSP)"
)
