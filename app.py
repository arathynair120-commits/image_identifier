
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

CLASSES = ["Airplane", "Automobile", "Bird", "Cat", "Deer"]
THRESHOLD = 0.70  # Provisional; calibrate using validation data

@st.cache_resource
def load_model():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = CNN(num_classes=5).to(device)
    model.load_state_dict(
        torch.load("models/cnn_oe.pth", map_location=device)
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
    "CNN classification with Outlier Exposure-based "
    "uncertainty detection"
)

st.info(
    "Known classes: Airplane, Automobile, Bird, Cat and Deer. "
    "Other categories may be unfamiliar to the model."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
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
            st.metric("Maximum Softmax Score", f"{confidence_value:.2%}")

            if is_known:
                st.success("Prediction: ACCEPTED AS A KNOWN CLASS")
                st.write(
                    "The score meets the current threshold. "
                    "This does not guarantee that the image is known."
                )
            else:
                st.warning("Prediction: POSSIBLY OUT-OF-DISTRIBUTION")
                st.write(
                    "The score is below the threshold. "
                    "The image may be unfamiliar."
                )

            st.progress(confidence_value)

        st.subheader("Class Confidence Scores")
        scores = probabilities[0].cpu().tolist()
        st.bar_chart({
            CLASSES[i]: scores[i]
            for i in range(len(CLASSES))
        })

        st.caption(
            f"Current threshold: {THRESHOLD:.2f}. "
            "This is a provisional threshold and must be calibrated "
            "on validation data."
        )

    except Exception as e:
        st.error(f"Could not process this image: {e}")

else:
    st.write(
        "Upload an image above to view its predicted class "
        "and uncertainty score."
    )

st.divider()
st.caption(
    "Model: CNN with Outlier Exposure | Dataset: CIFAR-10 | "
    "Scoring method: Maximum Softmax Probability (MSP)"
)
