
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from models.cnn import CNN

# Model settings
CLASSES = ["Airplane", "Automobile", "Bird", "Cat", "Deer"]
THRESHOLD = 0.82

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN(num_classes=5).to(device)
model.load_state_dict(
    torch.load("models/cnn.pth", map_location=device)
)
model.eval()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam not detected. Check camera permissions.")
    raise SystemExit

print("Live demo started. Press Q to quit.")

frame_count = 0
prediction = "Starting..."
confidence = 0.0
status = "ANALYZING"

while True:
    ret, frame = cap.read()

    if not ret:
        print("Unable to read webcam.")
        break

    frame_count += 1

    # Predict every 5 frames
    if frame_count % 5 == 0:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = torch.softmax(output, dim=1)
            confidence_tensor, predicted = probabilities.max(dim=1)

        confidence = confidence_tensor.item()
        prediction = CLASSES[predicted.item()]

        if confidence >= THRESHOLD:
            status = "KNOWN CLASS"
        else:
            status = "POSSIBLY UNKNOWN"

    # Display results
    cv2.putText(
        frame, f"Prediction: {prediction}",
        (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 0), 2
    )

    cv2.putText(
        frame, f"Confidence: {confidence * 100:.1f}%",
        (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 0), 2
    )

    color = (0, 255, 0) if status == "KNOWN CLASS" else (0, 0, 255)

    cv2.putText(
        frame, f"Status: {status}",
        (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, color, 2
    )

    cv2.imshow("Live Image Classification and OOD Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
