import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load class names
with open("coco.names", "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines()]

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load image
image_name = "zeb.png"
image = cv2.imread(image_name)

if image is None:
    print("Image not found!")
    exit()

height, width, channels = image.shape

# Convert image to blob
blob = cv2.dnn.blobFromImage(
    image, 0.00392, (416, 416), (0, 0, 0), True, crop=False
)

net.setInput(blob)

# Forward pass
outs = net.forward(output_layers)

boxes = []
confidences = []
class_ids = []

# -----------------------------
# Detection Processing (FIXED)
# -----------------------------
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)

        class_prob = float(scores[class_id])
        objectness = float(detection[4])

        confidence = objectness * class_prob  # ✅ correct YOLO confidence

        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(confidence)
            class_ids.append(class_id)

# -----------------------------
# Non-Max Suppression
# -----------------------------
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# -----------------------------
# Draw Results
# -----------------------------
if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        conf = confidences[i]   # ✅ correct confidence per box

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.putText(
            image,
            f"{label} {conf:.2f}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

# -----------------------------
# Show Output
# -----------------------------
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# Save result
cv2.imwrite("output.png", image)
print("Saved as output.png")