import cv2
import time
import os

# Load DNN model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Set the folder path to save images
save_folder = "detected_faces"

# Create the folder if it doesn't exist
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Initialize webcam
cap = cv2.VideoCapture(0)
screenshot_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Prepare image for DNN
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Confidence threshold
            face_detected = True
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (startX, startY, endX, endY) = box.astype("int")

            # Draw rectangle around face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # Label with confidence
            text = f"{confidence * 100:.1f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    if face_detected:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"face_{timestamp}_{screenshot_count}.jpg"
        full_path = os.path.join(save_folder, filename)
        cv2.imwrite(full_path, frame)
        print(f"[Saved] {full_path}")
        screenshot_count += 1
    
    cv2.imshow('DNN Face Detection with Screenshot Save', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
