import cv2
import numpy as np
url= (input("RTSP URL your camera here:"))
# Load YOLO model with pre-trained weights and configuration file
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load COCO names (class labels)
with open("coco.names", "r") as f:
    classes = f.read().strip().split('\n')

# RTSP URL of your CCTV camera
rtsp_url = url

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open RTSP stream.")
    exit()

while True:
    # Read a frame from the RTSP stream
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read frame.")
        break

    # Get the height and width of the frame
    height, width = frame.shape[:2]

    # Create a blob from the frame and perform forward pass
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    detections = net.forward(output_layers)

    # Loop over the detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out low confidence detections and consider only "person" and "dog" classes
            if confidence > 0.5 and (classes[class_id] == "person" or classes[class_id] == "dog"):
                center_x, center_y = int(obj[0] * width), int(obj[1] * height)
                w, h = int(obj[2] * width), int(obj[3] * height)

                # Draw bounding box and label on the frame
                x, y = int(center_x - w/2), int(center_y - h/2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"{classes[class_id]}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
