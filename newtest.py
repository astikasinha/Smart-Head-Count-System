#pip install mediapipe opencv-python

import cv2
import mediapipe as mp

# Initialize face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) 

# Frame skip setup
frame_count = 0
skip_every_n_frames = 4

# Crossing logic
crossing_count = 0
line_position = int(480 / 3)
face_y_history = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    height, width, _ = frame.shape
    cv2.line(frame, (0, line_position), (width, line_position), (0, 0, 255), 2)

    if frame_count % skip_every_n_frames == 0:
        # Process face detection every Nth frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_detection.process(image)
        image.flags.writeable = True

        if results.detections:
            for i, detection in enumerate(results.detections):
                bbox = detection.location_data.relative_bounding_box
                x, y, w, h = int(bbox.xmin * width), int(bbox.ymin * height), \
                             int(bbox.width * width), int(bbox.height * height)
                center_y = y + h // 2

                # Draw face box
                mp_drawing.draw_detection(frame, detection)

                # Track previous y-position for crossing logic
                prev_y = face_y_history.get(i, None)

                if prev_y is not None:
                    if prev_y > line_position and center_y <= line_position:
                        crossing_count += 1

                # Update position
                face_y_history[i] = center_y

    # Show counter
    cv2.putText(frame, f"Count: {crossing_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Face Crossing Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()