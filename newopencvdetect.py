import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Initialize the Face Detection model
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Define the line position at 1/3rd from the top of the screen
count = 0
face_tracker = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    LINE_Y = h // 3  # Line at 1/3rd of the screen height
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    new_tracker = {}

    # Draw the line at 1/3rd from the top
    cv2.line(frame, (0, LINE_Y), (w, LINE_Y), (0, 255, 0), 2)

    if results.detections:
        for idx, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            x, y, bw, bh = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
            cx, cy = x + bw // 2, y + bh // 2  # center of face

            # Track each face by center X (roughly)
            face_id = f"{idx}_{x // 50}"
            new_tracker[face_id] = cy

            if face_id in face_tracker:
                prev_y = face_tracker[face_id]
                if prev_y > LINE_Y and cy <= LINE_Y:
                    count += 1

            # Draw face bounding box and center
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    face_tracker = new_tracker

    # Show count on screen
    cv2.putText(frame, f"Count: {count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

    cv2.imshow("Blazeface Line Counter", frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()