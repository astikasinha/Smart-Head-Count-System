import cv2

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Start capturing video
cap = cv2.VideoCapture(0)

# Frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Vertical red line in the middle (for left-to-right crossing)
mid_line = frame_width // 2

# Trackers to store previous person positions
trackers = {}
next_id = 1
crossing_count = 0

# Helper function to determine if an x-coordinate is to the left or right of the line
def get_line_side(x, mid_line):
    return "left" if x < mid_line else "right"

while cap.isOpened():
    for _ in range(4):  # Skip frames for lower latency
        cap.grab()

    ret, frame = cap.read()
    if not ret:
        break

    # Resize for faster processing (optional)
    frame_resized = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    # Detect people using HOG + SVM
    boxes, weights = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)

    # Scale back coordinates if resized
    scale_x = frame.shape[1] / 640
    scale_y = frame.shape[0] / 480

    # Draw vertical mid-line
    cv2.line(frame, (mid_line, 0), (mid_line, frame_height), (0, 0, 255), 2)

    updated_trackers = {}

    for (x, y, w, h) in boxes:
        # Rescale coordinates to original frame
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        center_x = x + w // 2
        center_y = y + h // 2

        matched_id = None
        for track_id, (prev_x, prev_y, prev_side) in trackers.items():
            distance = ((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2) ** 0.5
            if distance < 50:
                matched_id = track_id
                break

        if matched_id is None:
            matched_id = next_id
            next_id += 1

        current_side = get_line_side(center_x, mid_line)
        if trackers.get(matched_id, (0, 0, "none"))[2] == "left" and current_side == "right":
            crossing_count += 1

        updated_trackers[matched_id] = (center_x, center_y, current_side)

        # Draw bounding box and ID
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {matched_id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    trackers = updated_trackers

    # Display crossing count
    cv2.putText(frame, f"Count: {crossing_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with detected persons and count
    cv2.imshow('Person Detection - Left to Right Crossing', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Processing complete.")
print("Frame width:", frame_width)
print("Frame height:", frame_height)
