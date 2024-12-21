import cv2
import time
from ultralytics import YOLO  # Make sure you have the ultralytics library installed

# Load the YOLO model with ByteTrack tracker
model = YOLO(r"C:\Users\astik\OneDrive\Desktop\mitzvahwebcam\model.pt")

# Initialize webcam capture
cap = cv2.VideoCapture(1)  # Use 0 for the default webcam

# Check if the webcam was opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize a set to track unique IDs and count the total number of persons
unique_ids_seen = set()

start_time = time.time()

# Process each frame from the webcam
while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        print("Error: Could not read frame.")
        break

    # Perform tracking on the current frame
    results = model.track(source=frame, persist=True, stream=False, tracker=r"C:\Users\astik\OneDrive\Desktop\mitzvahwebcam\bytetrack.yaml")

    # Process each result frame
    for result in results:
        for det in result.boxes:  # Iterate through detected boxes
            if det.id is not None:
                trackid = int(det.id)  # Get the tracking ID
                unique_ids_seen.add(trackid)  # Add to the global set of seen IDs

                # Get bounding box coordinates for each detected person
                x1, y1, x2, y2 = map(int, det.xyxy[0])  # Convert to integers for drawing

                # Draw the bounding box around the detected person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Display the tracking ID near the bounding box
                cv2.putText(frame, f'ID: {trackid}', (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Count the total number of unique persons seen so far
    total_person_count = len(unique_ids_seen)

    # Display the total unique person count in the frame
    cv2.putText(frame, f'Total Unique Persons: {total_person_count}', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame in a window
    cv2.imshow('Webcam Feed', frame)

    elapsed_time = time.time() - start_time
    if elapsed_time >= 60:
        unique_ids_seen.clear()  # Reset the unique IDs set
        start_time = time.time()

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam capture object
cap.release()
cv2.destroyAllWindows()
