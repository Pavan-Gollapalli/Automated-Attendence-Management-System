import cv2
import os
import time

# Name of the person (Change for different students)
student_name = "A j 21HT1A4320"

# Create folder if not exists
save_path = f"students/{student_name}"
os.makedirs(save_path, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

# Load OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img_count = 0
start_time = time.time()  # Start timer
max_images = 100  # Capture exactly 100 images
capture_duration = 30  # Maximum time limit (in seconds)

while img_count < max_images:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Convert to grayscale for better detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Crop the face region
        face = frame[y:y+h, x:x+w]
        img_count += 1
        img_name = os.path.join(save_path, f"{student_name}_{img_count}.jpg")
        cv2.imwrite(img_name, face)  # Save only the face
        print(f"Saved {img_name}")

        # Draw a rectangle around the face (for visualization)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Face Capture", frame)

    # Stop if 100 images are captured or time exceeds 30 seconds
    if img_count >= max_images or (time.time() - start_time) > capture_duration:
        break

    # Adjust capture speed dynamically (wait for a small time if capturing too fast)
    time.sleep(0.1)  # Small delay to control speed

cap.release()
cv2.destroyAllWindows()
print(f"Captured {img_count} images in {round(time.time() - start_time, 2)} seconds.")
