import streamlit as st
import cv2
from datetime import date, datetime
import sqlite3
import os
import logging
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle

# Import your existing functions from attendance.py
from attendance import initialize_database, train_or_load_model, predict_student, record_attendance

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the database
initialize_database()

# Load the model and class indices
model, class_indices = train_or_load_model()

# Streamlit app
st.title("Real-Time Attendance System")

# Instructions
st.write("Press 'Start' to begin the attendance system. Press 'q' to quit the webcam feed.")

# Button to start the attendance system
if st.button("Start"):
    # Initialize the webcam
    src = 0
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    if not cap.isOpened():
        st.error(f"Cannot open webcam with src={src}.")
        sys.exit(f"Cannot open webcam. Exiting.")

    # Load the face cascade
    face_cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        st.error(f"{face_cascade_path} not found. Please download it.")
        sys.exit(f"{face_cascade_path} not found. Exiting.")

    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # Set to keep track of recorded attendance
    recorded_set = set()

    # Placeholder for the webcam feed
    frame_placeholder = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame from webcam.")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        # Process each detected face
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            try:
                # Predict the student
                student_name, student_id, confidence = predict_student(model, face_roi, class_indices)
                today = date.today().strftime("%Y-%m-%d")
                record_key = (student_id, today)

                # Record attendance if not already recorded
                if record_key not in recorded_set:
                    record_attendance(student_name, student_id)
                    recorded_set.add(record_key)

                # Display the name and confidence on the frame
                text = f"{student_name} ({student_id}) {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            except ValueError:
                # Handle unrecognized faces
                cv2.putText(frame, "Unrecognized", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Display the frame in the Streamlit app
        frame_placeholder.image(frame, channels="BGR")

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Quitting the attendance system.")
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

# Display attendance records
st.header("Attendance Records")

# Add a date picker to filter records by date
selected_date = st.date_input("Select a date to view attendance records", date.today())

# Fetch and display attendance records from the database for the selected date
if selected_date:
    conn = sqlite3.connect("attendance.sqlite")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance WHERE date = ? ORDER BY timestamp DESC", (selected_date.strftime("%Y-%m-%d"),))
    records = cursor.fetchall()
    conn.close()

    if records:
        st.table(records)
    else:
        st.write(f"No attendance records found for {selected_date}.")