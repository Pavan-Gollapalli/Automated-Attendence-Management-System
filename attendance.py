import os
import sqlite3
import pickle
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from datetime import datetime, date
import logging
import sys

# ----------------------------
# 1. Configure Logging
# ----------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ----------------------------
# 2. Initialize SQLite DB
# ----------------------------
def initialize_database(db_path="attendance.sqlite"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_name TEXT NOT NULL,
            student_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    logger.info("SQLite database initialized.")

# ----------------------------
# 3. Train or Load Model
# ----------------------------
def train_or_load_model():
    model_path = "student_cnn_model.keras"
    class_indices_path = "class_indices.pkl"

    if os.path.exists(model_path) and os.path.exists(class_indices_path):
        try:
            model = load_model(model_path)
            with open(class_indices_path, "rb") as f:
                class_indices = pickle.load(f)
            logger.info("Model and class indices loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load model or class indices. Re-training model. Error: {e}")
            model, class_indices = train_model()
    else:
        model, class_indices = train_model()

    return model, class_indices

def train_model():
    base_path = "students"
    img_height, img_width = 128, 128
    batch_size = 32

    if not os.path.exists(base_path):
        logger.error(f"Data directory '{base_path}' does not exist.")
        sys.exit("Training data directory not found. Exiting.")

    datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    train_generator = datagen.flow_from_directory(
        base_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
    )
    val_generator = datagen.flow_from_directory(
        base_path,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
    )

    class_indices = train_generator.class_indices
    with open("class_indices.pkl", "wb") as f:
        pickle.dump(class_indices, f)

    model = Sequential([
        Input(shape=(img_height, img_width, 3)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(len(class_indices), activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=10,
        verbose=1,
    )

    model.save("student_cnn_model.keras")
    logger.info("Model trained and saved successfully.")
    return model, class_indices

# ----------------------------
# 4. Prediction Function
# ----------------------------
def predict_student(model, face_roi, class_indices, confidence_threshold=0.8):
    img_height, img_width = 128, 128
    frame_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb).resize((img_width, img_height))
    img_array = np.expand_dims(np.array(pil_img) / 255.0, axis=0)

    predictions = model.predict(img_array)
    predicted_prob = np.max(predictions)
    class_id = np.argmax(predictions)

    if predicted_prob < confidence_threshold:
        raise ValueError("Low confidence prediction - likely an unrecognized face.")

    inverse_class_indices = {v: k for k, v in class_indices.items()}
    class_name = inverse_class_indices[class_id]

    try:
        student_name, student_id = class_name.rsplit("_", 1)
    except ValueError:
        student_name = class_name
        student_id = "UnknownID"

    return student_name, student_id, predicted_prob

# ----------------------------
# 5. Attendance Recording
# ----------------------------
def record_attendance(student_name, student_id, db_path="attendance.sqlite"):
    today = date.today().strftime("%Y-%m-%d")
    now = datetime.now()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM attendance WHERE student_id = ? AND date = ?",
        (student_id, today)
    )
    if cursor.fetchone():
        conn.close()
        logger.info(f"Attendance already recorded for {student_name} ({student_id}) today.")
        return False

    cursor.execute("""
        INSERT INTO attendance (student_name, student_id, timestamp, date)
        VALUES (?, ?, ?, ?)
    """, (student_name, student_id, now.strftime("%Y-%m-%d %H:%M:%S"), today))
    conn.commit()
    conn.close()
    logger.info(f"Attendance recorded for {student_name} ({student_id}).")
    return True

# ----------------------------
# 6. Main Attendance Function
# ----------------------------
def main():
    initialize_database()

    model, class_indices = train_or_load_model()

    face_cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(face_cascade_path):
        logger.error(f"{face_cascade_path} not found. Please download it.")
        sys.exit(f"{face_cascade_path} not found. Exiting.")

    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    src = 0
    cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
    if not cap.isOpened():
        logger.error(f"Cannot open webcam with src={src}.")
        sys.exit(f"Cannot open webcam. Exiting.")

    recorded_set = set()
    logger.info("Starting real-time attendance system. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to grab frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            try:
                student_name, student_id, confidence = predict_student(model, face_roi, class_indices)
                today = date.today().strftime("%Y-%m-%d")
                record_key = (student_id, today)

                if record_key not in recorded_set:


                    
                    record_attendance(student_name, student_id)
                    recorded_set.add(record_key)

                text = f"{student_name} ({student_id}) {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            except ValueError:
                cv2.putText(frame, "Unrecognized", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        cv2.imshow('Real-Time Attendance', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Quitting the attendance system.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
