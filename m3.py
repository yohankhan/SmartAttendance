import streamlit as st


# --- UI ---
st.set_page_config(page_title="Smart Attendance System", layout="centered")
st.title("📸 Smart Attendance System")



hide_streamlit_style = """
<style>
/* Hide the entire top-right toolbar including deploy button */
[data-testid="stToolbar"] {
    display: none !important;
}

/* Hide hamburger menu */
#MainMenu {
    visibility: hidden;
}

/* Hide footer */
footer {
    visibility: hidden;
}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from pathlib import Path
import mediapipe as mp
import time

# --- Configuration ---
FACES_DIR = Path("faces")
FACES_DIR.mkdir(exist_ok=True)
ATTENDANCE_FILE = "attendance.csv"

mp_face_detection = mp.solutions.face_detection

# --- Utilities ---
def preprocess_face(image):
    return cv2.resize(image, (100, 100)).flatten() / 255.0

def cosine_similarity(a, b):
    a, b = a.flatten(), b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def show_toast(message, type="info", duration=2):
    toast_slot = st.sidebar.empty()
    if type == "success":
        toast_slot.success(message)
    elif type == "error":
        toast_slot.error(message)
    elif type == "warning":
        toast_slot.warning(message)
    else:
        toast_slot.info(message)
    time.sleep(duration)
    toast_slot.empty()

def detect_and_crop_face(image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y = int(bbox.xmin * iw), int(bbox.ymin * ih)
            w, h = int(bbox.width * iw), int(bbox.height * ih)
            pad = 20
            x, y = max(0, x - pad), max(0, y - pad)
            w, h = min(iw - x, w + pad * 2), min(ih - y, h + pad * 2)
            face = image[y:y+h, x:x+w]
            return preprocess_face(face)
    return None

def register_user(name):
    cap = cv2.VideoCapture(0)
    samples, count = [], 0
    stframe = st.empty()
    show_toast("Look directly into the camera. Capturing 5 samples...", type="info", duration=3)

    while count < 5:
        ret, frame = cap.read()
        if not ret:
            continue
        face = detect_and_crop_face(frame)
        if face is not None:
            samples.append(face)
            count += 1
            cv2.putText(frame, f"Captured {count}/5", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    cap.release()
    avg_face = np.mean(samples, axis=0)
    np.save(FACES_DIR / f"{name}.npy", avg_face)
    st.success(f"Registered Succesfully")
        # ... after done
    show_toast(f"User '{name}' registered successfully.", type="success", duration=3)


def recognize_and_log():
    known_faces, names = [], []
    for file in FACES_DIR.glob("*.npy"):
        known_faces.append(np.load(file))
        names.append(file.stem)

    if not known_faces:
        st.warning("No registered faces found.")
        return

    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    detected_once, unmatched_once = False, False
    timeout = datetime.now().timestamp() + 30

    while datetime.now().timestamp() < timeout:
        ret, frame = cap.read()
        if not ret:
            continue
        face = detect_and_crop_face(frame)
        if face is None:
            if not detected_once:
                show_toast("No face detected. Look directly into the camera.", type="warning", duration=3)
                detected_once = True
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            continue

        scores = [cosine_similarity(face, kf) for kf in known_faces]
        best_idx = np.argmax(scores)
        if scores[best_idx] > 0.75:
            name = names[best_idx]
            now = datetime.now()
            record = {"name": name, "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S")}
            df = pd.DataFrame([record])
            if Path(ATTENDANCE_FILE).exists():
                df.to_csv(ATTENDANCE_FILE, mode='a', header=False, index=False)
            else:
                df.to_csv(ATTENDANCE_FILE, index=False)
            show_toast(f"✅ Attendance marked for {name}.", type="success", duration=3)

            break
        else:
            if not unmatched_once:
                st.warning("Face not matched. Please try again.")
                unmatched_once = True
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()


menu = st.sidebar.selectbox("Select Action", ["Home", "Register", "Take Attendance", "Dashboard"])

if menu == "Home":
    st.subheader("Welcome 👋")
    st.write("Use the sidebar to Register or Take Attendance.")

elif menu == "Register":
    name = st.text_input("Enter Name for Registration")
    if st.button("Start Registration") and name:
        register_user(name.strip())

elif menu == "Take Attendance":
    if st.button("Start Attendance"):
        recognize_and_log()

elif menu == "Dashboard":
    st.subheader("📊 Attendance Dashboard")
    if Path(ATTENDANCE_FILE).exists():
        df = pd.read_csv(ATTENDANCE_FILE)
        filter_name = st.text_input("Filter by name")
        if filter_name:
            df = df[df['name'].str.contains(filter_name, case=False)]
        st.dataframe(df.sort_values(by="time", ascending=False))
    else:
        st.info("No attendance records found.")
