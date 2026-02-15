import streamlit as st
import cv2
import numpy as np

st.title("Leaf Health Detector (Image Upload Demo)")

st.write(
    "Upload a leaf image. The app uses simple color-based image processing "
    "to classify it as **Healthy** (green) or **Diseased** (yellow/brown)."
)

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

def dominant_color(img):
    b_mean = np.mean(img[:, :, 0])
    g_mean = np.mean(img[:, :, 1])
    r_mean = np.mean(img[:, :, 2])
    vals = [b_mean, g_mean, r_mean]
    return int(np.argmax(vals))  # 0=B,1=G,2=R

def detect_leaf_mask(img):
    kernel = np.ones((7, 7), np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # brown + yellow/green ranges (example thresholds) [web:10]
    mask_brown = cv2.inRange(hsv, (8, 60, 20), (30, 255, 200))
    mask_yellow_green = cv2.inRange(hsv, (10, 39, 64), (86, 255, 255))

    mask = cv2.bitwise_or(mask_yellow_green, mask_brown)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

if uploaded_file is not None:
    # read file into OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

    mask = detect_leaf_mask(img)
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
    label_text = "No leaf detected"
    color_box = (0, 255, 255)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        crop_img = img[y:y+h, x:x+w]

        dc = dominant_color(crop_img)

        if dc == 1:
            hist, _ = np.histogram(crop_img[:, :, 1], bins=10, range=(0, 10))
            if np.sum(hist) > 140:  # simple heuristic [web:10]
                color_box = (0, 255, 0)
                label_text = "Healthy Leaf"
            else:
                color_box = (0, 0, 255)
                label_text = "Diseased Leaf"
        else:
            color_box = (0, 0, 255)
            label_text = "Diseased Leaf"

        cv2.rectangle(output, (x, y), (x + w, y + h), color_box, 2)

    cv2.putText(
        output,
        label_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    st.subheader("Processed Image")
    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), channels="RGB")
    st.success(f"Result: {label_text}")
