import streamlit as st
from ultralytics import YOLO
import tempfile
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import altair as alt
from streamlit_image_comparison import image_comparison

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Ship Detection | YOLOv8",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 900;
    color: #0A1AFF;
    text-align: center;
}
.subtitle {
    font-size: 18px;
    color: #666;
    text-align: center;
    margin-bottom: 25px;
}
.footer {
    text-align: center;
    color: #999;
    font-size: 14px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown('<div class="main-title">🚢 Confidence-Based Ship Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">YOLOv8 Deep Learning • Remote Sensing</div>', unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Detection Controls")

conf_threshold = st.sidebar.slider(
    "Confidence Threshold (UI filter)",
    0.0, 1.0, 0.5, 0.01
)

show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)
show_labels = st.sidebar.checkbox("Show Confidence Labels", True)
show_heatmap = st.sidebar.checkbox("Show Heatmap", True)

st.sidebar.markdown("---")
st.sidebar.info("Model runs unfiltered. Threshold applied after inference.")

# ================= FILE UPLOAD =================
uploaded_file = st.file_uploader(
    "📤 Upload Ship / Vessel Image",
    type=["jpg", "jpeg", "png"]
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    return YOLO("D:\\CNN Model–Based Ship Detection from High-Resolution Aerial Images\\best.pt")

model = load_model()

# ================= PROCESS IMAGE =================
if uploaded_file:
    with st.spinner("⏳ Running YOLOv8 inference..."):
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

    # Run YOLO without confidence filtering
    results = model.predict(temp_path, conf=0.0)

    img = cv2.imread(temp_path)
    img_result = img.copy()

    boxes_all = results[0].boxes.xyxy
    confs_all = results[0].boxes.conf.tolist() if boxes_all is not None else []

    # Manual thresholding
    filtered = [
        (box, conf) for box, conf in zip(boxes_all, confs_all)
        if conf >= conf_threshold
    ]

    ship_count = len(filtered)
    confidences = [conf for _, conf in filtered]

    # ================= DRAW DETECTIONS =================
    heatmap = np.zeros(img.shape[:2], dtype=np.float32)

    for box, conf in filtered:
        x1, y1, x2, y2 = map(int, box)

        if conf >= 0.7:
            color = (0, 255, 0)
        elif conf >= 0.4:
            color = (0, 165, 255)
        else:
            color = (0, 0, 255)

        if show_boxes:
            cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)

        if show_labels:
            cv2.putText(
                img_result,
                f"Ship {int(conf * 100)}%",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        heatmap[y1:y2, x1:x2] += conf

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_result_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)

    # ================= KPI METRICS =================
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🚢 Ships Detected", ship_count)
    c2.metric(
    "📊 Avg Confidence",
    f"{int(np.mean(confidences) * 100)}%" if ship_count else "0%"
)

    c3.metric(
    "🎯 Max Confidence",
    f"{int(max(confidences) * 100)}%" if ship_count else "0%"
)

    c4.metric("✅ Status", "Detected" if ship_count else "No Ships")

    # ================= IMAGE COMPARISON =================
    st.subheader("🔍 Before vs After")
    image_comparison(
        img_rgb,
        img_result_rgb,
        label1="Original Image",
        label2="Detected Ships"
    )

    # ================= BAR GRAPH (SHIP vs CONFIDENCE) =================
    if ship_count:
        st.subheader("📊 Confidence per Detected Ship")

        df_bar = pd.DataFrame({
            "Ship": [f"Ship {i+1}" for i in range(ship_count)],
            "Confidence (%)": [int(c * 100) for c in confidences]
        })

        bar_chart = alt.Chart(df_bar).mark_bar().encode(
            x=alt.X("Ship:N", title="Detected Ships"),
            y=alt.Y("Confidence (%):Q", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(
                "Confidence (%):Q",
                scale=alt.Scale(scheme="redyellowgreen"),
                legend=None
            ),
            tooltip=["Ship", "Confidence (%)"]
        ).properties(height=350)

        st.altair_chart(bar_chart, use_container_width=True)

    # ================= DETECTION TABLE =================
    if ship_count:
        st.subheader("📋 Detection Details")

        table = []
        for i, (box, conf) in enumerate(filtered):
            x1, y1, x2, y2 = map(int, box)
            table.append({
                "Ship ID": i + 1,
                "Confidence (%)": int(conf * 100),
                "X1": x1, "Y1": y1,
                "X2": x2, "Y2": y2
            })

        st.dataframe(table, use_container_width=True)

    # ================= HEATMAP =================
    if ship_count and show_heatmap:
        st.subheader("🔥 Detection Heatmap")

        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        st.image(
            cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

    # ================= DOWNLOAD =================
    st.download_button(
        "⬇️ Download Detection Result",
        data=cv2.imencode(".png", img_result)[1].tobytes(),
        file_name="ship_detection_result.png",
        mime="image/png"
    )

    os.remove(temp_path)

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
<div class="footer">
<b>Ship Detection System</b><br>
YOLOv8 • Computer Vision • Remote Sensing<br>
Built for Research, Analytics & Production
</div>
""", unsafe_allow_html=True)
