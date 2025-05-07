import streamlit as st
import numpy as np
import cv2
import random

# Initialize session state for ROIs and their colors
if 'rois' not in st.session_state:
    st.session_state.rois = []  # Each ROI is a list of (x, y) points
if 'roi_colors' not in st.session_state:
    st.session_state.roi_colors = []  # Each color is a BGR tuple

st.sidebar.title("Polygon ROI Builder")

# Image uploader
uploaded_file = st.sidebar.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
else:
    image = None
    h = w = 0

# Button to add a new ROI
if st.sidebar.button("Tambah ROI Baru"):
    st.session_state.rois.append([])
    # Generate a random BGR color not used before
    while True:
        color = tuple(random.randint(0, 255) for _ in range(3))
        if color not in st.session_state.roi_colors:
            st.session_state.roi_colors.append(color)
            break

# Sidebar: ROI drawers
for idx, roi in enumerate(st.session_state.rois):
    with st.sidebar.expander(f"ROI ke-{idx+1}", expanded=False):
        # Add point button
        if st.button(f"Tambah Titik pada ROI {idx+1}"):  # unique label
            default = roi[-1] if roi else (0, 0)
            roi.append(default)

        st.write(f"Jumlah titik: {len(roi)}")
        if roi:
            # Select point to edit
            point_idx = st.number_input(
                "Pilih titik index", min_value=1, max_value=len(roi), value=1, step=1,
                key=f"sel_roi{idx}_pt"
            ) - 1
            # Sliders for selected point
            old_x, old_y = roi[point_idx]
            x = st.slider(
                f"ROI {idx+1} - Titik {point_idx+1} X", 0, w, old_x,
                key=f"roi{idx}_pt{point_idx}_x"
            )
            y = st.slider(
                f"ROI {idx+1} - Titik {point_idx+1} Y", 0, h, old_y,
                key=f"roi{idx}_pt{point_idx}_y"
            )
            roi[point_idx] = (x, y)

# Main area: display
st.title("Preview ROI Poligonal")
if image is not None:
    disp = image.copy()
    overlay = image.copy()

    # Draw all ROIs with distinct colors
    for idx, roi in enumerate(st.session_state.rois):
        if len(roi) >= 2:
            pts = np.array(roi, np.int32).reshape((-1, 1, 2))
            color = st.session_state.roi_colors[idx]
            cv2.polylines(disp, [pts], isClosed=True, color=color, thickness=2)
            cv2.fillPoly(overlay, [pts], color=color)

    # Blend overlay
    disp = cv2.addWeighted(overlay, 0.4, disp, 0.6, 0)
    # Convert BGR to RGB and show
    disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    st.image(disp, use_container_width=True)
else:
    st.info("Silakan upload gambar untuk memulai.")
