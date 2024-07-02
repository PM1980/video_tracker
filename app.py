import cv2
import numpy as np
import streamlit as st
from collections import deque

# Global variables
selection_in_progress = False
roi = None
tracker = None
trajectory = deque(maxlen=64)

def select_roi(event, x, y, flags, param):
    """Callback function for selecting ROI."""
    global frame, roi, selection_in_progress

    if event == cv2.EVENT_LBUTTONDOWN:
        roi = (x, y, 0, 0)
        selection_in_progress = True
    elif event == cv2.EVENT_MOUSEMOVE and selection_in_progress:
        roi = (roi[0], roi[1], x - roi[0], y - roi[1])
    elif event == cv2.EVENT_LBUTTONUP:
        roi = (roi[0], roi[1], x - roi[0], y - roi[1])
        selection_in_progress = False

def main():
    st.title("Object Tracking with Streamlit")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])

    if uploaded_file is not None:
        # OpenCV video capture
        video_bytes = uploaded_file.read()
        cap = cv2.VideoCapture()
        cap.open(uploaded_file.name)

        # Create a window and set mouse callback for ROI selection
        cv2.namedWindow("Select Region to Track")
        cv2.setMouseCallback("Select Region to Track", select_roi)

        global roi, tracker, trajectory

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Draw ROI selection rectangle
            if roi:
                cv2.rectangle(frame, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 2)

            # Select initial ROI
            if tracker is None and not selection_in_progress and roi:
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, roi)
                roi = None  # Reset roi after initialization

            # Update the tracker
            if tracker:
                ok, bbox = tracker.update(frame)

                if ok:
                    # Tracking success, draw bounding box
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

                    # Calculate and store center point
                    center = (int(bbox[0] + bbox[2] / 2), int(bbox[1] + bbox[3] / 2))
                    trajectory.appendleft(center)

                    # Draw trajectory
                    for i in range(1, len(trajectory)):
                        if trajectory[i - 1] is None or trajectory[i] is None:
                            continue
                        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                        cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), thickness)

                    # Calculate and display object speed (pixel distance between last two points)
                    if len(trajectory) >= 2:
                        distance = np.sqrt((trajectory[0][0] - trajectory[1][0]) ** 2 +
                                           (trajectory[0][1] - trajectory[1][1]) ** 2)
                        cv2.putText(frame, f"Speed: {distance:.2f} px/frame", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                else:
                    # Tracking failure
                    cv2.putText(frame, "Tracking failure detected", (100, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                    # Reset tracker
                    tracker = None

            # Convert frame to RGB for display in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Display the frame using Streamlit
            st.image(frame_rgb, channels="RGB", use_column_width=True)

        cap.release()

if __name__ == "__main__":
    main()
