import cv2
import time
import pandas as pd
import streamlit as st
import os
import numpy as np

from libs.core.frame_processor import process_frame_abandoned_detection
from libs.core.metrics import calculate_performance_metrics
from libs.core.model_loader import get_model_info, load_yolo_model

st.set_page_config(
    page_title="Smart Security: Abandoned Object Detection", layout="wide"
)

# Hide sidebar and default UI elements
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    header, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)
# Set up the main title and sidebar
st.title("🧳 Luggage Detection")
# Sidebar layout
st.sidebar.title("🎥 Video Input Mode")
video_input_mode = st.sidebar.radio(
    "Choose Input Source", options=["Upload", "Stream"], index=0, horizontal=True
)
uploaded_file = None
stream_url = ""
# Video input mode selection
if video_input_mode == "Upload":
    uploaded_file = st.sidebar.file_uploader(
        "📁 Browse a video file", type=["mp4", "avi", "mov"]
    )
if video_input_mode == "Stream":
    stream_url = st.sidebar.text_input(
        "🔗 Stream URL",
        value="rtmp://localhost:1935/live/stream",
        help="Enter your stream address.",
    )
# Model settings
st.sidebar.title("🔧 Model Settings")
available_models = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
]
selected_model = st.sidebar.selectbox("Select Model", available_models)
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)

# Constants for detection
SKIP_FACTOR = 5  # Process 1 out of 5 frames for better performance
classes_to_count = [24, 26, 28]  # Backpack, Handbag, Suitcase
luggage_classes = [24, 26, 28]
person_class = 0
abandoned_threshold = 15  # Reduced from 30 to make detection faster
iou_threshold = 0.5  # Reduced from 0.1 to make overlap detection more sensitive

target_class_mapping = {24: "Backpack", 26: "Handbag", 28: "Suitcase"}


@st.cache_resource
def load_model(path):
    return load_yolo_model(path)


# Upload Mode
if video_input_mode == "Upload" and uploaded_file is not None:
    input_video_path = "temp_video.mp4"
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)
    # Display original video
    with col1:
        st.subheader("📊 Original Video")
        cap = cv2.VideoCapture(input_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0
        # expected_processed_frames = (total_frames + skip_factor - 1) // skip_factor
        cap.release()
        st.video(input_video_path)

    if st.button("🚀 Start Processing"):
        try:
            model = load_model(selected_model)
            model_size, num_params = get_model_info(selected_model)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()
        cap = cv2.VideoCapture(input_video_path)
        assert cap.isOpened(), "Error reading video file"
        w, h, fps = (
            int(cap.get(x))
            for x in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            )
        )
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_text = st.empty()

        # Create placeholders for metrics and video display
        with col2:
            st.subheader("🎨 Live Processing")
            frame_placeholder = st.empty()
        frame_count = 0
        display_frame_count = 0
        frames_read = 0
        total_inference_time = 0
        total_objects_detected = 0
        fps_list = []
        object_counts_list = []
        metrics_placeholder = st.empty()
        overall_start_time = time.time()

        # Store last processed results to draw on skipped frames
        last_processed_frame_output = None
        last_counts_output = {}
        last_abandoned_info_output = []

        try:
            while cap.isOpened():
                success, frame = cap.read()
                frames_read += 1
                if not success:
                    break

                display_frame_count += 1

                current_display_frame = frame.copy()

                if frame is None or frame.size == 0:
                    print(f"[WARN Upload] Frame {frames_read} is empty, skipping.")
                    continue

                if (frames_read - 1) % SKIP_FACTOR == 0:
                    frame_count += 1
                    start_time = time.time()
                    try:
                        processed_frame, counts, abandoned_info, _ = (
                            process_frame_abandoned_detection(
                                frame,
                                model,
                                conf_threshold,
                                classes_to_count,
                                luggage_classes,
                                person_class=person_class,
                                abandoned_threshold=abandoned_threshold,
                                class_mapping=target_class_mapping,
                            )
                        )
                        last_processed_frame_output = processed_frame
                        last_counts_output = counts
                        last_abandoned_info_output = abandoned_info

                        current_display_frame = processed_frame
                        count_current_iter = (
                            sum(counts.values()) if isinstance(counts, dict) else 0
                        )

                    except Exception as e:
                        st.warning(f"Error processing frame {frames_read}: {str(e)}")
                        if last_processed_frame_output is not None:
                            current_display_frame = last_processed_frame_output
                        count_current_iter = (
                            sum(last_counts_output.values())
                            if isinstance(last_counts_output, dict)
                            else 0
                        )
                        pass

                    inference_time = time.time() - start_time
                    total_inference_time += inference_time
                    current_fps = 1.0 / inference_time if inference_time > 0 else 0
                    fps_list.append(current_fps)
                    total_objects_detected += count_current_iter
                    object_counts_list.append(count_current_iter)
                else:
                    if last_processed_frame_output is not None:
                        current_display_frame = last_processed_frame_output
                        count_current_iter = (
                            sum(last_counts_output.values())
                            if isinstance(last_counts_output, dict)
                            else 0
                        )
                    else:
                        count_current_iter = 0

                if (
                    display_frame_count % 10 == 0
                    or display_frame_count == 1
                    or frames_read >= total_frames
                ):
                    if (
                        current_display_frame is not None
                        and current_display_frame.size > 0
                    ):
                        display_rgb = cv2.cvtColor(
                            current_display_frame, cv2.COLOR_BGR2RGB
                        )
                        frame_placeholder.image(
                            display_rgb, channels="RGB", use_container_width=True
                        )
                    else:
                        print(
                            f"[WARN Upload] current_display_frame for frame_read {frames_read} is empty/None before display."
                        )

                    progress_text.markdown(
                        f"### Processing: {frames_read}/{total_frames} frames ({(frames_read / total_frames) * 100:.2f}%)"
                    )

                    processing_fps = (
                        frame_count / total_inference_time
                        if total_inference_time > 0
                        else 0
                    )
                    avg_objects_per_frame = (
                        total_objects_detected / frame_count if frame_count > 0 else 0
                    )
                    max_detected = max(object_counts_list) if object_counts_list else 0

                    metrics_data = calculate_performance_metrics(
                        video_length=total_frames / fps if fps > 0 else 0,
                        frame_size=(w, h),
                        model_name=selected_model,
                        num_params=num_params,
                        avg_fps=processing_fps,
                        avg_inference_time=(
                            total_inference_time / frame_count if frame_count > 0 else 0
                        ),
                        total_inference_time=total_inference_time,
                        total_objects_detected=total_objects_detected,
                        video_name=uploaded_file.name,
                        total_frames=total_frames,
                        processed_frames=frame_count,
                        avg_objects_per_frame=avg_objects_per_frame,
                        max_objects_detected=max_detected,
                        total_elapsed_time=time.time() - overall_start_time,
                    )
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_placeholder.dataframe(metrics_df)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            cap.release()
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Performance Metrics")
            final_metrics = calculate_performance_metrics(
                video_length=total_frames / fps if fps > 0 else 0,
                frame_size=(w, h),
                model_name=selected_model,
                num_params=num_params,
                avg_fps=(
                    frame_count / total_inference_time
                    if total_inference_time > 0
                    else 0
                ),
                avg_inference_time=(
                    total_inference_time / frame_count if frame_count > 0 else 0
                ),
                total_inference_time=total_inference_time,
                total_objects_detected=total_objects_detected,
                video_name=uploaded_file.name,
                total_frames=total_frames,
                processed_frames=frame_count,
                avg_objects_per_frame=(
                    total_objects_detected / frame_count if frame_count > 0 else 0
                ),
                max_objects_detected=(
                    max(object_counts_list) if object_counts_list else 0
                ),
                total_elapsed_time=time.time() - overall_start_time,
            )
            st.dataframe(pd.DataFrame(final_metrics))

        with col2:
            if frame_count > 0:
                st.subheader("📈 Detection Metrics")
                tab1, tab2 = st.tabs(["Objects Count", "FPS"])
                with tab1:
                    st.line_chart(
                        pd.DataFrame(
                            {
                                "frame": list(range(1, frame_count + 1)),
                                "Count": object_counts_list,
                            }
                        ).set_index("frame")
                    )
                with tab2:
                    st.line_chart(
                        pd.DataFrame(
                            {"frame": list(range(1, frame_count + 1)), "FPS": fps_list}
                        ).set_index("frame")
                    )

        try:

            if os.path.exists(input_video_path):
                os.remove(input_video_path)
        except Exception:
            pass


elif video_input_mode == "Stream":
    st.info(f"🔴 Connecting to stream: `{stream_url}`")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        st.error("❌ Could not open stream. Check the URL or network.")
    else:
        model = load_model(selected_model)
        model_size, num_params = get_model_info(selected_model)
        frame_placeholder = st.empty()
        fps_display = st.empty()
        object_display = st.empty()
        # Initialize for stream mode
        stream_frame_count = 0  # Frames read from stream
        model_frame_count_stream = 0  # Frames processed by model in stream
        last_processed_frame_stream = None
        last_counts_stream = {}
        last_abandoned_info_stream = []  # Though not directly used for display here yet
        last_inference_time_stream = 0.05  # Default to avoid div by zero, e.g. 20fps

        st.success("✅ Streaming started. Press Stop or refresh to exit.")
        stop_button = st.button("🛙 Stop Stream")

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            stream_frame_count += 1
            if not ret:
                st.warning("⚠️ Stream ended or cannot read frame.")
                break

            current_display_frame_stream = frame  # Start with raw frame

            if (stream_frame_count - 1) % SKIP_FACTOR == 0:
                model_frame_count_stream += 1
                processed_frame, counts, abandoned_info, inference_time = (
                    process_frame_abandoned_detection(
                        frame,
                        model,
                        conf_threshold,
                        classes_to_count,
                        luggage_classes,
                        person_class=person_class,
                        abandoned_threshold=abandoned_threshold,
                        class_mapping=target_class_mapping,
                    )
                )
                last_processed_frame_stream = processed_frame
                last_counts_stream = counts
                last_abandoned_info_stream = abandoned_info
                last_inference_time_stream = inference_time
                current_display_frame_stream = processed_frame
            else:
                # Skipped frame, use last processed results if available
                if last_processed_frame_stream is not None:
                    current_display_frame_stream = last_processed_frame_stream
                    counts = last_counts_stream  # Use last counts for display
                else:
                    # No processed frame yet, current_display_frame_stream is still raw
                    counts = {}  # No counts to display

            # Display logic for stream mode (every frame read, but content depends on skip_factor)
            if (
                current_display_frame_stream is not None
                and current_display_frame_stream.size > 0
            ):
                display_rgb_stream = cv2.cvtColor(
                    current_display_frame_stream, cv2.COLOR_BGR2RGB
                )
                frame_placeholder.image(
                    display_rgb_stream, channels="RGB", use_container_width=True
                )
            else:
                print(
                    f"[WARN Stream] current_display_frame_stream for frame {stream_frame_count} is empty/None."
                )

            # FPS display should reflect the rate of *model processing* if possible,
            # or be an estimate. Here, it shows FPS of the last processed frame.
            fps_val = (
                1.0 / last_inference_time_stream
                if last_inference_time_stream > 0
                else 0
            )
            fps_display.markdown(
                f"**Processing FPS:** {fps_val:.2f} (updates every {SKIP_FACTOR} frame(s))"
            )
            object_display.markdown(
                f"**Objects Detected (in last processed):** {sum(counts.values()) if isinstance(counts, dict) else 0}"
            )

            # Minimal sleep to allow Streamlit to handle events, adjust as needed
            # This doesn't control processing rate, just Streamlit responsiveness a bit.
            time.sleep(0.01)
        cap.release()
