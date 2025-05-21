import streamlit as st
import cv2
import time
import pandas as pd
from libs.core.model_loader import get_model_info, load_yolo_model
from libs.core.frame_processor import process_frame_object, DEFAULT_CLASS_MAPPING
from libs.core.metrics import calculate_performance_metrics

st.set_page_config(page_title="Bus Detection", layout="wide")

# Hide sidebar and other default UI elements
st.markdown(
    """
    <style>
    /* Hide the page navigation links in the sidebar */
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* Hide the sidebar toggle button (hamburger icon) */
    [data-testid="collapsedControl"] {
        display: none;
    }

    /* Optional: hide header/footer if needed */
    header, footer {
        visibility: hidden;
    }
    </style>
""",
    unsafe_allow_html=True,
)

st.title("🚌 Vehicle Detection - Live YOLO Video Processing")

# Sidebar: Upload video
uploaded_file = st.sidebar.file_uploader(
    "Upload a video file", type=["mp4", "avi", "mov"], key="sidebar"
)

# Sidebar: Model selection
st.sidebar.title("🔧 Settings")

# Only YOLOv11 models as specified
available_models = [
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
]

selected_model = st.sidebar.selectbox("Select Model", available_models)

# Classes to count (5=bus, 2=car by default in COCO dataset)
classes_to_count = [2, 5]  # Bus and car classes in COCO dataset

# Custom class mapping for this page
class_mapping = {2: "car", 5: "bus"}

# Confidence threshold slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 0.9, 0.3, 0.05
)  # Default changed to 0.3 for better detection

# Frame skipping factor selection
skip_factor = 5

# Main app behavior
if uploaded_file is not None:
    input_video_path = "temp_video.mp4"
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Create two columns with equal width for videos
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Original Video")
        # Get video dimensions for consistent display
        cap = cv2.VideoCapture(input_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps if fps > 0 else 0

        # Calculate expected number of frames to be processed with frame skipping
        if skip_factor > 0:
            # Formule corrigée pour calculer exactement le nombre de frames qui seront traités
            expected_processed_frames = (total_frames + skip_factor - 1) // skip_factor
        else:
            expected_processed_frames = (
                total_frames  # Process all frames if skip_factor is 0
            )

        cap.release()

        # Display video information AFTER the video
        st.video(input_video_path)
        # st.info(
        #     f"Video Info: {total_frames} frames, {fps} FPS, {video_duration:.2f} seconds"
        # )
        # st.info(
        #     f"With frame skip {skip_factor}, expecting to process ~{expected_processed_frames} frames"
        # )

    if st.button("🚀 Start Processing"):
        # Load model
        with st.spinner(f"Loading model {selected_model}..."):
            try:
                model = load_yolo_model(selected_model)
                model_size, num_params = get_model_info(selected_model)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.stop()

        # Initialize video capture
        cap = cv2.VideoCapture(input_video_path)
        assert cap.isOpened(), "Error reading video file"

        # Get video properties
        w, h, fps = (
            int(cap.get(x))
            for x in (
                cv2.CAP_PROP_FRAME_WIDTH,
                cv2.CAP_PROP_FRAME_HEIGHT,
                cv2.CAP_PROP_FPS,
            )
        )
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a text placeholder for progress updates
        progress_text = st.empty()

        # Create a frame placeholder for live preview
        with col2:
            st.subheader("🎬 Live Processing")
            frame_placeholder = st.empty()

        # Create metrics tracking
        frame_count = 0  # Number of frames actually processed
        frames_read = 0  # Total number of frames read
        total_inference_time = 0
        total_objects_detected = 0
        fps_list = []
        object_counts = []

        # Set up metrics display
        metrics_placeholder = st.empty()

        # Timing for overall processing speed
        overall_start_time = time.time()

        # Process video frame by frame
        try:
            while cap.isOpened():
                success, frame = cap.read()
                frames_read += 1

                # Check if we've reached the end of the video
                if not success:
                    progress_text.markdown(
                        f"### Processing: {frames_read-1}/{total_frames} frames (100.00%) - {frame_count} frames analyzed"
                    )
                    st.info(f"Reached end of video at frame {frames_read-1}.")
                    break

                # Only process every skip_factor-th frame
                # Added safety check to prevent modulo by zero
                if skip_factor > 1 and (frames_read - 1) % skip_factor != 0:
                    # Always process the last frame of the video
                    if frames_read != total_frames:
                        continue

                # Add safety check for valid frame
                if frame is None or frame.size == 0:
                    continue

                start_time = time.time()

                try:
                    processed_frame, counts, _ = process_frame_object(
                        frame, model, conf_threshold, classes_to_count, class_mapping
                    )

                    # Get total count across all classes
                    count = sum(counts.values()) if isinstance(counts, dict) else 0
                except Exception as e:
                    st.warning(f"Error processing frame {frames_read}: {str(e)}")
                    continue

                inference_time = time.time() - start_time

                frame_count += 1
                total_inference_time += inference_time
                total_objects_detected += count
                object_counts.append(count)

                # Calculate instantaneous FPS for this processed frame
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                fps_list.append(current_fps)

                # Update UI every 3rd processed frame or at the beginning/end
                if (
                    frame_count % 3 == 0
                    or frame_count == 1
                    or frames_read >= total_frames
                ):
                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        display_frame, channels="RGB", use_container_width=True
                    )

                    progress_percentage = (
                        min(100.0, (frames_read / total_frames) * 100)
                        if total_frames > 0
                        else 0
                    )
                    progress_text.markdown(
                        f"### Processing: {frames_read}/{total_frames} frames ({progress_percentage:.2f}%) - {frame_count} frames analyzed"
                    )

                    # Calculate FPS metrics
                    processing_fps = (
                        frame_count / total_inference_time
                        if total_inference_time > 0
                        else 0
                    )

                    # Prepare metrics data with both raw and adjusted values
                    avg_objects_per_frame = (
                        total_objects_detected / frame_count if frame_count > 0 else 0
                    )
                    max_detected = max(object_counts) if object_counts else 0

                    # Use centralized metrics function
                    elapsed_time = time.time() - overall_start_time

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
                        total_elapsed_time=elapsed_time,
                    )

                    # Add class-specific counts if available
                    if isinstance(counts, dict) and len(counts) > 0:
                        for class_name, class_count in counts.items():
                            metrics_data["Metric"].append(
                                f"{class_name.capitalize()} Count"
                            )
                            metrics_data["Value"].append(str(class_count))

                    # Create a simple DataFrame with only string values
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_placeholder.dataframe(metrics_df)

            # Check if we processed all expected frames
            if frames_read < total_frames:
                st.warning(
                    f"Video processing stopped at frame {frames_read} of {total_frames}. This might indicate an issue with the video file."
                )

        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
        finally:
            # Ensure the capture is always released
            cap.release()

        # Calculate final elapsed time
        total_elapsed_time = time.time() - overall_start_time

        # Final success message
        # st.success(
        #     f"✅ Processing completed! Processed {frame_count} out of {total_frames} frames in {total_elapsed_time:.2f} seconds."
        # )

        # Display performance data in two columns for better layout
        col1, col2 = st.columns(2)

        with col1:
            # Calculate final values with a single FPS metric
            final_fps_avg = (
                frame_count / total_inference_time if total_inference_time > 0 else 0
            )
            avg_objects_per_frame = (
                total_objects_detected / frame_count if frame_count > 0 else 0
            )
            max_detected = max(object_counts) if object_counts else 0

            # Use centralized metrics function for final display
            final_metrics = calculate_performance_metrics(
                video_length=total_frames / fps if fps > 0 else 0,
                frame_size=(w, h),
                model_name=selected_model,
                num_params=num_params,
                avg_fps=final_fps_avg,
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
                total_elapsed_time=total_elapsed_time,
            )

            # Add class-specific counts if available from the last processed frame
            if isinstance(counts, dict) and len(counts) > 0:
                for class_name, class_count in counts.items():
                    final_metrics["Metric"].append(f"{class_name.capitalize()} Count")
                    final_metrics["Value"].append(str(class_count))

            # Create a simple DataFrame that won't have type issues
            metrics_df = pd.DataFrame(final_metrics)
            st.dataframe(metrics_df)

        with col2:
            # Display chart of object counts over time
            if frame_count > 0:
                st.subheader("📈 Detection Metrics")

                # Create tabs for different charts
                tab1, tab2 = st.tabs(["Vehicle Count", "FPS"])

                with tab1:
                    count_chart_data = pd.DataFrame(
                        {
                            "frame": list(range(1, frame_count + 1)),
                            "Count": object_counts,
                        }
                    )
                    st.line_chart(count_chart_data.set_index("frame"))

                with tab2:
                    fps_chart_data = pd.DataFrame(
                        {"frame": list(range(1, frame_count + 1)), "FPS": fps_list}
                    )
                    st.line_chart(fps_chart_data.set_index("frame"))

else:
    st.info("👈 Please upload a video file to start.")
