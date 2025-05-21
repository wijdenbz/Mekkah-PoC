import cv2
import time
import pandas as pd
import streamlit as st
import os
from libs.core.model_loader import get_model_info, load_yolo_model
from libs.core.frame_processor import process_frame_object, DEFAULT_CLASS_MAPPING
from libs.core.metrics import calculate_performance_metrics

st.set_page_config(page_title="Crowd in Mecca", layout="wide")

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


st.title("🕋 Crowd in Mecca ")


# Sidebar layout

st.sidebar.title("🎥 Video Input Mode")

video_input_mode = st.sidebar.radio(
    "Choose Input Source", options=["Upload", "Stream"], index=0, horizontal=True
)


uploaded_file = None

stream_url = ""


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

skip_factor = 5


classes_to_count = [0]  # 0=person

target_class_mapping = {0: "person"}


@st.cache_resource
def load_model(path):

    return load_yolo_model(path)


# Upload Mode

if video_input_mode == "Upload" and uploaded_file is not None:

    input_video_path = "temp_video.mp4"

    with open(input_video_path, "wb") as f:

        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("📊 Original Video")

        cap = cv2.VideoCapture(input_video_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fps = int(cap.get(cv2.CAP_PROP_FPS))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_duration = total_frames / fps if fps > 0 else 0

        expected_processed_frames = (total_frames + skip_factor - 1) // skip_factor

        cap.release()

        st.video(input_video_path)

        # st.info(
        #     f"Video Info: {total_frames} frames, {fps} FPS, {video_duration:.2f} seconds"
        # )

        # st.info(
        #     f"With frame skip {skip_factor}, expecting to process ~{expected_processed_frames} frames"
        # )

    if st.button("🚀 Start Processing"):

        with st.spinner(f"Loading model {selected_model}..."):

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

        with col2:

            st.subheader("🎨 Live Processing")

            frame_placeholder = st.empty()

        frame_count = 0

        frames_read = 0

        total_inference_time = 0

        total_objects_detected = 0

        fps_list = []

        object_counts = []

        metrics_placeholder = st.empty()

        overall_start_time = time.time()

        try:

            while cap.isOpened():

                success, frame = cap.read()

                frames_read += 1

                if not success:

                    break

                if skip_factor > 1 and (frames_read - 1) % skip_factor != 0:

                    if frames_read != total_frames:

                        continue

                if frame is None or frame.size == 0:

                    continue

                start_time = time.time()

                try:

                    processed_frame, counts, _ = process_frame_object(
                        frame,
                        model,
                        conf_threshold,
                        classes_to_count,
                        target_class_mapping,
                    )

                    count = sum(counts.values()) if isinstance(counts, dict) else 0

                except Exception as e:

                    st.warning(f"Error processing frame {frames_read}: {str(e)}")

                    continue

                inference_time = time.time() - start_time

                frame_count += 1

                total_inference_time += inference_time

                total_objects_detected += count

                object_counts.append(count)

                current_fps = 1.0 / inference_time if inference_time > 0 else 0

                fps_list.append(current_fps)

                if (
                    frame_count % 3 == 0
                    or frame_count == 1
                    or frames_read >= total_frames
                ):

                    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                    frame_placeholder.image(
                        display_frame, channels="RGB", use_container_width=True
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

                    max_detected = max(object_counts) if object_counts else 0

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

        # st.success(
        #     f"✅ Done! Processed {frame_count}/{total_frames} frames in {time.time() - overall_start_time:.2f} sec"
        # )

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
                max_objects_detected=max(object_counts) if object_counts else 0,
                total_elapsed_time=time.time() - overall_start_time,
            )

            st.dataframe(pd.DataFrame(final_metrics))

        with col2:

            if frame_count > 0:

                st.subheader("📈 Detection Metrics")

                tab1, tab2 = st.tabs(["People Count", "FPS"])

                with tab1:

                    st.line_chart(
                        pd.DataFrame(
                            {
                                "frame": list(range(1, frame_count + 1)),
                                "Count": object_counts,
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

        with st.spinner(f"Loading model {selected_model}..."):

            model = load_model(selected_model)

            model_size, num_params = get_model_info(selected_model)

        frame_placeholder = st.empty()

        fps_display = st.empty()

        object_display = st.empty()

        st.success("✅ Streaming started. Press Stop or refresh to exit.")

        stop_button = st.button("🛙 Stop Stream")

        while cap.isOpened() and not stop_button:

            ret, frame = cap.read()

            if not ret:

                st.warning("⚠️ Stream ended or cannot read frame.")

                break

            processed_frame, count, inference_time = process_frame_object(
                frame, model, conf_threshold, [0], target_class_mapping
            )

            display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(
                display_frame, channels="RGB", use_container_width=True
            )

            fps_display.markdown(f"**FPS:** {1.0/inference_time:.2f}")

            object_display.markdown(f"**People Detected:** {count}")

            time.sleep(0.02)

        cap.release()
