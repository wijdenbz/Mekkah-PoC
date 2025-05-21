def calculate_performance_metrics(
    video_length,
    frame_size,
    model_name,
    num_params,
    avg_fps,
    avg_inference_time,
    total_inference_time,
    total_objects_detected,
    video_name,
    total_frames,
    processed_frames=None,
    avg_objects_per_frame=0,
    max_objects_detected=0,
    total_elapsed_time=None,
):
    """
    Calculate and return performance metrics for video processing.

    Args:
        video_length: Length of video in seconds
        frame_size: Tuple of (width, height)
        model_name: Name of the model used
        num_params: Number of model parameters
        avg_fps: Average frames processed per second
        avg_inference_time: Average inference time per frame
        total_inference_time: Total inference time for all frames
        total_objects_detected: Total number of objects detected
        video_name: Name of the video file
        total_frames: Total number of frames in the video
        avg_objects_per_frame: Average objects detected per frame
        max_objects_detected: Maximum objects detected in any frame
        total_elapsed_time: Total elapsed processing time (including overhead)
    """
    # Default processed_frames to total_frames if not provided
    if processed_frames is None:
        processed_frames = total_frames

    # Default elapsed time to inference time if not provided
    if total_elapsed_time is None:
        total_elapsed_time = total_inference_time

    return {
        "Metric": [
            "Total Frames",
            "Frame Size",
            "Model",
            "Parameters",
            "Total Inference Time (s)",
            "Total Processing Time (s)",
            "Inference Time per Frame (ms)",
            "Average FPS",
            "Average Objects per Frame",
            "Max Objects Detected",
        ],
        "Value": [
            total_frames,
            f"{frame_size[0]}x{frame_size[1]}",
            model_name,
            f"{num_params:,}",
            f"{total_inference_time:.2f}",
            f"{total_elapsed_time:.2f}",
            f"{avg_inference_time * 1000:.2f}",
            f"{avg_fps:.2f}",
            f"{avg_objects_per_frame:.2f}",
            max_objects_detected,
        ],
    }
