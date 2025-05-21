import cv2
import numpy as np
from time import time

from libs.core.visualization import DEFAULT_CLASS_MAPPING, draw_boxes
from libs.core.video_utils import ensure_frame_dims, resize_frame

# Global dictionary to keep track of luggage boxes and their 'abandoned' counters
# Key: tuple of box coordinates (x1,y1,x2,y2)
# Value: number of consecutive frames where the luggage was not near a person
ABANDONED_LUGGAGE_COUNTER = {}


def compute_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        boxA (list): First box coordinates [x1, y1, x2, y2]
        boxB (list): Second box coordinates [x1, y1, x2, y2]

    Returns:
        float: IoU value between 0 and 1
        - 0 means no overlap
        - 1 means perfect overlap
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


# New detection-only abandoned luggage logic
def process_frame_abandoned_detection(
    frame,
    model,
    conf_threshold,
    classes_to_count,
    luggage_classes,
    person_class=0,
    abandoned_threshold=20,  # Number of frames luggage must be alone to be considered abandoned
    iou_threshold=0.02,  # IoU threshold to determine if luggage is near a person
    class_mapping=None,
):
    """
    Process a frame to detect abandoned luggage and other objects.

    This function:
    1. Detects persons and luggage in the frame
    2. Tracks luggage that is not near any person
    3. Marks luggage as abandoned if it's alone for too long
    4. Draws boxes and warnings on the frame

    Args:
        frame (numpy.ndarray): Input video frame
        model: YOLO model for object detection
        conf_threshold (float): Confidence threshold for detections (0-1)
        classes_to_count (list): List of class IDs to count
        luggage_classes (list): List of class IDs that represent luggage
        person_class (int): Class ID for person detection
        abandoned_threshold (int): Frames threshold for abandoned detection
        iou_threshold (float): IoU threshold for person-luggage overlap
        class_mapping (dict): Mapping of class IDs to names

    Returns:
        tuple: (processed_frame, object_count, abandoned_luggage_info, processing_time)
            - processed_frame: Frame with drawn boxes and warnings
            - object_count: Dictionary of detected objects by class
            - abandoned_luggage_info: List of abandoned luggage boxes
            - processing_time: Time taken to process the frame
    """
    start_time = time()
    if class_mapping is None:
        class_mapping = DEFAULT_CLASS_MAPPING
    frame = ensure_frame_dims(frame)
    if frame is None:
        return np.zeros((640, 640, 3), dtype=np.uint8), {}, [], 0
    orig_frame = frame.copy()
    orig_height, orig_width = frame.shape[:2]
    resized = resize_frame(frame)
    try:
        # Run YOLO detection for both persons and luggage
        detect_classes = list(set(classes_to_count + [person_class]))
        results = model.predict(
            resized,
            conf=conf_threshold,
            iou=0.45,
            classes=detect_classes,
            verbose=False,
        )[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return frame, {}, [], time() - start_time

    # Scale detected boxes back to original frame size
    scale_x = orig_width / resized.shape[1]
    scale_y = orig_height / resized.shape[0]
    if boxes.shape[0] > 0:
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y

    # Separate detections into person boxes and luggage boxes
    person_boxes = [
        boxes[i] for i, cid in enumerate(class_ids) if int(cid) == person_class
    ]
    luggage_boxes = [
        boxes[i] for i, cid in enumerate(class_ids) if int(cid) in luggage_classes
    ]

    print(f"\nDebug Info:")
    print(f"Number of persons detected: {len(person_boxes)}")
    print(f"Number of luggage items detected: {len(luggage_boxes)}")

    # Initialize counters for this frame
    new_counter = {}  # Will store updated counter values
    abandoned_luggage_info = []  # Will store info about abandoned luggage

    # Process each detected luggage item
    for lug_idx, lug_box in enumerate(luggage_boxes):
        overlapped = False
        max_iou = 0.0
        # Check if this luggage overlaps with any person
        for per_idx, per_box in enumerate(person_boxes):
            # Calculate IoU (Intersection over Union) between luggage and person
            iou = compute_iou(lug_box, per_box)
            max_iou = max(max_iou, iou)
            # If IoU exceeds threshold, luggage is considered near a person
            if iou > iou_threshold:
                overlapped = True
                print(
                    f"Luggage {lug_idx} overlaps with Person {per_idx}, IoU: {iou:.3f}"
                )
                break

        if not overlapped:
            print(
                f"Luggage {lug_idx} has no overlap with any person (max IoU: {max_iou:.3f})"
            )

        # Convert box coordinates to tuple for dictionary key
        key = tuple(map(int, lug_box))

        # Update counter based on overlap status
        if not overlapped:
            # If no overlap, increment counter (luggage is alone)
            new_counter[key] = ABANDONED_LUGGAGE_COUNTER.get(key, 0) + 1
            print(f"Luggage {lug_idx} counter increased to: {new_counter[key]}")
        else:
            # If overlap detected, reset counter (luggage is with a person)
            new_counter[key] = 0
            print(f"Luggage {lug_idx} counter reset to 0 due to overlap")

        # Check if luggage should be marked as abandoned
        # Two conditions:
        # 1. Counter exceeds threshold (luggage alone for many frames)
        # 2. No persons detected AND luggage has been seen at least once
        if new_counter[key] > abandoned_threshold or (
            len(person_boxes) == 0 and new_counter[key] > 0
        ):
            print(f"Luggage {lug_idx} marked as abandoned! Counter: {new_counter[key]}")
            abandoned_luggage_info.append({"bbox": lug_box, "abandoned_flag": True})

    # Update global counter with new values
    # Clear old values and update with new ones to prevent memory growth
    ABANDONED_LUGGAGE_COUNTER.clear()
    ABANDONED_LUGGAGE_COUNTER.update(new_counter)
    print(f"Total abandoned items: {len(abandoned_luggage_info)}")

    # Count objects by class for display
    object_count = {}
    for i in classes_to_count:
        if i in class_mapping:
            class_name = class_mapping[i]
            count = sum(1 for cid in class_ids if int(cid) == i)
            object_count[class_name] = count
    # Add abandoned count to display
    object_count["Abandoned"] = len(abandoned_luggage_info)

    # Draw boxes and warnings on frame
    display_frame = draw_boxes(
        orig_frame.copy(),
        boxes,
        class_ids,
        confidences,
        orig_width,
        orig_height,
        resized.shape[1],
        resized.shape[0],
        classes_to_count,
        object_count,
        class_mapping,
        None,
        abandoned_luggage_info,
    )

    return display_frame, object_count, abandoned_luggage_info, time() - start_time


def process_frame_object(
    frame, model, conf_threshold, classes_to_count, class_mapping=None
):
    """
    Process a frame to detect and count objects.

    This function:
    1. Detects objects in the frame using YOLO
    2. Counts objects by class
    3. Draws bounding boxes with labels

    Args:
        frame (numpy.ndarray): Input video frame
        model: YOLO model for object detection
        conf_threshold (float): Confidence threshold for detections (0-1)
        classes_to_count (list): List of class IDs to count
        class_mapping (dict): Mapping of class IDs to names

    Returns:
        tuple: (processed_frame, object_count, processing_time)
            - processed_frame: Frame with drawn boxes
            - object_count: Dictionary of detected objects by class
            - processing_time: Time taken to process the frame
    """
    start_time = time()

    # Use default mapping if not provided
    if class_mapping is None:
        class_mapping = DEFAULT_CLASS_MAPPING

    # Ensure valid frame
    frame = ensure_frame_dims(frame)
    if frame is None:
        return np.zeros((640, 640, 3), dtype=np.uint8), {}, 0

    # Preprocess the frame
    orig_height, orig_width = frame.shape[:2]
    resized = resize_frame(frame)

    try:
        # Run YOLO prediction
        results = model.predict(
            resized,
            conf=conf_threshold,
            iou=0.45,
            classes=classes_to_count,
            verbose=False,
        )[0]

        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids = results.boxes.cls.cpu().numpy()
    except Exception as e:
        # Return empty results on error
        return frame, {}, time() - start_time

    # Count objects by class using np.unique for efficiency
    object_count = {}
    if len(class_ids) > 0:
        # Count occurrences of each class ID
        unique_ids, counts = np.unique(class_ids, return_counts=True)
        # fill the object_count dictionary with counts for specified classes
        for i in classes_to_count:
            if i in class_mapping:
                class_name = class_mapping[i]
                # check if the class ID exists in unique_ids
                # and get the corresponding count
                idx = np.where(unique_ids == i)[0]
                if len(idx) > 0:
                    object_count[class_name] = int(counts[idx[0]])
                else:
                    object_count[class_name] = 0

    # Draw boxes with detailed information
    display_frame = draw_boxes(
        frame.copy(),
        boxes,
        class_ids,
        confidences,
        orig_width,
        orig_height,
        resized.shape[1],
        resized.shape[0],
        classes_to_count,
        object_count,
        class_mapping,
    )

    # Return processed frame, counts dictionary, and inference time
    return display_frame, object_count, time() - start_time
