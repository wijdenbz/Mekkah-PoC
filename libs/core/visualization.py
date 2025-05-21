import cv2
import numpy as np

# Default class mapping for common COCO classes
DEFAULT_CLASS_MAPPING = {
    0: "human",
    2: "car",
    5: "bus",
    7: "truck",
    1: "bicycle",
    3: "motorcycle",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
}

# Default colors for different classes
DEFAULT_COLORS = {
    0: (255, 0, 0),  # Blue for person
    2: (0, 0, 255),  # Red for car
    5: (0, 255, 0),  # Green for bus
    7: (255, 255, 0),  # Cyan for truck
    1: (255, 0, 255),  # Magenta for bicycle
    3: (0, 255, 255),  # Yellow for motorcycle
    24: (255, 128, 0),  # Orange for backpack
    26: (128, 0, 255),  # Purple for handbag
    28: (0, 128, 255),  # Pink for suitcase
    "default": (255, 255, 255),  # White for others
    "abandoned": (0, 0, 255),  # Red for abandoned luggage
}


def draw_boxes(
    frame,
    boxes,
    classes_detected,
    confidences,
    orig_w,
    orig_h,
    res_w,
    res_h,
    classes_to_count,
    counts,
    class_mapping=None,
    colors=None,
    abandoned_luggage_info=None,
):
    """
    Draw bounding boxes on the frame with class-specific colors

    Args:
        frame: Original frame
        boxes: Bounding boxes
        classes_detected: Class IDs of detected objects
        confidences: Confidence scores
        orig_w, orig_h: Original frame dimensions
        res_w, res_h: Resized frame dimensions
        classes_to_count: List of class IDs to count
        counts: Dictionary of counts by class
        class_mapping: Dictionary mapping class IDs to names (default: DEFAULT_CLASS_MAPPING)
        colors: Dictionary mapping class IDs to colors (default: DEFAULT_COLORS)
        abandoned_luggage_info: List of dictionaries containing abandoned luggage information
    """
    # Use default mappings if none provided
    if class_mapping is None:
        class_mapping = DEFAULT_CLASS_MAPPING

    if colors is None:
        colors = DEFAULT_COLORS

    for idx, cls_id in enumerate(classes_detected):
        # Skip classes we're not counting
        if int(cls_id) not in classes_to_count:
            continue

        # Skip if we run out of boxes
        if idx >= len(boxes):
            continue

        try:
            x1, y1, x2, y2 = boxes[idx]

            # No need to scale - boxes should already be in original frame coordinates
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Ensure coordinates are within frame bounds
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            # Skip drawing if box is too small
            if x2 <= x1 or y2 <= y1:
                continue

            conf = confidences[idx] if idx < len(confidences) else 0.0
            # Increase thickness for all boxes
            box_thickness = 3
            color = colors.get(int(cls_id), colors["default"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

            # Add confidence score with appropriate background
            label = f"{conf:.2f}"
            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            y1 = max(y1, label_size[1])

            # Draw background for text
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - baseline),
                (x1 + label_size[0], y1),
                color,
                cv2.FILLED,
            )

            # Draw text in black
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
        except Exception as e:
            continue

    # Add summary text (counts) to the frame
    try:
        summary_text = ""
        for cls_id, cls_name in class_mapping.items():
            if cls_id in classes_to_count and cls_name in counts:
                if summary_text:
                    summary_text += " | "
                summary_text += f"{cls_name.capitalize()}: {counts[cls_name]}"

        if summary_text:
            # Draw text with background that spans the whole text
            font_scale = 1.2
            thickness = 3
            text_size, _ = cv2.getTextSize(
                summary_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )

            # Position at top-left with padding
            pos_x, pos_y = 10, 45

            # Draw background
            cv2.rectangle(
                frame,
                (pos_x - 8, pos_y - text_size[1] - 8),
                (pos_x + text_size[0] + 8, pos_y + 8),
                (0, 0, 0),  # Black background
                cv2.FILLED,
            )

            # Draw text
            cv2.putText(
                frame,
                summary_text,
                (pos_x, pos_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                thickness,
            )
    except Exception:
        pass

    # Draw prominent warning for abandoned luggage
    if abandoned_luggage_info is not None:
        for info in abandoned_luggage_info:
            x1, y1, x2, y2 = map(int, info["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 6)
            cv2.putText(
                frame,
                " ABANDONED !",
                (x1, max(y1 - 20, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                5,
            )

    return frame
