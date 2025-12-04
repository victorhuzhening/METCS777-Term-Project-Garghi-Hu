import numpy as np
import cv2
import json
from mediapipe import solutions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.framework.formats import landmark_pb2
from mmpose.structures import PoseDataSample

"""
Function is credited to Google's The MediaPipe Authors 2023 Copyright
"""
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


"""
End Credits
"""


def landmark_list_to_dict(landmark_list):
    """Convert a list of NormalizedLandmark objects to JSON-friendly dicts."""
    return [
        {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
        }
        for lm in landmark_list
    ]


def handedness_to_dict(handedness_list):
    """Convert MediaPipe handedness classification result."""
    out = []
    for h in handedness_list:
        out.append(
            {
                "category_name": getattr(h, "category_name", None),
                "index": int(getattr(h, "index", -1)),
                "score": float(getattr(h, "score", 0.0)),
                "display_name": getattr(h, "display_name", None),
            }
        )
    return out


def hand_landmarker_result_to_jsonable(result: HandLandmarkerResult):
    """
    Convert a single HandLandmarkerResult into a JSON-serializable dict.
    """
    data = {
        "hand_landmarks": [],
        "hand_world_landmarks": [],
        "handedness": [],
        "timestamps_ms": getattr(result, "timestamps_ms", None),
    }

    for landmark_list in getattr(result, "hand_landmarks", []):
        data["hand_landmarks"].append(landmark_list_to_dict(landmark_list))

    for world_landmark_list in getattr(result, "hand_world_landmarks", []):
        data["hand_world_landmarks"].append(landmark_list_to_dict(world_landmark_list))

    for handedness_list in getattr(result, "handedness", []):
        data["handedness"].append(handedness_to_dict(handedness_list))

    return data


def hand_landmarks_to_json(results_sequence):
    """
    Convert a list of HandLandmarkerResult (one per frame) into nested JSON.

    results_sequence: list of HandLandmarkerResult
    """
    frames = []
    for frame_idx, res in enumerate(results_sequence):
        frames.append(
            {
                "frame_index": frame_idx,
                "result": hand_landmarker_result_to_jsonable(res),
            }
        )

    return {"num_frames": len(frames), "frames": frames}


def tensor_to_list(x):
    """Utility: convert torch.Tensor / np.ndarray / None to plain Python lists."""
    if x is None:
        return None
    # torch tensor
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
    except ImportError:
        pass

    # numpy array or anything with .tolist()
    if hasattr(x, "tolist"):
        return x.tolist()

    # already Python lists or other types
    return x


def pred_instances_to_jsonable(pred_instances):
    """
    Convert PoseDataSample.pred_instances into a JSON-serializable dict.

    pred_instances is an InstanceData. Its fields are tensors/ndarrays:
      - keypoints: [num_instances, num_kpts, 2]
      - keypoint_scores: [num_instances, num_kpts]
      - scores: [num_instances]
      - bboxes: [num_instances, 4]
    """
    keypoints = getattr(pred_instances, "keypoints", None)
    keypoint_scores = getattr(pred_instances, "keypoint_scores", None)
    scores = getattr(pred_instances, "scores", None)
    bboxes = getattr(pred_instances, "bboxes", None)

    jsonable = {
        "keypoints": tensor_to_list(keypoints),
        "keypoint_scores": tensor_to_list(keypoint_scores),
        "scores": tensor_to_list(scores),
        "bboxes": tensor_to_list(bboxes),
    }
    return jsonable



def pose_data_to_json(results_sequence):
    frames = []

    for frame_idx, pose_samples in enumerate(results_sequence):
        # pose_samples: list[PoseDataSample] for this frame
        instances_json = [
            {
                "pred_instances": pred_instances_to_jsonable(sample.pred_instances)
            }
            for sample in pose_samples
        ]

        frames.append(
            {
                "frame_index": frame_idx,
                "instances": instances_json,
            }
        )

    return {
        "num_frames": len(frames),
        "frames": frames,
    }


def extract_left_right_hands(result: HandLandmarkerResult):
    """
    From a single HandLandmarkerResult, return two (21, 3) arrays:
      left_hand, right_hand, each with [x, y, z] per joint.
    If a hand is missing, it's all zeros.

    Returns:
      left_hand:  np.ndarray shape (21, 3)
      right_hand: np.ndarray shape (21, 3)
    """
    # Default: zeros
    left = np.zeros((21, 3), dtype=np.float32)
    right = np.zeros((21, 3), dtype=np.float32)

    hand_lms = getattr(result, "hand_landmarks", [])
    handedness = getattr(result, "handedness", [])

    for idx, lm_list in enumerate(hand_lms):
        # Convert landmarks to (21, 3)
        arr = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
        # Pad or truncate to 21 if weird size
        if arr.shape[0] < 21:
            pad = np.zeros((21 - arr.shape[0], 3), dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > 21:
            arr = arr[:21]

        # Determine if this is left or right from handedness
        hand_label = "unknown"
        if idx < len(handedness) and len(handedness[idx]) > 0:
            hand_label = handedness[idx][0].category_name.lower()

        if "left" in hand_label:
            left = arr
        elif "right" in hand_label:
            right = arr
        else:
            # If unknown, just put it into whichever is still zero
            if np.allclose(left, 0):
                left = arr
            else:
                right = arr

    return left, right
