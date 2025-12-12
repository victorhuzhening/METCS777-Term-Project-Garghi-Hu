import numpy as np
import cv2
import torch
from mediapipe import solutions
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarkerResult
from mediapipe.framework.formats import landmark_pb2
from mmpose.structures import PoseDataSample
import math
from collections import Counter


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


def extract_body_by_frame(pose_data_samples):
    """
    From a list[PoseDataSample] for a single frame, extract a single
    person's body pose as:

      body_xy:     (K, 2) xy coordinates
      body_scores: (K)  keypoint confidences

    If nothing is detected, returns zeros of size (K,2) and (K).
    We infer K from the first non-empty sample.
    """
    # Get first sample with keypoints - assumes only one person in frame
    for sample in pose_data_samples:
        instances = getattr(sample, "pred_instances", None)
        keypoints = getattr(sample, "keypoints", None)
        confidence_scores = getattr(sample, "keypoint_scores", None)

        if instances is None or keypoints is None or confidence_scores is None:
            continue

        body_coordinates = keypoints[0]
        body_scores = confidence_scores[0]

        if isinstance(body_coordinates, np.ndarray):
            body_coordinates = torch.from_numpy(body_coordinates.astype(np.float32))
        else:
            body_coordinates = body_coordinates.to(dtype=torch.float32)

        if isinstance(confidence_scores, np.ndarray):
            # flatten scores to 1D
            body_scores = torch.from_numpy(body_scores.astype(np.float32)).reshape(-1)
        else:
            body_scores = body_scores.to(dtype=torch.float32).reshape(-1)

        return body_coordinates, body_scores

    # No person in frame fail - return 0 length sample, handled by post_coordinates_to_seq()
    return torch.zeros((0, 2), dtype=torch.float32), torch.zeros((0,), dtype=torch.float32)


def extract_hands_per_frame(result: HandLandmarkerResult):
    """
    From a single HandLandmarkerResult, return two (21, 3) arrays:
      left_hand, right_hand, each with [x, y, z] per joint.
    If a hand is missing, it's all zeros.

    Returns:
      left_hand:  np.ndarray shape (21, 3)
      right_hand: np.ndarray shape (21, 3)
    """
    left = torch.zeros((21, 3), dtype=torch.float32)
    right = torch.zeros((21, 3), dtype=torch.float32)

    hand_landmarks = getattr(result, "hand_landmarks", [])
    handedness = getattr(result, "handedness", [])

    for idx, landmark_list in enumerate(hand_landmarks):
        landmark_array = torch.tensor(
            [[landmark.x, landmark.y, landmark.z] for landmark in landmark_list],
            dtype=torch.float32
        )
        # Pad or truncate to 21 coordinates if size is weird
        if landmark_array.shape[0] < 21:
            pad = torch.zeros((21 - landmark_array.shape[0], 3), dtype=torch.float32)
            landmark_array = torch.cat([landmark_array, pad], dim=0)
        elif landmark_array.shape[0] > 21:
            landmark_array = landmark_array[:21]

        hand_label = "unknown" # initialize left or right handedness
        if idx < len(handedness) and len(handedness[idx]) > 0:
            hand_label = handedness[idx][0].category_name.lower()

        if "left" in hand_label:
            left = landmark_array
        elif "right" in hand_label:
            right = landmark_array
        else:
            if torch.allclose(left, torch.zeros_like(left)): # if weird handedness we place into left
                left = landmark_array
            else:
                right = landmark_array
    return left, right


def pose_coordinates_to_seq(pose_coordinates_sequence, num_keypoints: int = 17):
    """
    Convert pose results into ML-friendly per-frame features:

    results_sequence: list where each element is `pose_samples` for that frame,
                      and `pose_samples` is a list[PoseDataSample].

    Returns:
      {
        "num_frames": T,
        "frames": [
          {
            "frame_index": t,
            "body_coordinates": torch.FloatTensor [K, 2],
            "body_scores":      torch.FloatTensor [K]
          },
          ...
        ]
      }
    """
    frames = []

    for frame_idx, pose_samples in enumerate(pose_coordinates_sequence):
        body_coordinates, body_scores = extract_body_by_frame(pose_samples)

        if body_coordinates.numel() == 0:
            # we handle no person detected with stable zero arrays
            body_coordinates = torch.zeros((num_keypoints, 2), dtype=torch.float32)
            body_scores = torch.zeros((num_keypoints,), dtype=torch.float32)

        frames.append(
            {
                "frame_index": frame_idx,
                "body_coordinates": body_coordinates.tolist(),   # [K,2]
                "body_scores": body_scores.tolist(),             # [K]
            }
        )

    return {"num_frames": len(frames), "frames": frames}


def hand_coordinates_to_seq(hand_coordinates_sequence):
    """
    Convert a list of HandLandmarkerResult (one per frame) into
    an ML-friendly JSON structure:

    {
      "num_frames": T,
      "frames": [
        {
          "frame_index": t,
          "left_hand":  [[x,y,z] * 21],
          "right_hand": [[x,y,z] * 21]
        },
        ...
      ]
    }

    Each hand is always shape [21, 3] (padded with zeros if missing).
    """
    frames = []
    for frame_idx, hand_samples in enumerate(hand_coordinates_sequence):
        if hand_samples is None:
            left = torch.zeros((21, 3), dtype=torch.float32)
            right = torch.zeros((21, 3), dtype=torch.float32)
        else:
            left, right = extract_hands_per_frame(hand_samples)

        frames.append(
            {
                "frame_index": frame_idx,
                "left_hand": left.tolist(),
                "right_hand": right.tolist(),
            }
        )

    return {"num_frames": len(frames), "frames": frames}


def build_id_to_token(vocab: dict) -> dict:
    """
    Builds id to token mapping helper.
    Converts {token: id} → {id: token}.
    """
    return {idx: tok for tok, idx in vocab.items()}


def tokens_to_text(
    ids,
    id_to_token,
    pad_id: int,
    bos_token: str = "<bos>",
    eos_token: str = "<eos>",
):
    """
    Convert a sequence of token IDs into a space-separated string.
    Skips <pad> and <bos>, stops at first <eos>.
    """
    tokens = []
    for i in ids:
        i = int(i)

        if i == pad_id:
            continue  # ignore padding

        tok = id_to_token.get(i, "<unk>")

        if tok == bos_token:
            continue  # skip <bos>

        if tok == eos_token:
            break  # stops at first <eos>

        tokens.append(tok)

    return " ".join(tokens)


def bleu1(pred_tokens, label_tokens):
    """
    Custom BLEU score calculation between label and prediction tokens.
    Avoids extra dependencies.
    """
    if len(pred_tokens) == 0:
        return 0.0

    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(label_tokens)

    overlap = sum(min(pred_counts[word], ref_counts[word]) for word in pred_counts)

    precision = overlap / len(pred_tokens)

    # brevity penalty
    label_len = len(label_tokens)
    pred_len = len(pred_tokens)

    if pred_len == 0:
        return 0.0
    if pred_len > label_len:
        bleu_score = 1.0
    else:
        bleu_score = math.exp(1.0 - label_len / pred_len)

    return bleu_score * precision


def rouge1_f1(pred_tokens, label_tokens):
    """
    Custom ROUGE score calculation between label and prediction tokens.
    Avoids extra dependencies.
    """
    if not pred_tokens or not label_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    label_counts = Counter(label_tokens)

    overlap = sum(min(pred_counts[word], label_counts[word]) for word in pred_counts)

    precision = overlap / len(pred_tokens)
    recall = overlap / len(label_tokens)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def iter_video_as_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("ERROR: Cannot open video file")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert cv frame (BGR) to RGB
            yield frame_rgb
    finally:
        cap.release()