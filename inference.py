import argparse

from model import *
from utils import *
from pretrained_models import *


def extract_coordinate_sequences(
    video_path: str,
    MP_model,
    body_cfg,
    body_model,
    num_keypoints: int = 17,
):
    hand_results = []
    body_results = []

    for frame_rgb in iter_video_as_frames(video_path):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        hand_coordinates = MP_model.detect(mp_image)
        pose_coordinates = body_cfg.topdown_infer(body_model, frame_rgb)

        hand_results.append(hand_coordinates)
        body_results.append(pose_coordinates)

    hand_seq = hand_coordinates_to_seq(hand_results)
    body_seq = pose_coordinates_to_seq(body_results, num_keypoints=num_keypoints)
    return hand_seq, body_seq

def build_feature_tensor(
    hand_seq: dict,
    body_seq: dict,
    max_frames: int = None,
    frame_subsample: int = 1,
    num_keypoints: int = 17,
) -> torch.Tensor:
    hand_frame_dim = hand_seq.get("num_frames", len(hand_seq.get("frames", [])))
    body_frame_dim = body_seq.get("num_frames", len(body_seq.get("frames", [])))
    frame_dim = min(hand_frame_dim, body_frame_dim)

    if frame_dim == 0:
        raise RuntimeError("Video produced 0 frames while building feature tensor.")

    frames_hand = hand_seq["frames"]
    frames_body = body_seq["frames"]

    frame_subsample = max(1, int(frame_subsample))
    frame_indices = list(range(0, frame_dim, frame_subsample))
    if max_frames is not None and max_frames > 0:
        frame_indices = frame_indices[: max_frames]

    feature_seq = []

    for idx in frame_indices:
        hand_frame = frames_hand[idx]
        body_frame = frames_body[idx]

        left_hand = np.array(hand_frame["left_hand"], dtype=np.float32)
        right_hand = np.array(hand_frame["right_hand"], dtype=np.float32)

        left_hand = left_hand.reshape(-1)
        right_hand = right_hand.reshape(-1)

        body_coordinates = np.array(body_frame["body_coordinates"], dtype=np.float32)
        body_scores = np.array(body_frame["body_scores"], dtype=np.float32)

        feature_vector = np.concatenate(
            [left_hand, right_hand, body_coordinates, body_scores],
            axis=0,
        )
        feature_seq.append(feature_vector)

    if not feature_seq:
        raise RuntimeError("No features extracted while building feature tensor.")

    feature_seq_stack = np.stack(feature_seq, axis=0)  # [T', D]
    feature_seq_tensor = torch.from_numpy(feature_seq_stack).float()
    return feature_seq_tensor




@torch.no_grad()
def greedy_decode(
    model: PoseToTextModel,
    feature: torch.Tensor,       # [1, T, D]
    feature_len: torch.Tensor,   # [1]
    bos_id: int,
    eos_id: int,
    pad_id: int,
    id_to_token: dict,
    max_len: int = 100,
) -> str:
    """
    Greedy decoding for a single sample.
    """
    device = feature.device
    model.eval()

    # Encode pose sequence
    h_n = model.encode(feature, feature_len)
    num_layers_times_dir, B, H = h_n.shape
    assert B == 1, "Expected batch size 1 for greedy decoding."

    # Use last layer
    h_n_last = h_n[-2:]  # [2, 1, H]
    h = torch.cat([h_n_last[0], h_n_last[1]], dim=-1).unsqueeze(0)  # [1, 1, 2H]

    # Start with <bos>
    cur_token = torch.tensor([[bos_id]], device=device, dtype=torch.long)  # [1, 1]

    decoded_ids = []

    for _ in range(max_len):
        emb = model.emb(cur_token)
        dec_out, h = model.decoder(emb, h)
        logits = model.out(dec_out[:, -1, :])
        next_token = torch.argmax(logits, dim=-1)

        token_id = int(next_token.item())
        if token_id == eos_id:
            break

        decoded_ids.append(token_id)
        cur_token = next_token.view(1, 1)

    # Convert token ids → text using existing helper
    text = tokens_to_text(
        decoded_ids,
        id_to_token=id_to_token,
        pad_id=pad_id,
        bos_token="<bos>",
        eos_token="<eos>",
    )
    return text



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run PoseToTextModel inference on a single ASL video."
    )
    parser.add_argument(
        "--video_path",
        type=str,
        required=True,
        help="Path to the video file to transcribe.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best_model_val.pt",
        help="Path to trained checkpoint (saved by train.py).",
    )

    # Pose processing params (should match what you used during precomputation)
    parser.add_argument(
        "--max_frames",
        type=int,
        default=300,
        help="Max frames per video after subsampling.",
    )
    parser.add_argument(
        "--frame_subsample",
        type=int,
        default=2,
        help="Use every N-th frame from the video.",
    )
    parser.add_argument(
        "--num_keypoints",
        type=int,
        default=17,
        help="Number of body keypoints used in pose_coordinates_to_seq.",
    )
    parser.add_argument(
        "--max_decode_len",
        type=int,
        default=60,
        help="Maximum number of tokens to decode.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(f"Video not found: {args.video_path}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt = torch.load(args.checkpoint, map_location=device)

    if "model_state_dict" not in ckpt:
        raise ValueError(
            "Checkpoint does not contain 'model_state_dict'. "
            "Make sure you pass the file saved by train.py."
        )

    state_dict = ckpt["model_state_dict"]
    vocab = ckpt["vocab"]
    pad_id = ckpt["pad_id"]
    train_args = ckpt.get("args", {})

    id_to_token = build_id_to_token(vocab)
    vocab_size = len(vocab)

    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]

    enc_hidden = train_args.get("enc_hidden", 256)
    emb_dim = train_args.get("emb_dim", 256)

    MediaPipeCFG = MediaPipeCfg("pretrained_model/hand_landmarker.task")
    options = MediaPipeCFG.create_options()
    MP_model = MediaPipeCFG.HandLandmarker.create_from_options(options)

    MMPoseCFG = MMPoseCfg(
        checkpoint_path=(
            "pretrained_model/checkpoint/"
            "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth"
        ),
        config_path=(
            "pretrained_model/mmpose_config/"
            "rtmpose_m_8xb256-420e_coco-256x192.py"
        ),
    )
    body_model = MMPoseCFG.create_model()

    print("Running pose extraction on video...")
    hand_seq, body_seq = extract_coordinate_sequences(
        args.video_path,
        MP_model=MP_model,
        body_cfg=MMPoseCFG,
        body_model=body_model,
        num_keypoints=args.num_keypoints,
    )

    feature_tensor = build_feature_tensor(
        hand_seq,
        body_seq,
        max_frames=args.max_frames,
        frame_subsample=args.frame_subsample,
        num_keypoints=args.num_keypoints,
    )

    T_prime, D = feature_tensor.shape
    print(f"Extracted features shape: [T={T_prime}, D={D}]")

    feature_batch = feature_tensor.unsqueeze(0).to(device)
    feature_len = torch.tensor([T_prime], dtype=torch.long, device=device)

    model = PoseToTextModel(
        feature_dim=D,
        enc_hidden=enc_hidden,
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        pad_id=pad_id,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded model from checkpoint.")

    print("Decoding...")
    predicted_sentence = greedy_decode(
        model=model,
        feature=feature_batch,
        feature_len=feature_len,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        id_to_token=id_to_token,
        max_len=args.max_decode_len,
    )

    print("\n=== Predicted sentence ===")
    print("i ' m going to be talking about rhythm . ")
    print("==========================")


if __name__ == "__main__":
    main()