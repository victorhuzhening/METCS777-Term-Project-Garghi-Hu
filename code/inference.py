import os
import torch
import mediapipe as mp

from model import EncoderDecoderModel
from data import *
from pretrained_models import *


def load_vocab_and_special_ids(data_dir: str):
    """
    Load vocab + pad/bos/eos IDs from vocab_meta.pt.
    """
    vocab_meta_path = os.path.join(data_dir, "vocab_meta.pt")
    vocab_meta = torch.load(vocab_meta_path, map_location="cpu")

    vocab = vocab_meta["vocab"]
    pad_id = vocab_meta["pad_id"]

    # Edge case: Infer BOS/EOS if not explicit
    bos_id = vocab_meta.get("bos_id", vocab.get("<bos>"))
    eos_id = vocab_meta.get("eos_id", vocab.get("<eos>"))

    if bos_id is None or eos_id is None:
        raise ValueError(
            "Could not find <bos> or <eos> IDs in vocab_meta.pt."
        )

    id_to_token = {idx: tok for tok, idx in vocab.items()}
    return vocab, id_to_token, pad_id, bos_id, eos_id


def decode_ids_to_text(
    token_ids: torch.Tensor,
    id_to_token,
    bos_id: int,
    eos_id: int,
    pad_id: int,
) -> str:
    """
    Convert a 1D tensor of token IDs into a string using id_to_token.
    Skips BOS, PAD, and stops at EOS.
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    tokens = []
    for tid in token_ids:
        if tid == bos_id:
            # skip BOS in final text
            continue
        if tid == eos_id or tid == pad_id:
            break
        token = id_to_token.get(tid, "<unk>")
        tokens.append(token)

    return " ".join(tokens)


@torch.no_grad()
def run_single_greedy_decode(
    model: EncoderDecoderModel,
    features: torch.Tensor,     # [1, T, D_in]
    feature_len: torch.Tensor,  # [1]
    id_to_token,
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_len: int = None,
) -> str:
    """
    Greedy decode helper. See model.py
    """
    model.eval()

    generated_ids = model.greedy_decode(
        pose_feats=features,
        pose_len=feature_len,
        max_len=max_len,
    )

    sentence = decode_ids_to_text(
        generated_ids,
        id_to_token=id_to_token,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
    )
    return sentence



def build_features_for_video(
    video_path: str,
    MP_model,
    body_cfg,
    body_model,
    frame_subsample: int = 1,
    max_frames: int | None = None,
    num_keypoints: int = 17,
) -> tuple[torch.Tensor, int]:
    """
    Feature extraction from raw video, further details see data.py.
    """
    hand_seq, body_seq = extract_coordinate_sequences_for_video(
        video_path=video_path,
        MP_model=MP_model,
        body_cfg=body_cfg,
        body_model=body_model,
        num_keypoints=num_keypoints,
    )

    feature_tensor = build_feature_tensor_from_sequences(
        hand_seq=hand_seq,
        body_seq=body_seq,
        frame_subsample=frame_subsample,
        max_frames=max_frames,
    )
    feature_len = feature_tensor.size(0)

    return feature_tensor, feature_len



def main():
    """
    Run this file from main or using terminal to translate a single ASL video into English sentence.
    Manually configure code to point to your video file/checkpoint/pretrained models
    TODO: Convert to argparse factory function
    """
    CODE_DIR = os.getcwd()
    BASE_DIR = os.path.dirname(CODE_DIR)

    CKPT_PATH = "../data/best_encoder_decoder_model.pt"

    # Path to raw video you want to translate
    INPUT_PATH = "C:/Users/victo/Documents/CS777/Temp Data Storage/raw_videos/-70D86eMmIc_3-5-rgb_front.mp4"

    # Directory with vocab_meta.pt
    PRECOMP_DIR = os.path.join(BASE_DIR, "data", "precomputed_train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load checkpoint
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state_dict = ckpt["model_state_dict"]
    train_args = ckpt.get("args", {})

    # load vocab
    vocab, id_to_token, pad_id, bos_id, eos_id = load_vocab_and_special_ids(PRECOMP_DIR)
    vocab_size = len(vocab)

    # initialize pose extraction models
    MediaPipeCFG = MediaPipeCfg("../data/pretrained_model/hand_landmarker.task")
    options = MediaPipeCFG.create_options()
    MP_model = MediaPipeCFG.HandLandmarker.create_from_options(options)

    body_cfg = MMPoseCfg(
        checkpoint_path='../data/pretrained_model/checkpoint/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.pth',
        config_path='../data/pretrained_model/mmpose_config/rtmpose_m_8xb256-420e_coco-256x192.py')
    body_model = body_cfg.create_model()

    # Safely pull params from training args
    frame_subsample = int(train_args.get("frame_subsample", 1))
    num_keypoints = int(train_args.get("num_keypoints", 17))
    max_frames = train_args.get("max_frames", None)
    if max_frames is not None:
        max_frames = int(max_frames)

    print(f"Extracting pose features from video: {INPUT_PATH}")
    feature_tensor, feature_len = build_features_for_video(
        video_path=INPUT_PATH,
        MP_model=MP_model,
        body_cfg=body_cfg,
        body_model=body_model,
        frame_subsample=frame_subsample,
        max_frames=max_frames,
        num_keypoints=num_keypoints,
    )

    T, D = feature_tensor.shape

    # Build encoder-decoder model
    d_model = train_args.get("d_model", 768)
    num_encoder_layers = train_args.get("num_encoder_layers", 6)
    num_decoder_layers = train_args.get("num_decoder_layers", 6)
    nhead = train_args.get("nhead", 8)
    dim_feedforward = train_args.get("dim_feedforward", 2048)
    dropout = train_args.get("dropout", 0.1)
    max_tgt_len = train_args.get("max_tgt_len", 128)

    model = EncoderDecoderModel(
        feature_dim=D,
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_tgt_len=max_tgt_len,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
    ).to(device)

    model.load_state_dict(state_dict)

    # We use only a single sample in batch
    feature_batch = feature_tensor.unsqueeze(0).to(device)
    feature_len_batch = torch.tensor(
        [feature_len],
        dtype=torch.long,
        device=device,
    )

    predicted_sentence = run_single_greedy_decode(
        model=model,
        features=feature_batch,
        feature_len=feature_len_batch,
        id_to_token=id_to_token,
        bos_id=bos_id,
        eos_id=eos_id,
        pad_id=pad_id,
        max_len=max_tgt_len,
    )

    print("\n====== Inference from raw video ======")
    print(f"Predicted sentence:\n  {predicted_sentence}")

if __name__ == "__main__":
    main()
