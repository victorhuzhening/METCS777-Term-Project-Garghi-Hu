import argparse
import time

import numpy as np
import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import *
from utils import bleu1, rouge1_f1
from model import EncoderDecoderModel
from tokenizer import *
import matplotlib.pyplot as plt


def shifted_labels_collate(batch, pad_id: int, max_target_token_len: int = None):
    """
    Collate function mainly to shift label sequence one token to the right.
    batch: list of dicts from PrecomputedASLData, each with:
       - "features": [T, D] FloatTensor
       - "feature_len": int
       - "label_ids": LongTensor [L]
       - "raw_label": str

    Returns:
       features:           [B, T_max, D]
       feature_len:        [B]
       decoder_input_ids:  [B, L-1]  (shifted-right inputs)
       decoder_target_ids: [B, L-1]  (targets)
       raw_labels:         list[str]
    """
    feature_list = []
    feature_len_list = []

    label_list = []
    raw_labels = []

    for sample in batch:
        features = sample["features"]
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        else:
            features = features.float()

        T, D = features.shape
        feature_list.append(features)
        feature_len_list.append(T)

        label_ids = sample["label_ids"]
        if isinstance(label_ids, np.ndarray):
            label_ids = torch.from_numpy(label_ids).long()
        else:
            label_ids = label_ids.long()

        # Optional: truncate to max token length
        if max_target_token_len is not None:
            label_ids = label_ids[:max_target_token_len] 

        label_list.append(label_ids)
        raw_labels.append(sample["raw_label"])

    B = len(batch)
    D = feature_list[0].shape[1]
    max_T = max(feature_len_list)
    max_L = max(len(l) for l in label_list)

    # Pad to avoid NaN
    features = torch.zeros(B, max_T, D, dtype=torch.float32)
    for i, feat in enumerate(feature_list):
        T = feat.shape[0]
        features[i, :T] = feat

    feature_len = torch.tensor(feature_len_list, dtype=torch.long)

    labels_batch = torch.full(
        (B, max_L), fill_value=pad_id, dtype=torch.long
    )
    for i, label in enumerate(label_list):
        L = len(label)
        labels_batch[i, :L] = label

    # Shift token for decoder input (target label):
    #   decoder_input_ids:  [<bos>, tok1, ..., tok_{L-1}]
    #   decoder_target_ids: [tok1,  ..., tok_{L-1}, <eos>]
    # If max_L == 1, this will produce empty sequences - AVOID
    if max_L <= 1:
        raise RuntimeError("Label sequences are too short (max_L <= 1).")

    decoder_input_ids = labels_batch[:, :-1].contiguous()   # contiguous safety check
    decoder_target_ids = labels_batch[:, 1:].contiguous()   

    return {
        "features": features,                      # [B, T_max, D]
        "feature_len": feature_len,                # [B]
        "decoder_input_ids": decoder_input_ids,    # [B, L-1]
        "decoder_target_ids": decoder_target_ids,  # [B, L-1]
        "raw_labels": raw_labels,                  # list[str]
    }



def ids_to_tokens(
    ids,
    id_to_token: dict,
    pad_id: int,
    bos_id: int,
    eos_id: int,
):
    """
    Strip special tokens and convert a sequence of ids to token list.
    """
    tokens = []
    for i in ids:
        if i in (pad_id, bos_id, eos_id):
            continue
        tokens.append(id_to_token.get(int(i), "<unk>"))      # <unk> is our default token for tokens the model cannot learn 
    return tokens


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    pad_id: int,
    id_to_token: dict,
    bos_id: int,
    eos_id: int,
):
    model.train()

    total_loss = 0.0
    total_tokens = 0

    # Evaluation metrics
    total_bleu = 0.0
    total_rouge = 0.0
    total_sentences = 0

    for batch in loader:
        features = batch["features"].to(device)         
        feature_len = batch["feature_len"].to(device)    
        decoder_input = batch["decoder_input_ids"].to(device)  
        decoder_target = batch["decoder_target_ids"].to(device) 

        logits = model(features, feature_len, decoder_input)   
        B, L, V = logits.shape

        loss = loss_fn(
            logits.reshape(B * L, V),
            decoder_target.reshape(B * L),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            non_pad = (decoder_target != pad_id)
            supervised = non_pad.sum().item()
            supervised = max(supervised, 1)
            total_loss += loss.item() * supervised
            total_tokens += supervised

            # Greedy decode for metrics
            pred_ids_batch = logits.argmax(dim=-1)      
            gt_ids_batch = decoder_target                    

            for i in range(B):
                pred_ids = pred_ids_batch[i].tolist()
                gt_ids = gt_ids_batch[i].tolist()

                pred_tokens = ids_to_tokens(
                    pred_ids, id_to_token, pad_id, bos_id, eos_id
                )
                gt_tokens = ids_to_tokens(
                    gt_ids, id_to_token, pad_id, bos_id, eos_id
                )

                if len(gt_tokens) == 0:
                    continue

                total_bleu += bleu1(pred_tokens, gt_tokens)
                total_rouge += rouge1_f1(pred_tokens, gt_tokens)
                total_sentences += 1

    avg_loss = total_loss / max(total_tokens, 1)
    avg_bleu = total_bleu / max(total_sentences, 1)
    avg_rouge = total_rouge / max(total_sentences, 1)

    return avg_loss, avg_bleu, avg_rouge


@torch.no_grad()
def evaluate(
    model,
    loader,
    loss_fn,
    device,
    pad_id: int,
    id_to_token: dict,
    bos_id: int,
    eos_id: int,
):
    """
    Exactly the same function as train_one_epoch just without updating optimizer
    for gradient descent.
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    total_bleu = 0.0
    total_rouge = 0.0
    total_sentences = 0

    for batch in loader:
        features = batch["features"].to(device)          
        feature_len = batch["feature_len"].to(device)    
        decoder_input = batch["decoder_input_ids"].to(device)  
        decoder_target = batch["decoder_target_ids"].to(device) 

        logits = model(features, feature_len, decoder_input)   
        B, L, V = logits.shape

        loss = loss_fn(
            logits.reshape(B * L, V),
            decoder_target.reshape(B * L),
        )

        non_pad = (decoder_target != pad_id)
        supervised = non_pad.sum().item()
        supervised = max(supervised, 1)
        total_loss += loss.item() * supervised
        total_tokens += supervised

        pred_ids_batch = logits.argmax(dim=-1)  
        gt_ids_batch = decoder_target                 

        for i in range(B):
            pred_ids = pred_ids_batch[i].tolist()
            gt_ids = gt_ids_batch[i].tolist()

            pred_tokens = ids_to_tokens(
                pred_ids, id_to_token, pad_id, bos_id, eos_id
            )
            gt_tokens = ids_to_tokens(
                gt_ids, id_to_token, pad_id, bos_id, eos_id
            )

            if len(gt_tokens) == 0:
                continue

            total_bleu += bleu1(pred_tokens, gt_tokens)
            total_rouge += rouge1_f1(pred_tokens, gt_tokens)
            total_sentences += 1

    avg_loss = total_loss / max(total_tokens, 1)
    avg_bleu = total_bleu / max(total_sentences, 1)
    avg_rouge = total_rouge / max(total_sentences, 1)

    return avg_loss, avg_bleu, avg_rouge



def main():
    """
    Factory function using argument parsers to configure custom training hyperparameters.
    Can be run using terminal or calling current file main.
    """
    parser = argparse.ArgumentParser(
        description="Train Encoder-Decoder model with basic tokenizer."
    )
    parser.add_argument(
        "--feature_dir",
        type=str,
        default="../data/precomputed_train",
        required=False,
        help="Directory with sample_*.pt files and vocab_meta.pt file.",
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8)
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100)
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-4)
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="plateau",
        choices=["none", "plateau"],
        help="ReduceLROnPlateau on val loss.",
    )
    parser.add_argument(
        "--num_pose_layers",
        type=int,
        default=6,
        help="Number of Transformer Encoder layers.",
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=6,
        help="Number of Transformer Decoder layers.",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=512,
        help="Hidden layer dimension for Transformer blocks (encoder/decoder).",
    )
    parser.add_argument(
        "--max_target_len",
        type=int,
        default=64,
        help="Maximum target length for truncate.",
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=0,
        help="Recommended to keep at 0 for local, else configure for cloud env.")
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.25,
        help="(Optional) Split training data into train and val",
    )
    parser.add_argument(
        "--save_path", 
        type=str, 
        default="../data/best_encoder_decoder_model.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load dataset
    full_dataset = PrecomputedASLData(args.feature_dir)
    num_samples = len(full_dataset)

    # Get vocab & special tokens from vocab_meta.pt
    vocab = full_dataset.vocab
    pad_id = full_dataset.pad_id
    bos_id = vocab["<bos>"]
    eos_id = vocab["<eos>"]
    vocab_size = len(vocab)

    id_to_token = {i: tok for tok, i in vocab.items()}

    # Optional train/val split
    if args.val_split <= 0.0:
        train_dataset = full_dataset
        val_dataset = None
        num_train = num_samples
        num_val = 0
    else:
        num_val = int(num_samples * args.val_split)
        num_val = max(1, num_val) if num_samples > 1 else 0
        num_train = num_samples - num_val

        if num_train <= 0:
            num_train = max(1, num_samples - 1)
            num_val = num_samples - num_train

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [num_train, num_val],
            generator=torch.Generator().manual_seed(42),  # try different seed
        )

    # Configure dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: shifted_labels_collate(
            b, pad_id, max_target_token_len=args.max_target_len,
        ),
    )

    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: shifted_labels_collate(
                b, pad_id, max_target_token_len=args.max_target_len
            ),
        )
    else:
        val_loader = None

    # Infer feature_dim from one batch - required for model
    first_batch = next(iter(train_loader))
    feature_dim = first_batch["features"].shape[-1]

    # Build model
    model = EncoderDecoderModel(
        feature_dim=feature_dim,
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_encoder_layers=args.num_pose_layers,
        num_decoder_layers=args.num_decoder_layers,
        pad_id=pad_id,
        bos_id=bos_id,
        eos_id=eos_id,
        max_tgt_len=args.max_target_len,
    ).to(device)

    # Ignore padding for loss
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    # AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # LR scheduler
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=True,
        )
    else:
        scheduler = None

    print(f"Total samples: {num_samples} | Train: {num_train} | Val: {num_val}")
    print(f"Vocab size: {vocab_size} | feature dim: {feature_dim}")
    print(f"Using AdamW with weight_decay={args.weight_decay}")
    print(f"lr_scheduler={args.lr_scheduler}")

    best_val_loss = float("inf")

    # Lists to store epoch-wise metrics
    epoch_list = []
    epoch_train_times = []
    epoch_val_losses = []
    epoch_val_rouges = []
    epoch_val_bleus = []

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        train_loss, train_bleu, train_rouge = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device,
            pad_id,
            id_to_token,
            bos_id,
            eos_id,
        )

        val_loss = None
        val_bleu = None
        val_rouge = None

        if val_loader is not None:
            val_loss, val_bleu, val_rouge = evaluate(
                model,
                val_loader,
                loss_fn,
                device,
                pad_id,
                id_to_token,
                bos_id,
                eos_id,
            )

            print(
                f"[Epoch {epoch:02d}] "
                f"Train loss: {train_loss:.4f}, BLEU-1: {train_bleu:.4f}, ROUGE-1: {train_rouge:.4f} | "
                f"Val loss: {val_loss:.4f}, BLEU-1: {val_bleu:.4f}, ROUGE-1: {val_rouge:.4f}"
            )

            # We only save models based on validation loss since ROUGE and BLEU can be "tricked"
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "vocab": vocab,
                        "pad_id": pad_id,
                        "bos_id": bos_id,
                        "eos_id": eos_id,
                        "args": vars(args),
                    },
                    args.save_path,
                )
                print(f"  → New best model saved to {args.save_path}")

            if scheduler is not None:
                scheduler.step(val_loss)
        else:
            print(
                f"[Epoch {epoch:02d}] "
                f"Train loss: {train_loss:.4f}, BLEU-1: {train_bleu:.4f}, ROUGE-1: {train_rouge:.4f}"
            )
            if scheduler is not None:
                scheduler.step(train_loss)

        epoch_time = time.time() - epoch_start_time

        epoch_list.append(epoch)
        epoch_train_times.append(epoch_time)
        if val_loader is not None:
            epoch_val_losses.append(val_loss)
            epoch_val_bleus.append(val_bleu)
            epoch_val_rouges.append(val_rouge)

    print("Training complete.")


    # Plot epoch-wise metrics in matplotlib graphs
    # TODO: needs to be refactored to TensorBoard
    if len(epoch_list) > 0:
        plt.figure()
        plt.plot(epoch_list, epoch_train_times, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Train time (seconds)")
        plt.title("Train Time vs Epoch")
        plt.grid(True)
        plt.savefig("train_time_vs_epoch.png", bbox_inches="tight")
        plt.close()
        print("Saved train_time_vs_epoch.png")

    if val_loader is not None and len(epoch_val_losses) == len(epoch_list):
        # Val loss vs epoch
        plt.figure()
        plt.plot(epoch_list, epoch_val_losses, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Val Loss")
        plt.title("Val Loss vs Epoch")
        plt.grid(True)
        plt.savefig("val_loss_vs_epoch.png", bbox_inches="tight")
        plt.close()
        print("Saved val_loss_vs_epoch.png")

        # Val ROUGE-1 vs epoch
        plt.figure()
        plt.plot(epoch_list, epoch_val_rouges, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Val ROUGE-1 F1")
        plt.title("Val ROUGE-1 vs Epoch")
        plt.grid(True)
        plt.savefig("val_rouge1_vs_epoch.png", bbox_inches="tight")
        plt.close()
        print("Saved val_rouge1_vs_epoch.png")

        # Val BLEU-1 vs epoch
        plt.figure()
        plt.plot(epoch_list, epoch_val_bleus, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Val BLEU-1")
        plt.title("Val BLEU-1 vs Epoch")
        plt.grid(True)
        plt.savefig("val_bleu1_vs_epoch.png", bbox_inches="tight")
        plt.close()
        print("Saved val_bleu1_vs_epoch.png")
    else:
        print("No validation set available; skipping val metric plots.")


if __name__ == "__main__":
    main()
