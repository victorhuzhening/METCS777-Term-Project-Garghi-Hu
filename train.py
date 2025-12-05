from utils import *
from data import *
from model import *
from mode_test import PoseToTextTransformer, build_model_from_dims

import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    device,
    pad_id: int,
    id_to_token: dict,
):
    model.train()

    total_loss = 0.0
    total_tokens = 0

    total_bleu = 0.0
    total_rouge = 0.0
    total_sentences = 0

    for batch in loader:
        features = batch["features"].to(device)          # [B, T, D]
        feature_len = batch["feature_len"].to(device)  # [B]
        labels = batch["labels"].to(device)      # [B, L]

        # Forward with teacher forcing; model expected to use labels as decoder input.
        logits = model(features, feature_len, labels)   # [B, L-1, V]

        # Targets are labels shifted left
        target = labels[:, 1:]                   # [B, L-1]

        B, Lm1, V = logits.shape

        loss = loss_fn(
            logits.reshape(B * Lm1, V),
            target.reshape(B * Lm1),
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ----- book-keeping -----
        with torch.no_grad():
            non_pad = (target != pad_id).sum().item()
            non_pad = max(non_pad, 1)
            total_loss += loss.item() * non_pad
            total_tokens += non_pad

            # Greedy predictions for metrics
            pred_ids_batch = logits.argmax(dim=-1)   # [B, L-1]
            ref_ids_batch = target                   # [B, L-1]

            for b in range(B):
                pred_ids = pred_ids_batch[b].tolist()
                ref_ids = ref_ids_batch[b].tolist()

                pred_str = tokens_to_text(
                    pred_ids, id_to_token, pad_id=pad_id,
                    bos_token="<bos>", eos_token="<eos>",
                )
                ref_str = tokens_to_text(
                    ref_ids, id_to_token, pad_id=pad_id,
                    bos_token="<bos>", eos_token="<eos>",
                )

                pred_toks = pred_str.split()
                ref_toks = ref_str.split()

                if len(ref_toks) == 0:
                    continue

                total_bleu += bleu1(pred_toks, ref_toks)
                total_rouge += rouge1_f1(pred_toks, ref_toks)
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
):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    total_bleu = 0.0
    total_rouge = 0.0
    total_sentences = 0

    for batch in loader:
        features = batch["features"].to(device)          # [B, T, D]
        feature_len = batch["feature_len"].to(device)  # [B]
        labels = batch["labels"].to(device)      # [B, L]

        logits = model(features, feature_len, labels)   # [B, L-1, V]
        target = labels[:, 1:]                   # [B, L-1]

        B, Lm1, V = logits.shape

        loss = loss_fn(
            logits.reshape(B * Lm1, V),
            target.reshape(B * Lm1),
        )

        non_pad = (target != pad_id).sum().item()
        non_pad = max(non_pad, 1)
        total_loss += loss.item() * non_pad
        total_tokens += non_pad

        # Greedy decode for metrics
        pred_ids_batch = logits.argmax(dim=-1)   # [B, L-1]
        ref_ids_batch = target                   # [B, L-1]

        for b in range(B):
            pred_ids = pred_ids_batch[b].tolist()
            ref_ids = ref_ids_batch[b].tolist()

            pred_str = tokens_to_text(
                pred_ids, id_to_token, pad_id=pad_id,
                bos_token="<bos>", eos_token="<eos>",
            )
            ref_str = tokens_to_text(
                ref_ids, id_to_token, pad_id=pad_id,
                bos_token="<bos>", eos_token="<eos>",
            )

            pred_toks = pred_str.split()
            ref_toks = ref_str.split()

            if len(ref_toks) == 0:
                continue

            total_bleu += bleu1(pred_toks, ref_toks)
            total_rouge += rouge1_f1(pred_toks, ref_toks)
            total_sentences += 1

    avg_loss = total_loss / max(total_tokens, 1)
    avg_bleu = total_bleu / max(total_sentences, 1)
    avg_rouge = total_rouge / max(total_sentences, 1)

    return avg_loss, avg_bleu, avg_rouge


def main():

    parser = argparse.ArgumentParser(description="Train feature-to-text model on precomputed ASL data.")
    parser.add_argument("--feature_dir",
                        type=str,
                        default="data/pre_train_data",
                        required=True,
                        help="Directory with sample_*.pt and vocab_meta.pt.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--enc_hidden", type=int, default=256)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["gru", "transformer"],
        help="Choose GRU baseline or Transformer (default).",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation.",
    )
    parser.add_argument("--save_path", type=str, default="best_model_val.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load full dataset (all precomputed samples)
    full_dataset = PrecomputedASLData(args.feature_dir)
    vocab = full_dataset.vocab
    pad_id = full_dataset.pad_id
    id_to_token = build_id_to_token(vocab)
    vocab_size = len(vocab)

    num_samples = len(full_dataset)

    # No validation set; use all data for training
    if args.val_split <= 0.0:
        train_dataset = full_dataset
        val_dataset = None
        num_train = num_samples
        num_val = 0
    else:
        num_val = int(num_samples * args.val_split)
        num_val = max(1, num_val) if num_samples > 1 else 0
        num_train = num_samples - num_val

        # Corner case: small data sample - force at least one train sample
        if num_train <= 0:
            num_train = max(1, num_samples - 1)
            num_val = num_samples - num_train

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [num_train, num_val],
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: asl_collate_func(b, pad_id=pad_id),
    )

    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=lambda b: asl_collate_func(b, pad_id=pad_id),
        )
    else:
        val_loader = None

    # Infer feature_dim from one batch
    first_batch = next(iter(train_loader))
    feature_dim = first_batch["features"].shape[-1]

    # Build model (default: Transformer from mode_test)
    if args.model_type == "gru":
        model = PoseToTextModel(
            feature_dim=feature_dim,
            enc_hidden=args.enc_hidden,
            vocab_size=vocab_size,
            emb_dim=args.emb_dim,
            pad_id=pad_id,
        ).to(device)
    else:
        model = build_model_from_dims(
            feature_dim=feature_dim,
            vocab_size=vocab_size,
            pad_id=pad_id,
        ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Total samples: {num_samples} | Train: {num_train} | Val: {num_val}")
    print(f"Vocab size: {vocab_size} | feature dim: {feature_dim} | model: {args.model_type}")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_bleu, train_rouge = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            device, pad_id, id_to_token,
        )

        if val_loader is not None:
            val_loss, val_bleu, val_rouge = evaluate(
                model, val_loader, loss_fn,
                device, pad_id, id_to_token,
            )

            print(
                f"[Epoch {epoch:02d}] "
                f"Train loss: {train_loss:.4f}, BLEU-1: {train_bleu:.4f}, ROUGE-1: {train_rouge:.4f} | "
                f"Val loss: {val_loss:.4f}, BLEU-1: {val_bleu:.4f}, ROUGE-1: {val_rouge:.4f}"
            )

            # Simple best-checkpoint saving by val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "vocab": vocab,
                        "pad_id": pad_id,
                        "args": vars(args),
                    },
                    args.save_path,
                )
                print(f"  → New best model saved to {args.save_path}")
        else:
            # No validation set: just print train metrics
            print(
                f"[Epoch {epoch:02d}] "
                f"Train loss: {train_loss:.4f}, BLEU-1: {train_bleu:.4f}, ROUGE-1: {train_rouge:.4f}"
            )

    print("Training complete.")


if __name__ == "__main__":
    main()
