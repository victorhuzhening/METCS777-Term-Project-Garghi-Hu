from utils import *
from data import *
from model import *  # must contain KeypointCTCModel

import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def build_ctc_targets(labels, label_len, pad_id, bos_id, eos_id, device):
    """
    Build 1D CTC targets and target lengths from padded label tensor.

    labels:    [B, L]  (padded with pad_id, includes <bos>/<eos>)
    label_len: [B]     (true label lengths including bos/eos)
    """
    B, L = labels.shape
    targets_list = []
    target_lengths = []

    for b in range(B):
        seq = labels[b, : label_len[b]].to(device)  # [L_b]
        # remove pad, bos, eos
        mask = (seq != pad_id)
        if bos_id is not None:
            mask &= (seq != bos_id)
        if eos_id is not None:
            mask &= (seq != eos_id)

        filtered = seq[mask]

        # ensure at least one target token for CTC
        if filtered.numel() == 0:
            filtered = torch.tensor([pad_id], dtype=torch.long, device=device)
            tgt_len = 1
        else:
            tgt_len = filtered.numel()

        targets_list.append(filtered)
        target_lengths.append(tgt_len)

    targets = torch.cat(targets_list, dim=0)  # 1D
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)  # [B]

    return targets, target_lengths


def ctc_greedy_decode_batch(log_probs, input_lengths, blank):
    """
    Greedy CTC decoding for a batch.

    log_probs:    [T, B, C]  (log-softmaxed)
    input_lengths:[B]
    returns: list of length B, each is a list[int] of token ids
    """
    T, B, C = log_probs.shape
    best_path = torch.argmax(log_probs, dim=-1)  # [T, B]

    decoded = []
    for b in range(B):
        T_b = int(input_lengths[b].item())
        seq = best_path[:T_b, b].tolist()

        collapsed = []
        prev = blank
        for t in seq:
            if t != prev and t != blank:
                collapsed.append(t)
            prev = t
        decoded.append(collapsed)

    return decoded


def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,        # nn.CTCLoss
    device,
    pad_id: int,
    bos_id: int,
    eos_id: int,
    id_to_token: dict,
):
    model.train()

    total_loss = 0.0
    total_tgt_tokens = 0

    total_bleu = 0.0
    total_rouge = 0.0
    total_sentences = 0

    for batch in loader:
        features = batch["features"].to(device)        # [B, T, D]
        feature_len = batch["feature_len"].to(device)  # [B]
        labels = batch["labels"].to(device)            # [B, L]
        label_len = batch["label_len"].to(device)      # [B]

        # Forward pass: CTC logits
        logits_ctc, input_lengths = model(features, feature_len)  # [T,B,V], [B]
        log_probs = logits_ctc.log_softmax(dim=-1)                # [T,B,V]

        # Build CTC targets
        targets, target_lengths = build_ctc_targets(
            labels, label_len, pad_id, bos_id, eos_id, device
        )

        # CTC loss
        loss = loss_fn(
            log_probs,        # [T,B,C]
            targets,          # 1D (sum target_lengths)
            input_lengths,    # [B]
            target_lengths,   # [B]
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ----- book-keeping -----
        with torch.no_grad():
            non_pad = int(target_lengths.sum().item())
            non_pad = max(non_pad, 1)
            total_loss += loss.item() * non_pad
            total_tgt_tokens += non_pad

            # Greedy CTC decode for metrics
            decoded_ids_batch = ctc_greedy_decode_batch(
                log_probs.detach(), input_lengths, blank=pad_id
            )

            B = labels.size(0)
            for b in range(B):
                pred_ids = decoded_ids_batch[b]

                # Reference: use raw labels (including bos/eos), let tokens_to_text handle them
                ref_ids = labels[b, : label_len[b]].tolist()

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

    avg_loss = total_loss / max(total_tgt_tokens, 1)
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
    bos_id: int,
    eos_id: int,
    id_to_token: dict,
):
    model.eval()

    total_loss = 0.0
    total_tgt_tokens = 0

    total_bleu = 0.0
    total_rouge = 0.0
    total_sentences = 0

    for batch in loader:
        features = batch["features"].to(device)        # [B, T, D]
        feature_len = batch["feature_len"].to(device)  # [B]
        labels = batch["labels"].to(device)            # [B, L]
        label_len = batch["label_len"].to(device)      # [B]

        logits_ctc, input_lengths = model(features, feature_len)  # [T,B,V]
        log_probs = logits_ctc.log_softmax(dim=-1)

        targets, target_lengths = build_ctc_targets(
            labels, label_len, pad_id, bos_id, eos_id, device
        )

        loss = loss_fn(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

        non_pad = int(target_lengths.sum().item())
        non_pad = max(non_pad, 1)
        total_loss += loss.item() * non_pad
        total_tgt_tokens += non_pad

        # Greedy CTC decode
        decoded_ids_batch = ctc_greedy_decode_batch(
            log_probs.detach(), input_lengths, blank=pad_id
        )

        B = labels.size(0)
        for b in range(B):
            pred_ids = decoded_ids_batch[b]
            ref_ids = labels[b, : label_len[b]].tolist()

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

    avg_loss = total_loss / max(total_tgt_tokens, 1)
    avg_bleu = total_bleu / max(total_sentences, 1)
    avg_rouge = total_rouge / max(total_sentences, 1)

    return avg_loss, avg_bleu, avg_rouge


def main():

    parser = argparse.ArgumentParser(description="Train CTC keypoint-to-text model on precomputed ASL data.")
    parser.add_argument("--feature_dir",
                        type=str,
                        default="precomputed_train",
                        required=True,
                        help="Directory with sample_*.pt and vocab_meta.pt.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--enc_hidden", type=int, default=256)  # reused as model dim
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

    # Get bos/eos ids if present
    bos_id = vocab.get("<bos>", None)
    eos_id = vocab.get("<eos>", None)

    num_samples = len(full_dataset)

    # Train/val split
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

    # Build CTC model
    model = KeypointCTCModel(
        input_dim=feature_dim,
        vocab_size=vocab_size,
        dim=args.enc_hidden,
        num_conv_blocks=3,
        num_transformer_blocks=3,
        max_len=5000,
    ).to(device)

    # CTC loss with blank = pad_id
    loss_fn = nn.CTCLoss(
        blank=pad_id,
        zero_infinity=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Total samples: {num_samples} | Train: {num_train} | Val: {num_val}")
    print(f"Vocab size: {vocab_size} | feature dim: {feature_dim}")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_bleu, train_rouge = train_one_epoch(
            model, train_loader, optimizer, loss_fn,
            device, pad_id, bos_id, eos_id, id_to_token,
        )

        if val_loader is not None:
            val_loss, val_bleu, val_rouge = evaluate(
                model, val_loader, loss_fn,
                device, pad_id, bos_id, eos_id, id_to_token,
            )

            print(
                f"[Epoch {epoch:02d}] "
                f"Train loss: {train_loss:.4f}, BLEU-1: {train_bleu:.4f}, ROUGE-1: {train_rouge:.4f} | "
                f"Val loss: {val_loss:.4f}, BLEU-1: {val_bleu:.4f}, ROUGE-1: {val_rouge:.4f}"
            )

            # Save best by val loss
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
            print(
                f"[Epoch {epoch:02d}] "
                f"Train loss: {train_loss:.4f}, BLEU-1: {train_bleu:.4f}, ROUGE-1: {train_rouge:.4f}"
            )

    print("Training complete.")


if __name__ == "__main__":
    main()
