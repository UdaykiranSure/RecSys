"""
train.py
--------
End-to-end training loop for the Ideological Sequential Recommender.

Usage:
    python train.py

Or with custom config overrides:
    python train.py --epochs 50 --lr 0.001 --delta 0.2 --alpha 0.5 --beta 0.3
"""

import argparse
import os
import pickle
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import Config, cfg
from data.dataset import build_dataloaders, ItemCatalog
from models.graph_module import GraphData
from models.recommender import IdeologicalRecommender
from loss.ideology_loss import build_loss
from evaluate import run_evaluation


# ── Graph cache: pre-compute node embeddings once per epoch ──────────────────

@torch.no_grad()
def compute_graph_embeddings(
    model:      IdeologicalRecommender,
    graph_data: GraphData,
    device:     str,
) -> torch.Tensor:
    """
    Run GraphSAGE over the full graph once and cache all node embeddings.
    Returned tensor lives on `device`.
    """
    model.eval()
    node_features = graph_data.node_features.to(device)
    adj_matrix    = graph_data.adj.to(device)
    all_node_embs = model.encode_graph(node_features, adj_matrix)   # [N, D]
    model.train()
    return all_node_embs


# ── Single training step ──────────────────────────────────────────────────────

def train_step(
    model:          IdeologicalRecommender,
    batch:          dict,
    all_node_embs:  torch.Tensor,
    user_node_idx:  torch.Tensor,
    loss_fn,
    optimizer:      torch.optim.Optimizer,
    grad_clip:      float,
    device:         str,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad()

    # Move batch to device
    hist_item_idx    = batch["hist_item_idx"].to(device)
    hist_ideo        = batch["hist_ideo"].to(device)
    padding_mask     = batch["padding_mask"].to(device)
    target_item_idx  = batch["target_item_idx"].to(device)
    target_ideo      = batch["target_ideo"].to(device)
    neg_item_idx     = batch["neg_item_idx"].to(device)
    neg_ideo         = batch["neg_ideo"].to(device)
    ideo_current     = batch["ideo_current"].to(device)
    direction        = batch["direction"].to(device)
    delta            = batch["delta"].to(device)

    # Forward pass
    outputs = model(
        hist_item_idx    = hist_item_idx,
        hist_ideo        = hist_ideo,
        padding_mask     = padding_mask,
        target_item_idx  = target_item_idx,
        target_ideo      = target_ideo,
        neg_item_idx     = neg_item_idx,
        neg_ideo         = neg_ideo,
        all_node_embs    = all_node_embs,
        user_node_idx    = user_node_idx.to(device),
    )

    # Compute loss
    loss_dict = loss_fn(
        score_pos    = outputs["score_pos"],
        score_neg    = outputs["score_neg"],
        target_ideo  = target_ideo,
        ideo_current = ideo_current,
        direction    = direction,
        delta        = delta,
    )

    loss_dict["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    return {k: v.item() for k, v in loss_dict.items()}


# ── Main training loop ────────────────────────────────────────────────────────

def train(config: Config = cfg):
    device = config.train.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    print(f"Using device: {device}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    print("Loading data...")
    train_dl, val_dl, test_dl, item_catalog = build_dataloaders(
        processed_dir  = config.data.processed_dir,
        max_seq_len    = config.data.max_seq_len,
        batch_size     = config.train.batch_size,
        num_workers    = config.train.num_workers,
        delta          = config.loss.delta,
        val_holdout    = config.data.val_holdout,
        test_holdout   = config.data.test_holdout,
        hard_neg_band  = config.loss.hard_negative_band,
        num_negatives  = config.loss.num_negatives,
    )

    # ── Load graph ────────────────────────────────────────────────────────
    print("Loading graph...")
    graph_data = GraphData(
        graph_data_path = Path(config.data.processed_dir) / "graph_data.pkl",
        device          = device,
    )

    # Load user → graph node index mapping
    with open(Path(config.data.processed_dir) / "scored_rt_sequences.pkl", "rb") as f:
        scored_seqs = pickle.load(f)
    user_ids      = list(scored_seqs.keys())
    user_node_idx = graph_data.get_user_indices(user_ids).to(device)

    # Build per-batch user node index lookup
    # (maps user position in dataset to graph node index)
    user_id_to_node = {uid: graph_data.node_id_map.get(uid, 0)
                       for uid in user_ids}

    # ── Build model ───────────────────────────────────────────────────────
    print("Building model...")
    model = IdeologicalRecommender(
        num_items         = item_catalog.num_items,
        embed_dim         = config.model.embed_dim,
        graph_input_dim   = config.model.graph_input_dim,
        graph_hidden_dim  = config.model.graph_hidden_dim,
        graph_num_layers  = config.model.graph_num_layers,
        sasrec_num_heads  = config.model.sasrec_num_heads,
        sasrec_num_layers = config.model.sasrec_num_layers,
        sasrec_ffn_dim    = config.model.sasrec_ffn_dim,
        sasrec_max_len    = config.model.sasrec_max_len,
        fusion_hidden     = config.model.fusion_hidden_dim,
        dropout           = config.model.graph_dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr           = config.train.learning_rate,
        weight_decay = config.train.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode     = "max",
        patience = config.train.scheduler_patience,
        factor   = config.train.scheduler_factor,
        verbose  = True,
    ) if config.train.use_scheduler else None

    loss_fn = build_loss(
        alpha = config.loss.alpha,
        beta  = config.loss.beta,
    )

    # ── Training state ────────────────────────────────────────────────────
    best_metric    = 0.0
    patience_count = 0
    checkpoint_dir = Path(config.train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting training...")
    print("=" * 60)

    for epoch in range(1, config.train.epochs + 1):
        epoch_start = time.time()

        # Pre-compute graph embeddings once per epoch (graph is static)
        all_node_embs = compute_graph_embeddings(model, graph_data, device)

        # ── Train ──────────────────────────────────────────────────────
        model.train()
        train_losses = {"total": 0., "bpr": 0., "direction": 0., "smoothness": 0.}
        num_steps = 0

        for step, batch in enumerate(train_dl):
            # Get graph node indices for this batch's users
            # (stored in dataset but need mapping here)
            # Note: user_node_idx is looked up per-sample via a collate fn
            # For simplicity we pass a zeros tensor as placeholder;
            # replace with proper per-batch lookup in production
            B = batch["hist_item_idx"].shape[0]
            batch_node_idx = torch.zeros(B, dtype=torch.long)  # placeholder
            # TODO: replace with proper per-sample user→node mapping
            # e.g. batch["user_node_idx"] if added to dataset __getitem__

            losses = train_step(
                model         = model,
                batch         = batch,
                all_node_embs = all_node_embs,
                user_node_idx = batch_node_idx,
                loss_fn       = loss_fn,
                optimizer     = optimizer,
                grad_clip     = config.train.grad_clip,
                device        = device,
            )

            for k in train_losses:
                train_losses[k] += losses[k]
            num_steps += 1

            if step % config.train.log_every_n_steps == 0:
                print(f"  Epoch {epoch} | Step {step}/{len(train_dl)} | "
                      f"Loss: {losses['total']:.4f}  "
                      f"(BPR: {losses['bpr']:.4f}  "
                      f"Dir: {losses['direction']:.4f}  "
                      f"Smooth: {losses['smoothness']:.4f})")

        # Average train losses
        for k in train_losses:
            train_losses[k] /= max(num_steps, 1)

        # ── Evaluate ───────────────────────────────────────────────────
        if epoch % config.train.eval_every_n_epochs == 0:
            val_metrics = run_evaluation(
                model         = model,
                dataloader    = val_dl,
                item_catalog  = item_catalog,
                all_node_embs = all_node_embs,
                device        = device,
                k_values      = config.eval.k_values,
                split         = "val",
            )

            monitor = val_metrics.get(config.train.early_stop_metric, 0.0)
            epoch_time = time.time() - epoch_start

            print(f"\nEpoch {epoch}/{config.train.epochs}  [{epoch_time:.1f}s]")
            print(f"  Train loss    : {train_losses['total']:.4f}  "
                  f"(BPR={train_losses['bpr']:.4f}  "
                  f"Dir={train_losses['direction']:.4f}  "
                  f"Smooth={train_losses['smoothness']:.4f})")
            print(f"  Val metrics   : " +
                  "  ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

            # Scheduler step
            if scheduler is not None:
                scheduler.step(monitor)

            # ── Checkpoint ────────────────────────────────────────────
            if monitor > best_metric:
                best_metric = monitor
                patience_count = 0
                ckpt_path = checkpoint_dir / "best_model.pt"
                torch.save({
                    "epoch"       : epoch,
                    "model_state" : model.state_dict(),
                    "optim_state" : optimizer.state_dict(),
                    "best_metric" : best_metric,
                    "config"      : config,
                }, ckpt_path)
                print(f"  ✓ Saved best model  ({config.train.early_stop_metric}="
                      f"{best_metric:.4f})  → {ckpt_path}")
            else:
                patience_count += 1
                print(f"  No improvement ({patience_count}/{config.train.early_stop_patience})")

            # ── Early stopping ────────────────────────────────────────
            if patience_count >= config.train.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # ── Final test evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading best model for final test evaluation...")
    ckpt = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    all_node_embs = compute_graph_embeddings(model, graph_data, device)

    test_metrics = run_evaluation(
        model         = model,
        dataloader    = test_dl,
        item_catalog  = item_catalog,
        all_node_embs = all_node_embs,
        device        = device,
        k_values      = config.eval.k_values,
        split         = "test",
    )

    print("\nFinal Test Metrics:")
    for k, v in test_metrics.items():
        print(f"  {k:<20} : {v:.4f}")

    return model, test_metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--delta",      type=float, default=None)
    parser.add_argument("--alpha",      type=float, default=None)
    parser.add_argument("--beta",       type=float, default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--device",     type=str,   default=None)
    args = parser.parse_args()

    # Override config with CLI args
    if args.epochs:     cfg.train.epochs         = args.epochs
    if args.lr:         cfg.train.learning_rate  = args.lr
    if args.delta:      cfg.loss.delta           = args.delta
    if args.alpha:      cfg.loss.alpha           = args.alpha
    if args.beta:       cfg.loss.beta            = args.beta
    if args.batch_size: cfg.train.batch_size     = args.batch_size
    if args.device:     cfg.train.device         = args.device

    train(cfg)
