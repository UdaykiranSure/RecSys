"""
train.py
--------
Training loop for the ideological sequential recommender.
"""

import argparse
import copy
import pickle
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from config import cfg
from data.dataset import build_dataloaders
from evaluate import run_evaluation
from loss.ideology_loss import IdeologyLoss
from models.recommender import IdeologyRecommender


def resolve_device(device_cfg: str) -> str:
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_cfg


def load_graph_tensors(processed_dir: str | Path, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    graph_path = Path(processed_dir) / "graph_data-3.pkl"
    with open(graph_path, "rb") as f:
        graph_data = pickle.load(f)

    graph_x = graph_data.x.float().to(device)
    graph_edge_index = graph_data.edge_index.to(device)

    # Z-score normalize each feature column to prevent exploding embeddings
    # (barbera scores and log_degree can have very different scales)
    mean = graph_x.mean(dim=0, keepdim=True)
    std  = graph_x.std(dim=0, keepdim=True).clamp(min=1e-6)
    graph_x = (graph_x - mean) / std

    return graph_x, graph_edge_index


def build_model(num_items: int, config) -> IdeologyRecommender:
    return IdeologyRecommender(
        num_items=num_items,
        num_graph_nodes=0,
        embed_dim=config.tweet_enc.output_dim,
        graph_node_feat=config.graph.node_feat_dim,
        hidden_dim=config.graph.hidden_dim,
        num_sage_layers=config.graph.num_layers,
        num_sasrec_layers=config.sasrec.num_layers,
        num_heads=config.sasrec.num_heads,
        max_seq_len=config.data.max_seq_len,
        dropout=config.graph.dropout,
    )


def train(config=cfg):
    config = copy.deepcopy(config)
    device = resolve_device(config.train.device)

    print(f"Using device: {device}")
    print("=" * 60)

    train_dl, val_dl, test_dl, item_catalog = build_dataloaders(
        processed_dir=config.paths.processed_dir,
        max_seq_len=config.data.max_seq_len,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        delta=config.loss.delta,
        val_holdout=config.data.val_holdout,
        test_holdout=config.data.test_holdout,
        hard_neg_band=config.loss.hard_neg_band,
        num_negatives=config.loss.num_negatives,
        min_item_freq=config.data.min_item_freq,
    )

    graph_x, graph_edge_index = load_graph_tensors(config.paths.processed_dir, device)

    model = build_model(item_catalog.num_items, config).to(device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    # LR schedule: linear warmup → cosine decay
    warmup_steps = config.train.warmup_steps
    total_steps  = config.train.num_epochs * len(train_dl)
    warmup_sched = LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched = CosineAnnealingLR(
        optimizer, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
    )

    loss_fn = IdeologyLoss(
        alpha_bpr=config.loss.alpha_bpr,
        alpha_ideology=config.loss.alpha_ideology,
        alpha_smoothness=config.loss.alpha_smoothness,
    )

    checkpoint_dir = Path(config.paths.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_model.pt"

    best_hit10 = -1.0
    no_improve = 0

    for epoch in range(1, config.train.num_epochs + 1):
        t0 = time.time()
        model.train()

        # ── Pre-compute full-graph embeddings ONCE per epoch ──────────────
        # Running GraphSAGE on every batch is the dominant time cost.
        # We compute it here (detached) so each batch only does an index lookup.
        # Note: graph_encoder weights are effectively frozen w.r.t. gradient
        # updates. The SASRec + Fusion components still train normally.
        with torch.no_grad():
            all_graph_embs = model.graph_encoder(graph_x, graph_edge_index)  # (N, d)
        # ─────────────────────────────────────────────────────────────────

        running_total = 0.0
        running_bpr = 0.0
        running_ideo = 0.0
        running_smooth = 0.0
        steps = 0
        nan_steps = 0

        for batch in train_dl:
            optimizer.zero_grad()

            user_idx = batch["user_idx"].to(device)
            seq_item_ids = batch["history_items"].to(device)
            seq_ideo_scores = batch["history_states"].to(device)
            pos_item_ids = batch["target_item"].to(device)
            pos_ideo_scores = batch["target_ideo"].to(device)
            neg_item_ids = batch["neg_item_idx"].to(device)
            neg_ideo_scores = batch["neg_ideo"].to(device)
            ideo_current = batch["ideo_current"].to(device)
            direction = batch["direction"].to(device)
            delta = batch["delta"].to(device)

            _, pos_scores, neg_scores = model(
                graph_x=graph_x,
                graph_edge_index=graph_edge_index,
                user_graph_idx=user_idx,
                seq_item_ids=seq_item_ids,
                seq_ideo_scores=seq_ideo_scores,
                pos_item_ids=pos_item_ids,
                pos_ideo_scores=pos_ideo_scores,
                neg_item_ids=neg_item_ids,
                neg_ideo_scores=neg_ideo_scores,
                all_graph_embs=all_graph_embs,
            )

            total, loss_parts = loss_fn(
                pos_scores=pos_scores,
                neg_scores=neg_scores,
                ideo_pos=pos_ideo_scores,
                ideo_current=ideo_current,
                direction=direction,
                delta=delta,
            )

            # Skip NaN batches instead of letting them corrupt weights
            if not torch.isfinite(total):
                nan_steps += 1
                if nan_steps <= 3:
                    print(
                        f"  [warn] NaN/Inf loss at step {steps+1} "
                        f"(pos={pos_scores.mean():.3f}, neg={neg_scores.mean():.3f}) — skipping"
                    )
                optimizer.zero_grad()
                steps += 1
                continue

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
            scheduler.step()

            running_total += float(total.item())
            running_bpr += loss_parts["loss_bpr"]
            running_ideo += loss_parts["loss_ideology"]
            running_smooth += loss_parts["loss_smoothness"]
            steps += 1

        good_steps = steps - nan_steps
        train_loss = running_total / max(good_steps, 1)

        val_metrics = run_evaluation(
            model=model,
            dataloader=val_dl,
            item_catalog=item_catalog,
            graph_x=graph_x,
            graph_edge_index=graph_edge_index,
            device=device,
            k_values=config.eval.k_values,
            split="val",
        )
        hit10 = val_metrics.get("hit@10", 0.0)

        cur_lr = scheduler.get_last_lr()[0]
        nan_note = f" nan_skipped={nan_steps}" if nan_steps else ""
        print(
            f"Epoch {epoch}/{config.train.num_epochs} "
            f"[{time.time() - t0:.1f}s] "
            f"train={train_loss:.4f} "
            f"bpr={running_bpr / max(good_steps, 1):.4f} "
            f"ideo={running_ideo / max(good_steps, 1):.4f} "
            f"smooth={running_smooth / max(good_steps, 1):.4f} "
            f"lr={cur_lr:.2e} "
            f"val_hit@10={hit10:.4f}"
            f"{nan_note}"
        )

        if hit10 > best_hit10:
            best_hit10 = hit10
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "best_hit10": best_hit10,
                    "config": config,
                },
                best_path,
            )
            print(f"Saved best checkpoint to {best_path}")
        else:
            no_improve += 1
            print(f"No val improvement ({no_improve}/{config.train.early_stopping})")

        if no_improve >= config.train.early_stopping:
            print("Early stopping triggered")
            break

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    test_metrics = run_evaluation(
        model=model,
        dataloader=test_dl,
        item_catalog=item_catalog,
        graph_x=graph_x,
        graph_edge_index=graph_edge_index,
        device=device,
        k_values=config.eval.k_values,
        split="test",
    )

    print("Final test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    return model, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.epochs is not None:
        cfg.train.num_epochs = args.epochs
    if args.lr is not None:
        cfg.train.learning_rate = args.lr
    if args.delta is not None:
        cfg.loss.delta = args.delta
    if args.alpha is not None:
        cfg.loss.alpha_ideology = args.alpha
    if args.beta is not None:
        cfg.loss.alpha_smoothness = args.beta
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.device is not None:
        cfg.train.device = args.device

    train(cfg)
