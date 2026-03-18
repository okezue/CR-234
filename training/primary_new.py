import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from traj_dataloader import (
    DEFAULT_TRAJ_OKEZUE_PATH,
    TrajDataset,
    TrajMode,
    pad_collate,
)




RAW_FEAT_SIZE = 17
CONT_FEAT_SIZE = 4
NUM_CAT_FIELDS = 13


class TrajLSTM(nn.Module):
    """
    LSTM over variable-length traj sequences.
    Uses embeddings for categorical card-id fields and separate heads for:
      - next (x, y)
      - next time
      - next card
    """

    def __init__(
        self,
        num_cards: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        emb_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_cards = num_cards
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb_dim = emb_dim

        self.card_emb = nn.Embedding(
            num_embeddings=num_cards,
            embedding_dim=emb_dim,
            padding_idx=0,
        )

        lstm_input_size = CONT_FEAT_SIZE + NUM_CAT_FIELDS * emb_dim

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.trunk = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.xy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        )

        self.t_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.card_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_cards),
        )

    def _build_lstm_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, 17)
        returns: (B, T, 4 + 13 * emb_dim)
        """
        cont = x[:, :, :CONT_FEAT_SIZE]
        cats = x[:, :, CONT_FEAT_SIZE:].long()


        cats = cats.clamp(min=0, max=self.num_cards - 1)

        emb = self.card_emb(cats)
        emb = emb.flatten(start_dim=2)

        z = torch.cat([cont, emb], dim=-1)
        return z

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: (B, T, 17)
        lengths: (B,)
        returns:
          pred_xy:   (B, 2)
          pred_t:    (B, 1)
          pred_card: (B, num_cards)
        """
        z = self._build_lstm_input(x)

        packed = pack_padded_sequence(
            z,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.lstm(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True)

        B = x.size(0)
        idx = (lengths - 1).clamp(min=0)
        last_out = outputs[torch.arange(B, device=x.device), idx, :]

        h = self.trunk(last_out)
        pred_xy = self.xy_head(h)
        pred_t = self.t_head(h)
        pred_card = self.card_head(h)
        return pred_xy, pred_t, pred_card


def _unpack_dataset_item(item):
    """
    Supports:
      1) (seq, length, target_x, target_y, target_time, target_card)
      2) (seq, length, target_xy, target_card), where target_xy is shape (3,)
      3) (seq, length, target_x, target_y, target_time, target_card, reward, done) — traj with reward/done
    """
    if len(item) == 8:
        seq, length, target_x, target_y, target_time, target_card, _reward, _done = item
        target_xy = torch.tensor(
            [float(target_x), float(target_y), float(target_time)],
            dtype=torch.float32,
        )
        return seq, length, target_xy, int(target_card)
    if len(item) == 6:
        seq, length, target_x, target_y, target_time, target_card = item
        target_xy = torch.tensor(
            [float(target_x), float(target_y), float(target_time)],
            dtype=torch.float32,
        )
        return seq, length, target_xy, int(target_card)
    if len(item) == 4:
        seq, length, target_xy, target_card = item
        if not torch.is_tensor(target_xy):
            target_xy = torch.tensor(target_xy, dtype=torch.float32)
        target_xy = target_xy.float()
        return seq, length, target_xy, int(target_card)
    raise ValueError(f"Unexpected dataset item format with len={len(item)}")


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_xy: nn.Module,
    criterion_t: nn.Module,
    criterion_card: nn.Module,
    device: torch.device,
):
    """
    Returns:
      avg_xy_loss, avg_t_loss, avg_card_loss, card_acc
    """
    model.eval()
    total_xy = 0.0
    total_t = 0.0
    total_card = 0.0
    correct_card = 0
    n_samples = 0

    with torch.no_grad():
        for x, lengths, target_xy, target_card, _reward, _done in dataloader:
            x = x.to(device, dtype=torch.float32)
            lengths = lengths.to(device)
            target_xy = target_xy.to(device, dtype=torch.float32)
            target_card = target_card.to(device).long()

            target_xy_only = target_xy[:, :2]
            target_t = target_xy[:, 2]

            pred_xy, pred_t, pred_card = model(x, lengths)

            loss_xy = criterion_xy(pred_xy, target_xy_only)
            loss_t = criterion_t(pred_t.squeeze(-1), target_t)
            loss_card = criterion_card(pred_card, target_card)

            b = x.size(0)
            total_xy += loss_xy.item() * b
            total_t += loss_t.item() * b
            total_card += loss_card.item() * b
            correct_card += (pred_card.argmax(dim=1) == target_card).sum().item()
            n_samples += b

    model.train()

    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    return (
        total_xy / n_samples,
        total_t / n_samples,
        total_card / n_samples,
        correct_card / n_samples,
    )


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: dict,
    config: dict,
):
    torch.save(
        {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
        },
        path,
    )


def train_traj_lstm(
    csv_path: str | None = None,
    mode: TrajMode = "both",
    num_epochs: int = 10,
    batch_size: int = 64,
    hidden_size: int = 128,
    num_layers: int = 2,
    emb_dim: int = 16,
    learning_rate: float = 1e-3,
    dropout: float = 0.2,
    loss_t_weight: float = 1.0,
    loss_xy_weight: float = 1.0,
    loss_card_weight: float = 1.0,
    skip_ability: bool = False,
    val_frac: float = 0.2,
    max_battle_count: int | None = None,
    plot_curve: bool = True,
    curve_save_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = "checkpoints",
    dataset_cache_path: str | Path | None = None,
    seed: int = 42,
):
    if csv_path is None:
        csv_path = DEFAULT_TRAJ_OKEZUE_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if dataset_cache_path is None:
        dataset_cache_path = f"ds_{mode}.pt"
    dataset_cache_path = Path(dataset_cache_path)

    if dataset_cache_path.exists():
        print(f"Loading cached dataset from {dataset_cache_path}")
        full_dataset = torch.load(dataset_cache_path, weights_only=False)
    else:
        print("Building dataset from CSV")
        full_dataset = TrajDataset(
            csv_path=csv_path,
            skip_ability=skip_ability,
            mode=mode,
            max_battle_count=max_battle_count,
        )
        torch.save(full_dataset, dataset_cache_path)
        print(f"Saved dataset cache to {dataset_cache_path}")

    n = len(full_dataset)
    if n < 2:
        raise ValueError(f"Dataset too small: {n}")

    n_val = max(1, int(n * val_frac))
    n_train = n - n_val
    if n_train <= 0:
        raise ValueError(f"Train split is empty. n={n}, val_frac={val_frac}")

    split_gen = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=split_gen,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate,
    )

    num_cards = full_dataset.get_num_cards()

    print(f"Dataset size: {n}")
    print(f"Train size:   {n_train}")
    print(f"Val size:     {n_val}")
    print(f"Num cards:    {num_cards}")

    model = TrajLSTM(
        num_cards=num_cards,
        hidden_size=hidden_size,
        num_layers=num_layers,
        emb_dim=emb_dim,
        dropout=dropout,
    ).to(device)

    criterion_xy = nn.MSELoss()
    criterion_t = nn.MSELoss()
    criterion_card = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    history = {
        "train_xy": [],
        "train_t": [],
        "train_card": [],
        "val_xy": [],
        "val_t": [],
        "val_card": [],
        "val_acc": [],
    }

    log_dir = Path("runs") / f"{mode}_hs{hidden_size}_nl{num_layers}_emb{emb_dim}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    global_step = 0

    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_dir is not None:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "num_cards": num_cards,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "emb_dim": emb_dim,
        "dropout": dropout,
        "mode": mode,
    }

    model.train()

    for epoch in range(num_epochs):
        epoch_loss_xy = 0.0
        epoch_loss_t = 0.0
        epoch_loss_card = 0.0
        n_samples = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
            unit="batch",
        )

        for x, lengths, target_xy, target_card, _reward, _done in pbar:
            x = x.to(device, dtype=torch.float32)
            lengths = lengths.to(device)
            target_xy = target_xy.to(device, dtype=torch.float32)
            target_card = target_card.to(device).long()

            target_xy_only = target_xy[:, :2]
            target_t = target_xy[:, 2]

            optimizer.zero_grad()

            pred_xy, pred_t, pred_card = model(x, lengths)

            loss_xy = criterion_xy(pred_xy, target_xy_only)
            loss_t = criterion_t(pred_t.squeeze(-1), target_t)
            loss_card = criterion_card(pred_card, target_card)

            loss = (
                loss_xy_weight * loss_xy
                + loss_t_weight * loss_t
                + loss_card_weight * loss_card
            )

            if not torch.isfinite(loss):
                tqdm.write(
                    f"Skipping non-finite batch: "
                    f"loss_xy={loss_xy.item():.4f}, "
                    f"loss_t={loss_t.item():.4f}, "
                    f"loss_card={loss_card.item():.4f}"
                )
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            writer.add_scalar("batch/loss_xy", loss_xy.item(), global_step)
            writer.add_scalar("batch/loss_t", loss_t.item(), global_step)
            writer.add_scalar("batch/loss_card", loss_card.item(), global_step)
            writer.add_scalar("batch/loss_total", loss.item(), global_step)
            global_step += 1

            b = x.size(0)
            epoch_loss_xy += loss_xy.item() * b
            epoch_loss_t += loss_t.item() * b
            epoch_loss_card += loss_card.item() * b
            n_samples += b

            pbar.set_postfix(
                loss_xy=f"{loss_xy.item():.3f}",
                loss_t=f"{loss_t.item():.3f}",
                loss_card=f"{loss_card.item():.3f}",
            )

        if n_samples == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - no valid batches")
            continue

        train_xy = epoch_loss_xy / n_samples
        train_t = epoch_loss_t / n_samples
        train_card = epoch_loss_card / n_samples

        val_xy, val_t, val_card, val_acc = _evaluate(
            model,
            val_loader,
            criterion_xy,
            criterion_t,
            criterion_card,
            device,
        )

        history["train_xy"].append(train_xy)
        history["train_t"].append(train_t)
        history["train_card"].append(train_card)
        history["val_xy"].append(val_xy)
        history["val_t"].append(val_t)
        history["val_card"].append(val_card)
        history["val_acc"].append(val_acc)

        writer.add_scalar("epoch/train_xy", train_xy, epoch + 1)
        writer.add_scalar("epoch/train_t", train_t, epoch + 1)
        writer.add_scalar("epoch/train_card", train_card, epoch + 1)

        writer.add_scalar("epoch/val_xy", val_xy, epoch + 1)
        writer.add_scalar("epoch/val_t", val_t, epoch + 1)
        writer.add_scalar("epoch/val_card", val_card, epoch + 1)
        writer.add_scalar("epoch/val_acc", val_acc, epoch + 1)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_xy={train_xy:.4f} train_t={train_t:.4f} train_card={train_card:.4f} | "
            f"val_xy={val_xy:.4f} val_t={val_t:.4f} val_card={val_card:.4f} val_acc={val_acc:.4f}"
        )

        if ckpt_dir is not None:
            ckpt_path = ckpt_dir / f"{mode}_epoch_{epoch + 1}.pt"
            _save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                epoch + 1,
                history,
                config,
            )
            print(f"Checkpoint saved to {ckpt_path}")

    if plot_curve and len(history["train_xy"]) > 0:
        _plot_training_curve(history, save_path=curve_save_path, mode=mode)

    writer.close()
    return model, history, config


def _plot_training_curve(
    history: dict,
    save_path: str | Path | None = None,
    mode: str = "both",
):
    epochs = range(1, len(history["train_xy"]) + 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].plot(epochs, history["train_xy"], label="train")
    axes[0].plot(epochs, history["val_xy"], label="val")
    axes[0].set_title("XY loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_t"], label="train")
    axes[1].plot(epochs, history["val_t"], label="val")
    axes[1].set_title("Time loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history["train_card"], label="train")
    axes[2].plot(epochs, history["val_card"], label="val")
    axes[2].set_title("Card loss")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(epochs, history["val_acc"], label="val acc")
    axes[3].set_title("Val card accuracy")
    axes[3].set_xlabel("Epoch")
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = Path.cwd() / f"training_curve_{mode}.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curve saved to {save_path}")


def _load_model_from_checkpoint(
    model_path: str | Path,
    device: torch.device,
    fallback_num_cards: int,
    fallback_hidden_size: int = 128,
    fallback_num_layers: int = 2,
    fallback_emb_dim: int = 16,
    fallback_dropout: float = 0.2,
):
    state = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(state, dict) and "config" in state and "state_dict" in state:
        cfg = state["config"]
        model = TrajLSTM(
            num_cards=cfg.get("num_cards", fallback_num_cards),
            hidden_size=cfg.get("hidden_size", fallback_hidden_size),
            num_layers=cfg.get("num_layers", fallback_num_layers),
            emb_dim=cfg.get("emb_dim", fallback_emb_dim),
            dropout=cfg.get("dropout", fallback_dropout),
        ).to(device)
        model.load_state_dict(state["state_dict"])
        return model, cfg

    model = TrajLSTM(
        num_cards=fallback_num_cards,
        hidden_size=fallback_hidden_size,
        num_layers=fallback_num_layers,
        emb_dim=fallback_emb_dim,
        dropout=fallback_dropout,
    ).to(device)
    model.load_state_dict(state)
    return model, {
        "num_cards": fallback_num_cards,
        "hidden_size": fallback_hidden_size,
        "num_layers": fallback_num_layers,
        "emb_dim": fallback_emb_dim,
        "dropout": fallback_dropout,
    }


def test_saved_model(
    model_path: str,
    csv_path: str | None = None,
    cached_ds_path: str | None = None,
    mode: TrajMode = "both",
    num_examples: int = 3,
    hidden_size: int = 128,
    num_layers: int = 2,
    emb_dim: int = 16,
    dropout: float = 0.2,
    skip_ability: bool = False,
    max_battle_count: int | None = 500,
    seed: int = 42,
):
    """
    Load a saved model and print example trajectories with predictions vs ground truth.
    """
    if csv_path is None:
        csv_path = DEFAULT_TRAJ_OKEZUE_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cached_ds_path is None:
        dataset = TrajDataset(
            csv_path=csv_path,
            skip_ability=skip_ability,
            mode=mode,
            max_battle_count=max_battle_count,
        )
    else:
        dataset = torch.load(cached_ds_path, weights_only=False)

    num_cards = dataset.get_num_cards()

    model, cfg = _load_model_from_checkpoint(
        model_path=model_path,
        device=device,
        fallback_num_cards=num_cards,
        fallback_hidden_size=hidden_size,
        fallback_num_layers=num_layers,
        fallback_emb_dim=emb_dim,
        fallback_dropout=dropout,
    )
    model.eval()

    print("Loaded model config:")
    print(cfg)

    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=rng).tolist()[:num_examples]
    if not indices:
        print("No examples in dataset.")
        return

    for ex_idx, idx in enumerate(indices):
        item = dataset[idx]
        seq, length, target_xy, target_card = _unpack_dataset_item(item)

        L = seq.size(0)

        print(f"\n{'=' * 60}")
        print(f"Example {ex_idx + 1}/{num_examples} (seq length = {L})")
        print(f"{'=' * 60}")
        print("Trajectory:")

        for i in range(L):
            x_i = seq[i, 0].item()
            y_i = seq[i, 1].item()
            t_i = seq[i, 2].item()
            side_i = seq[i, 3].item()
            card_idx_i = int(seq[i, 16].item())
            card_name_i = dataset.get_card_name(card_idx_i)
            side_str = "team" if side_i > 0.5 else "opponent"

            print(
                f"  step {i + 1}: "
                f"time={t_i:.4f} side={side_str} card={card_name_i} "
                f"x={x_i:.4f} y={y_i:.4f}"
            )

        x_batch = seq.unsqueeze(0).to(device, dtype=torch.float32)
        lengths_batch = length.unsqueeze(0).to(device)

        with torch.no_grad():
            pred_xy, pred_t, pred_card = model(x_batch, lengths_batch)

        pred_x = pred_xy[0, 0].item()
        pred_y = pred_xy[0, 1].item()
        pred_time = pred_t[0, 0].item()
        pred_card_idx = pred_card[0].argmax().item()
        pred_card_name = dataset.get_card_name(pred_card_idx)

        true_x = float(target_xy[0].item())
        true_y = float(target_xy[1].item())
        true_time = float(target_xy[2].item())
        true_card_idx = int(target_card)
        true_card_name = dataset.get_card_name(true_card_idx)


        last_step = seq[L - 1]
        hand_at_pred = [
            dataset.get_card_name(int(last_step[j].item()))
            for j in range(4, 8)
        ]
        deck_at_pred = [
            dataset.get_card_name(int(last_step[j].item()))
            for j in range(8, 16)
        ]
        print("\nHand at prediction time (last step in trajectory):")
        print(f"  {', '.join(hand_at_pred)}")
        print("Deck at prediction time:")
        print(f"  {', '.join(deck_at_pred)}")

        print("\nGround truth next move:")
        print(
            f"  x={true_x:.4f} y={true_y:.4f} "
            f"time={true_time:.4f} card={true_card_name}"
        )
        print("Model predicts next move:")
        print(
            f"  x={pred_x:.4f} y={pred_y:.4f} "
            f"time={pred_time:.4f} card={pred_card_name}"
        )

    print(f"\n{'=' * 60}")
    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise ValueError("Usage: python train.py [planner|reacter]")

    mode = sys.argv[1]
    if mode not in {"planner", "reacter"}:
        raise ValueError("Mode must be 'planner' or 'reacter'")

    print(f"Training {mode}")





























    test_saved_model(
        model_path="checkpoints/planner_epoch_7.pt",
        mode=mode,
        num_examples=3,
        max_battle_count=500,
        cached_ds_path="ds_planner.pt"
    )
