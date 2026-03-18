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

        self.card_emb = nn.Embedding(
            num_embeddings=num_cards,
            embedding_dim=emb_dim,
            padding_idx=0,
        )

        lstm_input_size = CONT_FEAT_SIZE + NUM_CAT_FIELDS * emb_dim
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm_card = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.lstm_xy = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.lstm_t = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.xy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 2),
        )

        self.t_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        self.card_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_cards),
        )

    def _build_lstm_input(self, x: torch.Tensor) -> torch.Tensor:
        cont = x[:, :, :CONT_FEAT_SIZE]
        cats = x[:, :, CONT_FEAT_SIZE:].long()
        cats = cats.clamp(min=0, max=self.num_cards - 1)

        emb = self.card_emb(cats)
        emb = emb.flatten(start_dim=2)

        return torch.cat([cont, emb], dim=-1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        z = self._build_lstm_input(x)

        packed = pack_padded_sequence(
            z,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_out_xy, _ = self.lstm_xy(packed)
        packed_out_t, _ = self.lstm_t(packed)
        packed_out_card, _ = self.lstm_card(packed)

        outputs_xy, _ = pad_packed_sequence(packed_out_xy, batch_first=True)
        outputs_t, _ = pad_packed_sequence(packed_out_t, batch_first=True)
        outputs_card, _ = pad_packed_sequence(packed_out_card, batch_first=True)

        b = x.size(0)
        idx = (lengths - 1).clamp(min=0)

        last_out_xy = outputs_xy[torch.arange(b, device=x.device), idx, :]
        last_out_t = outputs_t[torch.arange(b, device=x.device), idx, :]
        last_out_card = outputs_card[torch.arange(b, device=x.device), idx, :]

        pred_xy = self.xy_head(last_out_xy)
        pred_t = self.t_head(last_out_t)
        pred_card = self.card_head(last_out_card)

        return pred_xy, pred_t, pred_card


def load_initial_weights(
    model: nn.Module,
    init_model_path: str | Path,
    device: torch.device,
    strict: bool = False,
):
    init_model_path = Path(init_model_path)
    state = torch.load(init_model_path, map_location=device, weights_only=False)

    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    print(f"Loaded initial weights from {init_model_path}")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion_xy: nn.Module,
    criterion_t: nn.Module,
    criterion_card: nn.Module,
    device: torch.device,
):
    model.eval()

    total_xy = 0.0
    total_t = 0.0
    total_card = 0.0
    total_total = 0.0
    correct_card = 0
    n_samples = 0

    with torch.no_grad():
        for x, lengths, target_xy, target_card in dataloader:
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
            loss_total = (
                loss_xy * _evaluate.loss_xy_weight
                + loss_t * _evaluate.loss_t_weight
                + loss_card * _evaluate.loss_card_weight
            )

            batch_size = x.size(0)
            total_xy += loss_xy.item() * batch_size
            total_t += loss_t.item() * batch_size
            total_card += loss_card.item() * batch_size
            total_total += loss_total.item() * batch_size
            correct_card += (pred_card.argmax(dim=1) == target_card).sum().item()
            n_samples += batch_size

    model.train()

    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    return (
        total_xy / n_samples,
        total_t / n_samples,
        total_card / n_samples,
        total_total / n_samples,
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
    name: str = "Noname",
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
    init_model_path: str | Path | None = None,
    init_strict: bool = False,
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

    if init_model_path is not None:
        load_initial_weights(
            model=model,
            init_model_path=init_model_path,
            device=device,
            strict=init_strict,
        )

    criterion_xy = nn.MSELoss()
    criterion_t = nn.MSELoss()
    criterion_card = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )

    history = {
        "train_total": [],
        "train_xy": [],
        "train_t": [],
        "train_card": [],
        "val_total": [],
        "val_xy": [],
        "val_t": [],
        "val_card": [],
        "val_acc": [],
    }

    log_dir = Path("runs") / f"{name}_{mode}_hs{hidden_size}_nl{num_layers}_emb{emb_dim}"
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
        "init_model_path": str(init_model_path) if init_model_path is not None else None,
        "init_strict": init_strict,
        "loss_t_weight": loss_t_weight,
        "loss_xy_weight": loss_xy_weight,
        "loss_card_weight": loss_card_weight,
    }

    _evaluate.loss_t_weight = loss_t_weight
    _evaluate.loss_xy_weight = loss_xy_weight
    _evaluate.loss_card_weight = loss_card_weight

    model.train()

    for epoch in range(num_epochs):
        epoch_loss_xy = 0.0
        epoch_loss_t = 0.0
        epoch_loss_card = 0.0
        epoch_loss_total = 0.0
        n_samples = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            leave=True,
            unit="batch",
        )

        for x, lengths, target_xy, target_card in pbar:
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
                    "Skipping non-finite batch: "
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

            batch_size = x.size(0)
            epoch_loss_xy += loss_xy.item() * batch_size
            epoch_loss_t += loss_t.item() * batch_size
            epoch_loss_card += loss_card.item() * batch_size
            epoch_loss_total += loss.item() * batch_size
            n_samples += batch_size

            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                xy=f"{loss_xy.item():.3f}",
                t=f"{loss_t.item():.3f}",
                card=f"{loss_card.item():.3f}",
            )

        if n_samples == 0:
            print(f"Epoch {epoch + 1}/{num_epochs} - no valid batches")
            continue

        train_xy = epoch_loss_xy / n_samples
        train_t = epoch_loss_t / n_samples
        train_card = epoch_loss_card / n_samples
        train_total = epoch_loss_total / n_samples

        val_xy, val_t, val_card, val_total, val_acc = _evaluate(
            model,
            val_loader,
            criterion_xy,
            criterion_t,
            criterion_card,
            device,
        )

        history["train_total"].append(train_total)
        history["train_xy"].append(train_xy)
        history["train_t"].append(train_t)
        history["train_card"].append(train_card)
        history["val_total"].append(val_total)
        history["val_xy"].append(val_xy)
        history["val_t"].append(val_t)
        history["val_card"].append(val_card)
        history["val_acc"].append(val_acc)

        writer.add_scalar("epoch/train_total", train_total, epoch + 1)
        writer.add_scalar("epoch/train_xy", train_xy, epoch + 1)
        writer.add_scalar("epoch/train_t", train_t, epoch + 1)
        writer.add_scalar("epoch/train_card", train_card, epoch + 1)
        writer.add_scalar("epoch/val_total", val_total, epoch + 1)
        writer.add_scalar("epoch/val_xy", val_xy, epoch + 1)
        writer.add_scalar("epoch/val_t", val_t, epoch + 1)
        writer.add_scalar("epoch/val_card", val_card, epoch + 1)
        writer.add_scalar("epoch/val_acc", val_acc, epoch + 1)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_total={train_total:.4f} | "
            f"train_xy={train_xy:.4f} | "
            f"train_t={train_t:.4f} | "
            f"train_card={train_card:.4f} | "
            f"val_total={val_total:.4f} | "
            f"val_xy={val_xy:.4f} | "
            f"val_t={val_t:.4f} | "
            f"val_card={val_card:.4f} | "
            f"val_acc={val_acc:.4f}"
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

    if plot_curve and len(history["train_card"]) > 0:
        _plot_training_curve(history, save_path=curve_save_path, mode=mode)

    writer.close()
    return model, history, config


def _plot_training_curve(
    history: dict,
    save_path: str | Path | None = None,
    mode: str = "both",
):
    epochs = range(1, len(history["train_card"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_card"], label="train card")
    axes[0].plot(epochs, history["val_card"], label="val card")
    axes[0].set_title("Card loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["val_acc"], label="val acc")
    axes[1].set_title("Val card accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = Path.cwd() / f"training_curve_{mode}.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curve saved to {save_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise ValueError(
            "Usage: python train.py [planner|reacter] [name] [optional_init_model_path]"
        )

    mode = sys.argv[1]
    name = sys.argv[2]
    init_model_path = sys.argv[3] if len(sys.argv) >= 4 else None

    if mode not in {"planner", "reacter"}:
        raise ValueError("Mode must be 'planner' or 'reacter'")

    print(f"Training mode={mode}")

    model, history, config = train_traj_lstm(
        mode=mode,
        name=name,
        init_model_path=init_model_path,
        init_strict=False,
        num_epochs=4,
        batch_size=128,
        hidden_size=128,
        num_layers=2,
        emb_dim=16,
        learning_rate=1e-3,
        dropout=0.2,
        val_frac=0.15,
        plot_curve=True,
        curve_save_path=f"results/{name}_{mode}_curve.png",
        checkpoint_dir=f"checkpoints/{name}",
        dataset_cache_path=f"ds_{mode}.pt",
        loss_t_weight=1000.0,
        loss_xy_weight=100.0,
        loss_card_weight=1.0,
    )

    Path("models").mkdir(parents=True, exist_ok=True)
    final_model_path = f"models/{name}_{mode}_model.pt"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": config,
            "history": history,
        },
        final_model_path,
    )
    print(f"Final model saved to {final_model_path}")
