from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

TrajMode = Literal["planner", "reacter", "both"]

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_TRAJ_PATH = DATA_DIR / "traj.csv"
DEFAULT_TRAJ_OKEZUE_PATH = DATA_DIR / "traj_okezue.csv"
PAD_IDX = 0


def build_card_vocab(df: pd.DataFrame) -> dict[str, int]:
    vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1}
    cols = [c for c in (["card"] + [f"hand_{i}" for i in range(4)] + [f"deck_{i}" for i in range(8)]) if c in df.columns]
    if not cols:
        return vocab
    uniq = pd.unique(df[cols].values.ravel())

    for name in uniq:
        if name is None or (isinstance(name, float) and pd.isna(name)):
            name = ""
        else:
            name = str(name).strip()
        if name and name != "nan" and name not in vocab:
            vocab[name] = len(vocab)
    return vocab


def encode_card(vocab: dict[str, int], name: str) -> int:
    return vocab.get(str(name).strip() if pd.notna(name) else "", vocab.get("<unk>", 1))


def _encode_column(vocab: dict[str, int], col: pd.Series) -> np.ndarray:
    return col.map(lambda x: encode_card(vocab, x)).to_numpy(dtype=np.int64)


def pad_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs, lengths, tx, ty, ttime, tcard, reward, done = zip(*batch)
    lengths_t = torch.stack(lengths)
    padded = pad_sequence(seqs, batch_first=True, padding_value=float(PAD_IDX))
    target_xy = torch.stack((torch.stack(tx), torch.stack(ty), torch.stack(ttime)), dim=1)
    target_card = torch.stack(tcard)
    reward_t = torch.stack(reward)
    done_t = torch.stack(done)
    return padded, lengths_t, target_xy, target_card, reward_t, done_t


class TrajDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path = DEFAULT_TRAJ_PATH,
        skip_ability: bool = True,
        mode: TrajMode = "both",
        max_battle_count: int | None = None,
        ):
        self.csv_path = Path(csv_path)
        self.skip_ability = skip_ability
        self.mode = mode
        self.max_battle_count = max_battle_count

        df = pd.read_csv(self.csv_path)
        required = {"battle_id", "x", "y", "card", "time", "side"}
        required.update(f"hand_{i}" for i in range(4))
        required.update(f"deck_{i}" for i in range(8))
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{self.csv_path} missing columns: {sorted(missing)}; has {list(df.columns)}")
        has_reward = "reward" in df.columns
        if not has_reward:
            df["reward"] = 0.0
        df = df.groupby("battle_id").filter(lambda g: g["card"].notna().all())
        df["x"] = df["x"].fillna(499.000000)
        df["y"] = df["y"].fillna(499.000000)

        if skip_ability:
            df = df[~df["card"].astype(str).str.contains("ability", na=False)]

        self.vocab = build_card_vocab(df)
        self.idx_to_card: dict[int, str] = {idx: name for name, idx in self.vocab.items()}
        self.num_cards = len(self.vocab)
        n_battles = df.battle_id.nunique()
        print(f"Total number of battles: {n_battles}")

        df = df.sort_values(["battle_id", "time"])
        df.x = (df.x - 499.000000)/(17500.000000-499.000000)
        df.y = (df.y - 499.000000)/(31500.000000-499.000000)
        df.time = df.time/6000.0
        groups = list(df.groupby("battle_id", sort=False))

        self.samples: list[tuple[torch.Tensor, float, float, float, int, float, float]] = []
        x_col = "x"
        y_col = "y"
        time_col = "time"
        side_col = "side"
        hand_cols = [f"hand_{i}" for i in range(4)]
        deck_cols = [f"deck_{i}" for i in range(8)]

        for i, (_battle_id, grp) in enumerate(groups):
            if self.max_battle_count is not None and i >= self.max_battle_count:
                break
            if i % 500 == 0 and i > 0:
                print(f"  {i} / {min(n_battles, self.max_battle_count or n_battles)} battles")
            grp = grp.reset_index(drop=True)
            if len(grp) < 2:
                continue

            n_rows = len(grp)
            side_is_t = (grp[side_col].astype(str).str.strip() == "t").to_numpy()

            card_idx = _encode_column(self.vocab, grp["card"])
            hand_idxs = np.column_stack([_encode_column(self.vocab, grp[c]) for c in hand_cols])
            deck_idxs = np.column_stack([_encode_column(self.vocab, grp[c]) for c in deck_cols])

            reward_vals = grp["reward"].to_numpy(dtype=np.float64) if has_reward else np.zeros(n_rows, dtype=np.float64)

            x_vals = grp[x_col].to_numpy(dtype=np.float64)
            y_vals = grp[y_col].to_numpy(dtype=np.float64)
            time_vals = grp[time_col].to_numpy(dtype=np.float64)

            for raw_team in ("t", "o"):
                side_enc = side_is_t.astype(np.float32) if raw_team == "t" else (~side_is_t).astype(np.float32)
                target_ok = side_is_t if raw_team == "t" else ~side_is_t
                if raw_team == "o":
                    x_feat = (1.0 - x_vals).astype(np.float32)
                    y_feat = (1.0 - y_vals).astype(np.float32)
                else:
                    x_feat = (x_vals).astype(np.float32)
                    y_feat = (y_vals).astype(np.float32)

                for t in range(1, n_rows):
                    if not target_ok[t]:
                        continue
                    last_move_was_team = side_enc[t - 1] > 0.5
                    if self.mode == "planner" and not last_move_was_team:
                        continue
                    if self.mode == "reacter" and last_move_was_team:
                        continue
                    sl = slice(0, t)
                    steps = np.concatenate([
                        x_feat[sl, np.newaxis],
                        y_feat[sl, np.newaxis],
                        time_vals[sl, np.newaxis].astype(np.float32),
                        side_enc[sl, np.newaxis],
                        hand_idxs[sl].astype(np.float32),
                        deck_idxs[sl].astype(np.float32),
                        card_idx[sl, np.newaxis].astype(np.float32),
                    ], axis=1)
                    seq_tensor = torch.from_numpy(steps)
                    raw_tx = float(grp.iloc[t][x_col])
                    raw_ty = float(grp.iloc[t][y_col])
                    target_x = (1.0 - raw_tx) if raw_team == "o" else raw_tx
                    target_y = (1.0 - raw_ty) if raw_team == "o" else raw_ty
                    target_time = float(time_vals[t])
                    target_card_idx = int(card_idx[t])
                    sample_reward = float(reward_vals[t])
                    sample_done = 1.0 if t == n_rows - 1 else 0.0
                    self.samples.append((seq_tensor, target_x, target_y, target_time, target_card_idx, sample_reward, sample_done))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq, target_x, target_y, target_time, target_card_idx, reward, done = self.samples[idx]
        length = torch.tensor(seq.size(0), dtype=torch.long)
        return (
            seq,
            length,
            torch.tensor(target_x, dtype=torch.float32),
            torch.tensor(target_y, dtype=torch.float32),
            torch.tensor(target_time, dtype=torch.float32),
            torch.tensor(target_card_idx, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def get_vocab(self) -> dict[str, int]:
        return self.vocab.copy()

    def get_num_cards(self) -> int:
        return self.num_cards

    def get_card_name(self, idx: int) -> str:
        return self.idx_to_card.get(int(idx), "<unk>")


def get_traj_dataloader(
    csv_path: str | Path = DEFAULT_TRAJ_PATH,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    skip_ability: bool = True,
    mode: TrajMode = "both",
    max_battle_count: int | None = None,
):
    from torch.utils.data import DataLoader

    dataset = TrajDataset(
        csv_path=csv_path,
        skip_ability=skip_ability,
        mode=mode,
        max_battle_count=max_battle_count,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=pad_collate,
    )


def save_dataset_pt(csv_path: str | Path | None = None, out_dir: str | Path | None = None) -> None:
    csv_path = csv_path or DEFAULT_TRAJ_PATH
    out_dir = Path(out_dir or DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("planner", "reacter", "both"):
        print(f"saving {mode}")
        ds = TrajDataset(str(csv_path), skip_ability=True, mode=mode)
        out_path = out_dir / f"ds_{mode}.pt"
        torch.save(
            {
                "samples": ds.samples,
                "vocab": ds.vocab,
                "idx_to_card": ds.idx_to_card,
                "num_cards": ds.num_cards,
                "mode": mode,
            },
            out_path,
        )
        print(f"Saved {len(ds)} samples -> {out_path}")


class SavedTrajDataset(Dataset):
    def __init__(self, payload: dict):
        self.samples = payload["samples"]
        self.vocab = payload.get("vocab", {})
        self.idx_to_card = payload.get("idx_to_card", {})
        self.num_cards = payload.get("num_cards", len(self.vocab))
        self.mode = payload.get("mode", "unknown")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq, target_x, target_y, target_time, target_card_idx, reward, done = self.samples[idx]
        length = torch.tensor(seq.size(0), dtype=torch.long)
        return (
            seq,
            length,
            torch.tensor(float(target_x), dtype=torch.float32),
            torch.tensor(float(target_y), dtype=torch.float32),
            torch.tensor(float(target_time), dtype=torch.float32),
            torch.tensor(int(target_card_idx), dtype=torch.long),
            torch.tensor(float(reward), dtype=torch.float32),
            torch.tensor(float(done), dtype=torch.float32),
        )

    def get_card_name(self, idx: int) -> str:
        return self.idx_to_card.get(int(idx), "<unk>")


if __name__ == "__main__":
    import sys
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader

    arg1 = sys.argv[1] if len(sys.argv) > 1 else "test"
    if arg1 == "build":
        source = sys.argv[2] if len(sys.argv) > 2 else ""
        if source == "okezue":
            csv_path = DEFAULT_TRAJ_OKEZUE_PATH
        elif source == "win":
            csv_path = DATA_DIR / "traj_win.csv"
        else:
            csv_path = DEFAULT_TRAJ_PATH
        print(f"Building from {csv_path}")
        save_dataset_pt(csv_path=csv_path)
        mode = "both"
    else:
        mode = sys.argv[2] if (len(sys.argv) > 2 and arg1 == "test") else "both"

    pt_path = DATA_DIR / f"ds_{mode}.pt"
    print(f"Testing saved dataset {pt_path}")
    payload = torch.load(pt_path, weights_only=False)
    ds = SavedTrajDataset(payload)
    print(f"SavedTrajDataset: {len(ds)} samples, vocab size = {ds.num_cards}, mode={ds.mode}")

    loader = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=pad_collate)
    for x, lengths, target_xy, target_card, reward, done in loader:
        print(f"x shape: {x.shape}, lengths: {lengths.tolist()}")
        print(
            f"target_xy shape: {target_xy.shape}, target_card shape: {target_card.shape}, "
            f"reward: {reward.shape}, done: {done.shape}"
            f"raw reward: {reward}, raw done: {done}"
        )
        print(
            f"first target: (x,y,time)=({target_xy[0,0].item():.4f}, {target_xy[0,1].item():.4f}, "
            f"{target_xy[0,2].item():.4f}), card={ds.get_card_name(target_card[0].item())}"
        )
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        break
    print("Data loader OK.")
