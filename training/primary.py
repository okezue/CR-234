import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class DummySequenceDataset(Dataset):
    """
    Tiny dummy dataset of 1D sequences.
    Per-timestep targets: at step t we predict the next value (position t+1).
    """

    def __init__(self, num_samples: int = 128, seq_len: int = 5):
        super().__init__()
        self.seq_len = seq_len
        # Simple predictable pattern (normalized)
        data = torch.arange(0, (num_samples + seq_len) + 1, dtype=torch.float32)
        data = (data - data.mean()) / data.std()

        self.inputs = []
        self.targets = []
        for i in range(num_samples):
            seq = data[i : i + seq_len].unsqueeze(-1)  # (seq_len, 1)
            # At step t predict value at t+1: targets[t] = data[i + t + 1]
            tgt = data[i + 1 : i + seq_len + 1].unsqueeze(-1)  # (seq_len, 1)
            self.inputs.append(seq)
            self.targets.append(tgt)

        self.inputs = torch.stack(self.inputs)  # (N, seq_len, 1)
        self.targets = torch.stack(self.targets)  # (N, seq_len, 1)

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]


class SmallLSTMRegressor(nn.Module):
    """
    Small LSTM that feeds each timestep's hidden state into a fully
    connected layer for per-timestep prediction.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 16, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        outputs, _ = self.lstm(x)  # outputs: (batch, seq_len, hidden_size)
        out = self.fc(outputs)  # (batch, seq_len, 1)
        return out


def train_small_lstm(
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DummySequenceDataset(num_samples=2560, seq_len=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SmallLSTMRegressor(input_size=1, hidden_size=16, num_layers=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for x, y in dataloader:
            x = x.to(device)  # (batch, seq_len, 1)
            y = y.to(device)  # (batch, seq_len, 1)

            optimizer.zero_grad()
            preds = model(x)  # (batch, seq_len, 1)
            loss = criterion(preds, y)  # scalar, averaged over all positions
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x.size(0)

        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} - loss: {avg_loss:.4f}")

    return model


if __name__ == "__main__":
    # This is a small, self-contained example and is
    # intentionally unrelated to the rest of the project.
    train_small_lstm()

