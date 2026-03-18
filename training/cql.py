import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0



CONT_FEAT_SIZE = 4
NUM_CAT_FIELDS = 13


def masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    x: [B, T, D]
    lengths: [B]
    returns final valid hidden state per sequence: [B, D]
    """
    idx = (lengths - 1).clamp(min=0)
    return x[torch.arange(x.size(0), device=x.device), idx]


def gumbel_softmax_sample(logits: torch.Tensor, tau: float = 1.0, hard: bool = True) -> torch.Tensor:
    """
    Straight-through Gumbel-Softmax sample.
    returns one-hot-like tensor [B, A]
    """
    y = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
    return y


class SequenceEncoder(nn.Module):
    """
    Trajectory encoder compatible with TrajLSTM in training/primary.py:
      - embeds all 13 categorical card fields with a shared embedding
      - concatenates with 4 continuous features
      - runs an LSTM over the sequence
      - returns the final hidden state (per sequence)
    """

    def __init__(
        self,
        state_dim: int,
        num_cards: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        emb_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_cards = num_cards
        self.hidden_dim = hidden_dim
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
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

    def _build_lstm_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 17]
        returns: [B, T, 4 + 13 * emb_dim]
        """
        cont = x[:, :, :CONT_FEAT_SIZE]
        cats = x[:, :, CONT_FEAT_SIZE:].long()
        cats = cats.clamp(min=0, max=self.num_cards - 1)
        emb = self.card_emb(cats)
        emb = emb.flatten(start_dim=2)
        return torch.cat([cont, emb], dim=-1)

    def forward(self, states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        states: [B, T, 17]
        lengths: [B]
        returns: [B, hidden_dim]
        """
        z = self._build_lstm_input(states)
        packed = nn.utils.rnn.pack_padded_sequence(
            z, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        return masked_mean(out, lengths)


class HybridPolicyLSTM(nn.Module):
    """
    Policy over:
      discrete action a_d in {0, ..., num_discrete-1}
      continuous params a_c in R^cont_dim, conditioned on a_d

    We output:
      logits for discrete
      per-discrete-action Gaussian params for continuous
    """
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        cont_dim: int,
        hidden_dim: int = 384,
        num_layers: int = 2,
        emb_dim: int = 16,
    ):
        super().__init__()
        self.num_discrete = num_discrete
        self.cont_dim = cont_dim

        self.encoder = SequenceEncoder(state_dim, num_discrete, hidden_dim, num_layers, emb_dim)

        self.logits_head = nn.Linear(hidden_dim, num_discrete)


        self.mu_head = nn.Linear(hidden_dim, num_discrete * cont_dim)
        self.log_std_head = nn.Linear(hidden_dim, num_discrete * cont_dim)

    def forward(self, states: torch.Tensor, lengths: torch.Tensor):
        """
        returns:
          logits: [B, A]
          mu_all: [B, A, C]
          log_std_all: [B, A, C]
        """
        h = self.encoder(states, lengths)

        logits = self.logits_head(h)

        mu_all = self.mu_head(h).view(-1, self.num_discrete, self.cont_dim)
        log_std_all = self.log_std_head(h).view(-1, self.num_discrete, self.cont_dim)
        log_std_all = torch.clamp(log_std_all, LOG_STD_MIN, LOG_STD_MAX)

        return logits, mu_all, log_std_all

    def _mask_logits_by_hand(
        self,
        states: torch.Tensor,
        lengths: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mask logits so only cards currently in hand are legal.
        Hand is taken from the last valid timestep of each sequence.
        """
        B, T, D = states.shape
        device = states.device


        idx = (lengths - 1).clamp(min=0)
        last = states[torch.arange(B, device=device), idx]


        hand_ids = last[:, CONT_FEAT_SIZE : CONT_FEAT_SIZE + 4].long()
        hand_ids = hand_ids.clamp(min=0, max=self.num_discrete - 1)


        valid = torch.zeros(B, self.num_discrete, device=device, dtype=torch.bool)
        valid.scatter_(1, hand_ids, True)

        valid[:, 0] = False


        masked_logits = logits.masked_fill(~valid, float("-inf"))
        all_invalid = ~valid.any(dim=1)
        if all_invalid.any():
            masked_logits[all_invalid] = logits[all_invalid]
        return masked_logits

    def sample(self, states: torch.Tensor, lengths: torch.Tensor, temperature: float = 1.0):
        """
        Samples hybrid action and computes log pi(a_d, a_c | s).

        returns:
          disc_onehot: [B, A]            differentiable straight-through sample
          disc_index: [B]               hard index
          cont_action: [B, C]           tanh-squashed continuous action
          logp: [B]
          probs: [B, A]
        """
        logits, mu_all, log_std_all = self.forward(states, lengths)

        logits = self._mask_logits_by_hand(states, lengths, logits)
        probs = F.softmax(logits, dim=-1)


        disc_onehot = gumbel_softmax_sample(logits, tau=temperature, hard=True)
        disc_index = disc_onehot.argmax(dim=-1)


        log_probs_disc_all = F.log_softmax(logits, dim=-1)
        logp_disc = (disc_onehot * log_probs_disc_all).sum(dim=-1)


        chooser = disc_onehot.unsqueeze(-1)
        mu = (mu_all * chooser).sum(dim=1)
        log_std = (log_std_all * chooser).sum(dim=1)
        std = log_std.exp()


        mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
        std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
        std = std.clamp_min(1e-3)

        dist = Normal(mu, std)
        z = dist.rsample()
        cont_action = torch.tanh(z)


        logp_cont = dist.log_prob(z).sum(dim=-1)
        logp_cont -= torch.log(1 - cont_action.pow(2) + 1e-6).sum(dim=-1)

        logp = logp_disc + logp_cont
        return disc_onehot, disc_index, cont_action, logp, probs


class HybridQLSTM(nn.Module):
    """
    Q(s, a_d, a_c)
    Input:
      sequence state history
      discrete action one-hot
      continuous action vector
    """
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        cont_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        emb_dim: int = 16,
    ):
        super().__init__()
        self.encoder = SequenceEncoder(state_dim, num_discrete, hidden_dim, num_layers, emb_dim)

        self.q_mlp = nn.Sequential(
            nn.Linear(hidden_dim + num_discrete + cont_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        states: torch.Tensor,
        lengths: torch.Tensor,
        disc_onehot: torch.Tensor,
        cont_action: torch.Tensor,
    ) -> torch.Tensor:
        """
        states: [B, T, state_dim]
        lengths: [B]
        disc_onehot: [B, A]
        cont_action: [B, C]
        returns: [B]
        """
        h = self.encoder(states, lengths)
        x = torch.cat([h, disc_onehot, cont_action], dim=-1)
        return self.q_mlp(x).squeeze(-1)


@dataclass
class SACBatch:
    states: torch.Tensor
    lengths: torch.Tensor
    disc_actions: torch.Tensor
    cont_actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    next_lengths: torch.Tensor
    dones: torch.Tensor


class HybridSACLSTM(nn.Module):
    def __init__(
        self,
        state_dim: int,
        num_discrete: int,
        cont_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 1,
        emb_dim: int = 16,
        alpha: float = 0.2,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        super().__init__()
        self.num_discrete = num_discrete
        self.cont_dim = cont_dim
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.policy = HybridPolicyLSTM(state_dim, num_discrete, cont_dim, hidden_dim, num_layers, emb_dim).to(device)

        self.q1 = HybridQLSTM(state_dim, num_discrete, cont_dim, hidden_dim, num_layers, emb_dim).to(device)
        self.q2 = HybridQLSTM(state_dim, num_discrete, cont_dim, hidden_dim, num_layers, emb_dim).to(device)

        self.q1_target = HybridQLSTM(state_dim, num_discrete, cont_dim, hidden_dim, num_layers, emb_dim).to(device)
        self.q2_target = HybridQLSTM(state_dim, num_discrete, cont_dim, hidden_dim, num_layers, emb_dim).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

    def soft_update_targets(self):
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)

    def make_onehot(self, disc_actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(disc_actions, num_classes=self.num_discrete).float()

    def critic_loss(self, batch: SACBatch):
        disc_onehot = self.make_onehot(batch.disc_actions)

        with torch.no_grad():
            next_disc_onehot, _, next_cont_action, next_logp, _ = self.policy.sample(
                batch.next_states, batch.next_lengths
            )

            q1_next = self.q1_target(batch.next_states, batch.next_lengths, next_disc_onehot, next_cont_action)
            q2_next = self.q2_target(batch.next_states, batch.next_lengths, next_disc_onehot, next_cont_action)
            q_next = torch.min(q1_next, q2_next)

            target = batch.rewards + self.gamma * (1.0 - batch.dones) * (q_next - self.alpha * next_logp)

        q1_pred = self.q1(batch.states, batch.lengths, disc_onehot, batch.cont_actions)
        q2_pred = self.q2(batch.states, batch.lengths, disc_onehot, batch.cont_actions)

        q1_loss = F.mse_loss(q1_pred, target)
        q2_loss = F.mse_loss(q2_pred, target)
        return q1_loss + q2_loss, {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "target_mean": target.mean().item(),
            "q1_mean": q1_pred.mean().item(),
            "q2_mean": q2_pred.mean().item(),
        }

    def policy_loss(self, batch: SACBatch, temperature: float = 1.0):
        disc_onehot, _, cont_action, logp, probs = self.policy.sample(
            batch.states, batch.lengths, temperature=temperature
        )

        q1_pi = self.q1(batch.states, batch.lengths, disc_onehot, cont_action)
        q2_pi = self.q2(batch.states, batch.lengths, disc_onehot, cont_action)
        q_pi = torch.min(q1_pi, q2_pi)

        loss = (self.alpha * logp - q_pi).mean()
        entropy_disc = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1).mean()

        return loss, {
            "policy_loss": loss.item(),
            "logp_mean": logp.mean().item(),
            "q_pi_mean": q_pi.mean().item(),
            "disc_entropy": entropy_disc.item(),
        }






def train_step(
    agent: HybridSACLSTM,
    batch: SACBatch,
    policy_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    temperature: float = 1.0,
):

    q_optimizer.zero_grad()
    q_loss, q_info = agent.critic_loss(batch)
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(agent.q1.parameters()) + list(agent.q2.parameters()), 1.0)
    q_optimizer.step()


    policy_optimizer.zero_grad()
    pi_loss, pi_info = agent.policy_loss(batch, temperature=temperature)
    pi_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), 1.0)
    policy_optimizer.step()

    agent.soft_update_targets()

    info = {}
    info.update(q_info)
    info.update(pi_info)
    return info






def _append_next_step(
    states: torch.Tensor,
    lengths: torch.Tensor,
    target_xy: torch.Tensor,
    target_card: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build a simple next_state by appending the target action as one more step.
    We don't have exact next hand/deck in the targets, so we copy the last valid step
    and overwrite (x,y,time,card) and set side=1.0.
    """
    device = states.device
    B, T, D = states.shape
    idx = (lengths - 1).clamp(min=0)
    last = states[torch.arange(B, device=device), idx].clone()


    last[:, 0:3] = target_xy.to(device, dtype=states.dtype)
    last[:, 3] = 1.0
    last[:, -1] = target_card.to(device, dtype=states.dtype)

    next_states = torch.cat([states, last.unsqueeze(1)], dim=1)
    next_lengths = lengths + 1
    return next_states, next_lengths


def traj_batch_to_sac(
    x: torch.Tensor,
    lengths: torch.Tensor,
    target_xy: torch.Tensor,
    target_card: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
) -> SACBatch:
    """
    Convert a batch from training/traj_dataloader.py into a SACBatch.
    - discrete action = target_card
    - continuous action = target_xy mapped to tanh range [-1,1]
    - reward/done from dataset
    - next_state = appended target step (approx)
    """
    states = x
    disc_actions = target_card.long()
    cont_actions = (target_xy.float() * 2.0 - 1.0).clamp(-1.0, 1.0)
    next_states, next_lengths = _append_next_step(states, lengths, target_xy, target_card)
    return SACBatch(
        states=states,
        lengths=lengths.long(),
        disc_actions=disc_actions,
        cont_actions=cont_actions,
        rewards=reward.float(),
        next_states=next_states,
        next_lengths=next_lengths.long(),
        dones=done.float(),
    )


def load_traj_card_head_into_policy(
    agent: HybridSACLSTM,
    ckpt_path: str | Path,
    strict: bool = False,
) -> None:
    """
    Hot-start HybridSACLSTM's policy from a TrajLSTM checkpoint trained in primary.py.
    Copies:
      - all LSTM parameters (encoder.lstm.*)
      - final card prediction linear layer into policy.logits_head (card logits)
    Does NOT touch xy/t heads from TrajLSTM.
    """
    ckpt_path = Path(ckpt_path)
    payload = torch.load(ckpt_path, map_location="cpu")
    state_dict = payload.get("model_state", payload.get("model", payload))


    enc_sd = agent.policy.encoder.state_dict()
    updated_lstm = 0
    for name, param in state_dict.items():
        if not name.startswith("lstm."):
            continue
        if name in enc_sd and enc_sd[name].shape == param.shape:
            enc_sd[name] = param
            updated_lstm += 1
    if updated_lstm == 0 and strict:
        raise ValueError(f"No matching LSTM parameters found to load from {ckpt_path}")
    agent.policy.encoder.load_state_dict(enc_sd, strict=False)




    w_key = "card_head.3.weight"
    b_key = "card_head.3.bias"
    if w_key not in state_dict or b_key not in state_dict:
        msg = f"{ckpt_path} does not contain expected keys '{w_key}'/'{b_key}'"
        if strict:
            raise ValueError(msg)
        print("Warning:", msg)
        return

    w = state_dict[w_key]
    b = state_dict[b_key]
    head = agent.policy.logits_head
    if head.weight.shape != w.shape or head.bias.shape != b.shape:
        msg = (
            f"Shape mismatch for card head: ckpt {w.shape},{b.shape} vs "
            f"policy {head.weight.shape},{head.bias.shape}"
        )
        if strict:
            raise ValueError(msg)
        print("Warning:", msg)
        return

    with torch.no_grad():
        head.weight.copy_(w)
        head.bias.copy_(b)

    print(
        f"Loaded {updated_lstm} LSTM parameter tensors and TrajLSTM card_head "
        f"into policy from {ckpt_path}"
    )


def train_epochs(
    agent: HybridSACLSTM,
    dataloader,
    policy_optimizer: torch.optim.Optimizer,
    q_optimizer: torch.optim.Optimizer,
    epochs: int = 5,
    temperature: float = 1.0,
    log_every: int = 100,
    writer: SummaryWriter | None = None,
    run_name: str | None = None,
):
    agent.train()
    step = 0
    run_name = run_name or "cql_run"
    for ep in range(1, epochs + 1):
        sums = {
            "q1_loss": 0.0,
            "q2_loss": 0.0,
            "policy_loss": 0.0,
            "target_mean": 0.0,
            "q1_mean": 0.0,
            "q2_mean": 0.0,
            "logp_mean": 0.0,
            "disc_entropy": 0.0,
        }
        n = 0

        pbar = tqdm(dataloader, desc=f"Epoch {ep}/{epochs}", leave=False)
        for x, lengths, target_xy, target_card, reward, done in pbar:
            x = x.to(agent.device, dtype=torch.float32)
            lengths = lengths.to(agent.device)
            target_xy = target_xy.to(agent.device, dtype=torch.float32)
            target_card = target_card.to(agent.device)
            reward = reward.to(agent.device)
            done = done.to(agent.device)

            batch = traj_batch_to_sac(x, lengths, target_xy, target_card, reward, done)
            info = train_step(agent, batch, policy_optimizer, q_optimizer, temperature=temperature)

            n += 1
            for k in sums:
                if k in info:
                    sums[k] += float(info[k])

            step += 1
            if writer is not None:
                writer.add_scalar("train/q1_loss", info.get("q1_loss", float("nan")), step)
                writer.add_scalar("train/q2_loss", info.get("q2_loss", float("nan")), step)
                writer.add_scalar("train/policy_loss", info.get("policy_loss", float("nan")), step)
                writer.add_scalar("train/disc_entropy", info.get("disc_entropy", float("nan")), step)
                writer.add_scalar("train/target_mean", info.get("target_mean", float("nan")), step)

            if log_every and (step % log_every == 0):
                pbar.set_postfix(
                    q1=f"{info.get('q1_loss', float('nan')):.4f}",
                    q2=f"{info.get('q2_loss', float('nan')):.4f}",
                    pi=f"{info.get('policy_loss', float('nan')):.4f}",
                    ent=f"{info.get('disc_entropy', float('nan')):.4f}",
                )

        if n == 0:
            print(f"Epoch {ep}/{epochs}: no batches")
            continue
        avg = {k: v / n for k, v in sums.items()}
        print(
            f"Epoch {ep}/{epochs} done  "
            f"q1={avg['q1_loss']:.4f} q2={avg['q2_loss']:.4f} pi={avg['policy_loss']:.4f} "
            f"entropy={avg['disc_entropy']:.4f} target_mean={avg['target_mean']:.4f}"
        )
        if writer is not None:
            writer.add_scalar("epoch/q1_loss", avg["q1_loss"], ep)
            writer.add_scalar("epoch/q2_loss", avg["q2_loss"], ep)
            writer.add_scalar("epoch/policy_loss", avg["policy_loss"], ep)
            writer.add_scalar("epoch/disc_entropy", avg["disc_entropy"], ep)
            writer.add_scalar("epoch/target_mean", avg["target_mean"], ep)






if __name__ == "__main__":
    import sys

    device = "cuda" if torch.cuda.is_available() else "cpu"












    argv = list(sys.argv[1:])
    init_ckpt = None
    if "--init_card_ckpt" in argv:
        idx = argv.index("--init_card_ckpt")
        if idx + 1 >= len(argv):
            raise ValueError("--init_card_ckpt given without path")
        init_ckpt = argv[idx + 1]

        del argv[idx : idx + 2]

    arg1 = argv[0] if len(argv) > 0 else ""

    from traj_dataloader import (
        DATA_DIR,
        DEFAULT_TRAJ_OKEZUE_PATH,
        DEFAULT_TRAJ_PATH,
        SavedTrajDataset,
        get_traj_dataloader,
        pad_collate,
    )
    from torch.utils.data import DataLoader

    if arg1 in ("pt", "saved"):
        mode = argv[1] if len(argv) > 1 else "both"
        pt_path = DATA_DIR / f"ds_{mode}.pt"
        payload = torch.load(pt_path, weights_only=False)
        ds = SavedTrajDataset(payload)
        dl = DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, collate_fn=pad_collate)
        num_discrete = int(getattr(ds, "num_cards", 0) or 0)
        if not num_discrete:
            raise ValueError(f"Saved dataset {pt_path} missing num_cards.")
        sample_x, *_ = next(iter(dl))
        state_dim = int(sample_x.size(-1))
    else:
        source = arg1
        if source == "okezue":
            csv_path = DEFAULT_TRAJ_OKEZUE_PATH
        elif source == "win":
            csv_path = DATA_DIR / "traj_win.csv"
        else:
            csv_path = (DATA_DIR / "traj_win.csv") if (DATA_DIR / "traj_win.csv").exists() else DEFAULT_TRAJ_PATH

        dl = get_traj_dataloader(
            csv_path=csv_path,
            batch_size=64,
            shuffle=True,
            num_workers=0,
            skip_ability=True,
            mode="planner",
        )

        sample_x, *_ = next(iter(dl))
        state_dim = int(sample_x.size(-1))
        num_discrete = int(getattr(dl.dataset, "get_num_cards", lambda: 0)() or getattr(dl.dataset, "num_cards", 0))
        if not num_discrete:
            raise ValueError("Could not infer num_discrete (vocab size) from dataset.")

    cont_dim = 3



    hidden_dim = 384
    num_layers = 2
    emb_dim = 16
    if init_ckpt is not None:
        try:
            payload_cfg = torch.load(init_ckpt, map_location="cpu")
            cfg = payload_cfg.get("config", {})
            ck_hidden = cfg.get("hidden_size")
            ck_layers = cfg.get("num_layers")
            ck_emb = cfg.get("emb_dim")
            if ck_hidden is not None:
                hidden_dim = int(ck_hidden)
            if ck_layers is not None:
                num_layers = int(ck_layers)
            if ck_emb is not None:
                emb_dim = int(ck_emb)
            print(
                f"Using hidden_dim={hidden_dim}, num_layers={num_layers}, emb_dim={emb_dim} "
                f"from TrajLSTM config"
            )
        except Exception as e:
            print(f"Could not read config from {init_ckpt}: {e}; using default hidden_dim={hidden_dim}, num_layers={num_layers}")

    agent = HybridSACLSTM(
        state_dim=state_dim,
        num_discrete=num_discrete,
        cont_dim=cont_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        emb_dim=emb_dim,
        alpha=0.2,
        gamma=0.99,
        tau=0.005,
        device=device,
    )


    if init_ckpt is not None:
        print(f"Hot-starting policy card head from {init_ckpt}")
        load_traj_card_head_into_policy(agent, init_ckpt, strict=False)

    policy_optimizer = torch.optim.Adam(agent.policy.parameters(), lr=3e-4)
    q_optimizer = torch.optim.Adam(list(agent.q1.parameters()) + list(agent.q2.parameters()), lr=3e-4)


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"cql_{'pt' if arg1 in ('pt', 'saved') else 'csv'}_{timestamp}"
    runs_root = Path(__file__).resolve().parent / "runs"
    writer = SummaryWriter(log_dir=str(runs_root / run_name))


    epochs = 5
    train_epochs(
        agent,
        dl,
        policy_optimizer,
        q_optimizer,
        epochs=epochs,
        temperature=1.0,
        log_every=100,
        writer=writer,
        run_name=run_name,
    )
    writer.close()
