"""Phase 3, LSTM step: architecture + training loop.

Every design choice here is explained in its own comment rather than
collected at the top, so the reasoning stays next to the code it justifies
(see scripts/train_lstm.py's module docstring for the run-level decisions:
which split, which epochs budget, what gets saved).

Target formulation (decided Aug 23 2026, same as the confirmed XGBoost
winner): predict `gap_final = final_quali_time - practice_reference`, never
`final_quali_time` directly - the baseline's real-data run proved the
direct formulation catastrophic (656,522% val MAPE, the circuit-baseline
problem predicted at design time). Reconstruct to absolute time
(`pred_gap + practice_reference`) before ever computing MAPE/R2.

This REPLACES Task_List.txt's original "three masked output heads
(Q1/Q2/Q3)" line, written before the Aug 23 2026 target simplification -
that design point is stale. One scalar output per driver-weekend, exactly
like the XGBoost baseline, for the same reason: every driver who took part
in qualifying has exactly one final_quali_time, so no per-segment masking
is needed at all, for either model.

`leave_one_round_out_cv_lstm` is a separate diagnostic, added after the
LSTM's first real run didn't beat the XGBoost baseline on the standard
80/20 split - see its own docstring for why a single split's "doesn't beat
baseline" verdict deserved the same skepticism the XGBoost LORO check
already applied to a misleadingly-good era-1 number on that same kind of
split. CONFIRMED on real data (Aug 24 2026) to beat the baseline
(1.053%/R²=0.989 pooled vs XGBoost's 1.153%/0.980) - the LSTM is the
leading model candidate.

`train_final_model` is the last step before deployment: once
cross-validation (leave_one_round_out_cv_lstm) has answered "does this
architecture generalize" and "roughly how many epochs does it need",
this function trains the actual artifact that ships, on 100% of the
available non-holdout data, for a fixed epoch count rather than repeating
early stopping against yet another carved-out val slice. See its own
docstring for the full reasoning.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset

from f1qp.modeling.baseline import mape, r_squared
from f1qp.modeling.sequences import FeatureImputer, FeatureScaler, build_lstm_sequences

RANDOM_SEED = 42


class QualifyingLSTM(nn.Module):
    """One LSTM over up to 3 practice-session steps, plus static context,
    predicting the practice-to-qualifying gap.

    Architecture decisions:

    - `hidden_size=64`, `num_layers=1`: 64 hidden units is the Task_List.txt
      spec. Single layer, not stacked: the training pool is ~1,300 driver-
      weekend sequences (see phase3-planning.md) - a second recurrent layer
      roughly doubles the parameter count for a sequence that's at most 3
      steps long, where a single layer already has enough capacity to
      combine 3 timesteps. Stacking is an easy later experiment if the
      single-layer model underfits, but starting there for a dataset this
      small is the standard "least complexity that could work" call.
    - `pack_padded_sequence` before the LSTM, not a raw zero-filled tensor:
      this makes the sprint weekend's absent 3rd session literally never
      seen by the recurrence (the LSTM's internal state simply isn't
      updated for a step beyond a sequence's real length) rather than fed
      in as a learned-to-be-ignored zero vector. `h_n` (the final hidden
      state PyTorch returns from a packed sequence) is therefore already
      "the hidden state after the last REAL session", exactly what's wanted
      - no separate gather-by-length step needed.
    - Static context (`is_sprint`, `era`) is concatenated onto `h_n` AFTER
      the recurrence, not fed in as extra per-timestep features: both are
      properties of the whole weekend, not of any one practice session, so
      repeating them at every timestep would just be redundant signal the
      LSTM has to learn to ignore. `era` here follows
      docs/feature_engineering.md's original Phase 2 design intent (listed
      as global LSTM context, "Phase 3 needs season-stratified validation
      to trust an aggregate MAPE across both eras") - note this makes the
      LSTM's inputs NOT identical to the XGBoost baseline's (era was never
      one of `ALL_FEATURE_COLS`, so the baseline never saw it). That's a
      deliberate asymmetry, not an oversight: era is known before the
      weekend starts (not label leakage), and Task_List.txt's own
      instruction is "only counts as a win if it beats the baseline" - the
      LSTM is allowed to use every legitimate signal available to it, and
      if era genuinely helps that's a fair win, not free credit.
    - Dropout `0.3-0.5` per Task_List.txt: applied after the LSTM (on
      `h_n`) and again inside the small feed-forward head, not as recurrent
      dropout inside the LSTM cell itself (PyTorch's built-in LSTM only
      supports inter-layer recurrent dropout, which does nothing with
      `num_layers=1`). Defaulted to 0.4, the middle of the specified range;
      exposed as a constructor argument so it's a one-line change to try
      0.3 or 0.5 if the learning curves (saved by the training script) show
      under- or over-fitting.
    - The feed-forward head is `Linear(64+2 -> 32) -> ReLU -> Dropout ->
      Linear(32 -> 1)`, not a direct `Linear(64+2 -> 1)`: one small hidden
      layer lets the model combine the sequence summary with the static
      context nonlinearly (e.g. "is_sprint changes what a given hidden
      state means") before collapsing to a scalar, at negligible extra
      parameter cost (~2.1k params) for a dataset this size.
    """

    def __init__(
        self,
        n_features: int,
        n_static: int = 2,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.post_lstm_dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size + n_static, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, X: torch.Tensor, lengths: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(X, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_last = h_n[-1]  # (batch, hidden_size) - top layer's final hidden state
        h_last = self.post_lstm_dropout(h_last)
        combined = torch.cat([h_last, static], dim=1)
        return self.head(combined).squeeze(-1)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_mape: float
    val_r2: float
    val_mape_by_era: Dict = field(default_factory=dict)
    val_r2_by_era: Dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0


@dataclass
class TrainResult:
    model: QualifyingLSTM
    best_epoch: int
    best_val_mape: float
    history: List[EpochMetrics]
    total_seconds: float


def _reconstruct_abs(pred_gap: np.ndarray, practice_reference: np.ndarray) -> np.ndarray:
    return pred_gap + practice_reference


def train_lstm(
    train_X, train_lengths, train_static, train_y_gap,
    val_X, val_lengths, val_static, val_y_abs, val_practice_reference, val_era,
    n_features: int,
    max_epochs: int = 300,
    patience: int = 25,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    huber_delta: float = 1.0,
    grad_clip_norm: float = 5.0,
    dropout: float = 0.4,
    seed: int = RANDOM_SEED,
    device: Optional[str] = None,
    verbose: bool = True,
) -> TrainResult:
    """Train with early stopping on validation MAPE (reconstructed absolute
    time), not on the training loss - the number that decides whether this
    beats the 1.2% target is MAPE on real lap times, and it's what the
    XGBoost baseline was judged on, so it's the only fair stopping/selection
    criterion for an apples-to-apples comparison.

    Decisions on the numbers below:
    - `max_epochs=300`, `patience=25`: the training pool is ~1,300
      sequences (see phase3-planning.md) feeding a ~25k-parameter model -
      small enough that overfitting, not underfitting, is the real risk
      (Task_List.txt's own small-sample flag). 25 epochs of no improvement
      is generous enough not to stop on ordinary noisy fluctuation but
      short enough to not waste time chasing an already-plateaued run.
    - `batch_size=64`: with ~1,300 train sequences this is ~20 batches per
      epoch - small enough to keep the stochasticity that regularizes a
      small dataset (full-batch gradient descent tends to memorize small
      data faster), large enough that each batch is a reasonably stable
      gradient estimate.
    - `nn.HuberLoss(delta=1.0)`: matches the XGBoost baseline's
      `reg:pseudohubererror(huber_slope=1.0)` choice and the reasoning
      behind it - the verified Sao Paulo 2024 / Las Vegas 2025 / Spa 2026
      rain weekends and the Saudi Arabia 2023 mechanical one-off dominate
      the gap target's tail; a squared-error loss would let a handful of
      real, legitimate weekends distort gradient updates for the ~95% of
      ordinary ones. Same delta value as the baseline's slope for a
      consistent notion of "how big an error counts as an outlier" across
      both models.
    - `Adam(lr=1e-3, weight_decay=1e-4)`: a standard, low-risk default for
      a model this small; the weight decay is extra L2 regularization on
      top of dropout, again aimed at the small-sample overfitting risk
      rather than at any observed problem.
    - Gradient clipping at global norm 5.0: cheap insurance against the
      occasional exploding-gradient batch (more of a known LSTM failure
      mode in general than something observed here) - clips rarely if the
      model is training normally, and costs nothing when it doesn't
      trigger.

    `verbose=True` (default) prints one line PER EPOCH as it happens - not
    buffered until the end - plus a startup line and an early-stopping
    notice, all with `flush=True` so the run's own progress is visible
    live in a terminal rather than appearing to hang and then dumping
    everything at once when training finishes. Set `verbose=False` for the
    test suite's short training runs, where the extra output is just
    noise.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = QualifyingLSTM(n_features=n_features, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=huber_delta)

    train_ds = TensorDataset(
        torch.as_tensor(train_X, dtype=torch.float32),
        torch.as_tensor(train_lengths, dtype=torch.int64),
        torch.as_tensor(train_static, dtype=torch.float32),
        torch.as_tensor(train_y_gap, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_X_t = torch.as_tensor(val_X, dtype=torch.float32, device=device)
    val_lengths_t = torch.as_tensor(val_lengths, dtype=torch.int64)
    val_static_t = torch.as_tensor(val_static, dtype=torch.float32, device=device)

    n_params = sum(p.numel() for p in model.parameters())
    n_batches_per_epoch = max(1, -(-len(train_ds) // batch_size))  # ceil
    if verbose:
        print(
            f"[train_lstm] device={device}  n_params={n_params:,}  "
            f"n_train={len(train_ds)}  n_val={len(val_X)}  "
            f"batches/epoch={n_batches_per_epoch}  max_epochs={max_epochs}  "
            f"patience={patience}",
            flush=True,
        )
        print(
            f"{'epoch':>6}  {'train_loss':>10}  {'val_MAPE%':>9}  {'val_R2':>7}  "
            f"{'sec':>6}  {'best_so_far':>11}",
            flush=True,
        )

    history: List[EpochMetrics] = []
    best_val_mape = float("inf")
    best_state = None
    best_epoch = -1
    epochs_without_improvement = 0
    start_time = time.monotonic()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        running_loss = 0.0
        n_batches = 0
        for X_batch, len_batch, static_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            static_batch = static_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch, len_batch, static_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_pred_gap = model(val_X_t, val_lengths_t, val_static_t).cpu().numpy()
        val_pred_abs = _reconstruct_abs(val_pred_gap, val_practice_reference)
        val_mape_score = mape(val_y_abs, val_pred_abs)
        val_r2_score = r_squared(val_y_abs, val_pred_abs)

        val_mape_by_era, val_r2_by_era = {}, {}
        for era_value in np.unique(val_era):
            idx = val_era == era_value
            val_mape_by_era[era_value] = mape(val_y_abs[idx], val_pred_abs[idx])
            val_r2_by_era[era_value] = r_squared(val_y_abs[idx], val_pred_abs[idx])

        elapsed = time.monotonic() - epoch_start
        history.append(EpochMetrics(
            epoch=epoch,
            train_loss=running_loss / max(n_batches, 1),
            val_mape=val_mape_score,
            val_r2=val_r2_score,
            val_mape_by_era=val_mape_by_era,
            val_r2_by_era=val_r2_by_era,
            elapsed_seconds=elapsed,
        ))

        improved = val_mape_score < best_val_mape
        if improved:
            best_val_mape = val_mape_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose:
            # Printed immediately, one line per epoch, not batched until the
            # loop ends - flush=True so it shows up live even if stdout is
            # being redirected/piped rather than a raw terminal.
            marker = f"* ({best_val_mape:.3f}%)" if improved else ""
            print(
                f"{epoch:6d}  {history[-1].train_loss:10.4f}  {val_mape_score:9.3f}  "
                f"{val_r2_score:7.3f}  {elapsed:6.2f}  {marker:>11}",
                flush=True,
            )

        if epochs_without_improvement >= patience:
            if verbose:
                print(
                    f"[train_lstm] Stopping early at epoch {epoch}: no val MAPE "
                    f"improvement for {patience} consecutive epochs "
                    f"(best was epoch {best_epoch}, {best_val_mape:.3f}%).",
                    flush=True,
                )
            break
    else:
        if verbose:
            print(
                f"[train_lstm] Reached max_epochs={max_epochs} without triggering "
                f"early stopping (best was epoch {best_epoch}, {best_val_mape:.3f}%) - "
                f"consider raising max_epochs if val MAPE was still improving.",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    total_seconds = time.monotonic() - start_time
    if verbose:
        print(
            f"[train_lstm] Done. Best epoch {best_epoch}, val MAPE {best_val_mape:.3f}%, "
            f"total wall-clock {total_seconds:.1f}s ({total_seconds / 60:.1f} min).",
            flush=True,
        )

    return TrainResult(
        model=model,
        best_epoch=best_epoch,
        best_val_mape=best_val_mape,
        history=history,
        total_seconds=total_seconds,
    )


def leave_one_round_out_cv_lstm(
    wide_df,
    feature_cols: List[str],
    fcols,
    era_value=1,
    train_kwargs: Optional[dict] = None,
    verbose: bool = True,
) -> dict:
    """LSTM analogue of `f1qp.modeling.baseline.leave_one_round_out_cv`.

    Motivation (Aug 23 2026): the LSTM's first real run didn't beat the
    XGBoost baseline (1.445% vs 1.090%) on the standard 80/20 split - but
    that XGBoost baseline's own 80/20 split was ALREADY shown to have a
    small-sample era-1 artifact (see the LORO entry in phase3-planning.md),
    so a single LSTM run on the same kind of split deserves the same
    skepticism before "doesn't beat the baseline" gets treated as settled.
    Every 2026 round is held out and tested exactly once, trained on
    everything else, mirroring the real Round 13 deployment far better
    than one random split - and, per the observed ~1 second per full
    training run, cheap enough to just do rather than trust one split.

    Unlike the XGBoost LORO (which trains directly on all non-test data,
    since XGBoost's `n_estimators=300` needs no separate tuning set), each
    LSTM fold still needs its OWN internal train/val split for early
    stopping - a network can't know when to stop without seeing held-out
    loss during training. Reuses the wide_df's EXISTING 'train'/'val'
    split column for that internal split (after removing the fold's test
    round from both), rather than inventing a fresh one per fold - this
    mirrors how the model would really be tuned in production: train +
    early-stop against the usual val set, then check generalization to a
    genuinely new round.

    Feature imputation and scaling are refit PER FOLD, on that fold's own
    train rows only - reusing the single global scaler/imputer across
    folds would leak each held-out round's distribution into statistics
    the model was implicitly tuned against, the same no-leakage rule
    already applied everywhere else in this project.

    `wide_df` should already be filtered to `has_target` and have the
    holdout (Zandvoort) split excluded - same precondition as
    `build_lstm_sequences` and `scripts/train_lstm.py`'s own usage.

    The returned dict also includes `residuals_by_round`: {round_number:
    array of SIGNED residuals (pred_abs - actual_abs) on reconstructed
    absolute time}, one entry per round, each computed by a model that
    never trained on that round. Added (Aug 24 2026) for
    f1qp.modeling.conformal's split-conformal calibration step - these are
    genuinely out-of-fold residuals, exactly what a conformal calibration
    pool needs, and were already being computed here for MAPE/R2 anyway.
    Doesn't change any existing key or behavior.
    """
    train_kwargs = dict(train_kwargs or {})
    train_kwargs.setdefault("verbose", False)  # 11 folds x per-epoch spam would drown the per-fold summary

    batch = build_lstm_sequences(wide_df, feature_cols, fcols)
    round_arr = batch.ids[fcols.round_number].to_numpy()
    split_arr = batch.split

    era_rounds = sorted(np.unique(round_arr[batch.era == era_value]))
    if not era_rounds:
        raise ValueError(f"No rounds found for era {era_value!r} in this dataset")

    per_round = {}
    residuals_by_round = {}
    all_true, all_pred = [], []
    fold_start_all = time.monotonic()

    for round_number in era_rounds:
        fold_start = time.monotonic()
        is_test_round = (batch.era == era_value) & (round_arr == round_number)
        remaining = ~is_test_round
        train_idx = remaining & (split_arr == "train")
        val_idx = remaining & (split_arr == "val")

        imputer = FeatureImputer.fit(batch.X[train_idx], batch.mask[train_idx])
        X_imputed = imputer.transform(batch.X, batch.mask)
        scaler = FeatureScaler.fit(X_imputed[train_idx], batch.mask[train_idx])
        X_scaled = scaler.transform(X_imputed, batch.mask)

        result = train_lstm(
            train_X=X_scaled[train_idx],
            train_lengths=batch.lengths[train_idx],
            train_static=batch.static[train_idx],
            train_y_gap=batch.y_gap[train_idx],
            val_X=X_scaled[val_idx],
            val_lengths=batch.lengths[val_idx],
            val_static=batch.static[val_idx],
            val_y_abs=batch.y_abs[val_idx],
            val_practice_reference=batch.practice_reference[val_idx],
            val_era=batch.era[val_idx],
            n_features=len(feature_cols),
            **train_kwargs,
        )

        model = result.model
        model.eval()
        with torch.no_grad():
            test_pred_gap = model(
                torch.as_tensor(X_scaled[is_test_round], dtype=torch.float32),
                torch.as_tensor(batch.lengths[is_test_round], dtype=torch.int64),
                torch.as_tensor(batch.static[is_test_round], dtype=torch.float32),
            ).numpy()
        test_pred_abs = test_pred_gap + batch.practice_reference[is_test_round]
        test_y_true = batch.y_abs[is_test_round]

        round_mape = mape(test_y_true, test_pred_abs)
        round_r2 = r_squared(test_y_true, test_pred_abs)
        per_round[round_number] = {
            "mape": round_mape,
            "r2": round_r2,
            "n_test": int(is_test_round.sum()),
            "best_epoch": result.best_epoch,
            "internal_val_mape": result.best_val_mape,
        }
        all_true.append(test_y_true)
        all_pred.append(test_pred_abs)
        residuals_by_round[round_number] = test_pred_abs - test_y_true

        if verbose:
            print(
                f"[loro_lstm] round={round_number:>3}  n_test={int(is_test_round.sum()):3d}  "
                f"MAPE={round_mape:6.3f}%  R2={round_r2:6.3f}  "
                f"(internal best_epoch={result.best_epoch}, "
                f"internal val_MAPE={result.best_val_mape:.3f}%)  "
                f"fold_seconds={time.monotonic() - fold_start:.2f}",
                flush=True,
            )

    all_true_arr = np.concatenate(all_true)
    all_pred_arr = np.concatenate(all_pred)
    pooled = {
        "mape": mape(all_true_arr, all_pred_arr),
        "r2": r_squared(all_true_arr, all_pred_arr),
        "n_test": len(all_true_arr),
    }
    if verbose:
        print(
            f"[loro_lstm] Done. {len(era_rounds)} folds, "
            f"total {time.monotonic() - fold_start_all:.1f}s. "
            f"Pooled MAPE={pooled['mape']:.3f}%  R2={pooled['r2']:.3f}",
            flush=True,
        )

    return {
        "era_value": era_value,
        "per_round": per_round,
        "pooled": pooled,
        "residuals_by_round": residuals_by_round,
    }


@dataclass
class FinalTrainResult:
    model: QualifyingLSTM
    train_loss_history: List[float]
    n_epochs: int
    total_seconds: float


def train_final_model(
    train_X,
    train_lengths,
    train_static,
    train_y_gap,
    n_features: int,
    n_epochs: int,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    huber_delta: float = 1.0,
    grad_clip_norm: float = 5.0,
    dropout: float = 0.4,
    seed: int = RANDOM_SEED,
    device: Optional[str] = None,
    verbose: bool = True,
) -> FinalTrainResult:
    """Train the FINAL production model on ALL available (non-holdout)
    data, for a FIXED number of epochs - deliberately no validation set
    and no early stopping.

    Why fixed epochs instead of early stopping: early stopping needs a
    held-out validation set to know when to stop, but the entire point of
    this function is to use every available non-holdout driver-weekend as
    training signal (Task_List.txt: "Train on 2023-2025 + 2026 R1-R11") -
    carving out a val slice here would shrink the final model's training
    data purely to answer a stopping-point question the cross-validation
    step has already answered. `n_epochs` should come from the
    already-confirmed `leave_one_round_out_cv_lstm` result, not be guessed
    fresh here - see scripts/train_final_lstm.py for the exact value used
    and the real numbers behind it.

    There are deliberately no `val_*` parameters at all (unlike
    `train_lstm`) - not merely unused, structurally absent - so "this
    specific model has no freshly-computed held-out generalization metric"
    is a fact enforced by the function signature, not just a comment
    someone could miss. The trustworthy generalization estimate for a
    model trained this way remains the leave-one-round-out pooled result
    computed earlier (see Task_List.txt / phase3-planning.md) - this
    function's job is only to produce the deployable artifact, not to
    re-measure what LORO already measured.

    Every other hyperparameter default matches `train_lstm`'s own defaults
    and reasoning (same architecture, same Huber loss/robustness rationale,
    same dropout/weight_decay/grad-clip choices) - this is the same model,
    just trained differently at the very end.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = QualifyingLSTM(n_features=n_features, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.HuberLoss(delta=huber_delta)

    train_ds = TensorDataset(
        torch.as_tensor(train_X, dtype=torch.float32),
        torch.as_tensor(train_lengths, dtype=torch.int64),
        torch.as_tensor(train_static, dtype=torch.float32),
        torch.as_tensor(train_y_gap, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    n_params = sum(p.numel() for p in model.parameters())
    n_batches_per_epoch = max(1, -(-len(train_ds) // batch_size))  # ceil
    if verbose:
        print(
            f"[train_final_model] device={device}  n_params={n_params:,}  "
            f"n_train={len(train_ds)} (ALL non-holdout rows - no val split)  "
            f"batches/epoch={n_batches_per_epoch}  n_epochs={n_epochs} (FIXED, "
            f"no early stopping)",
            flush=True,
        )
        print(f"{'epoch':>6}  {'train_loss':>10}  {'sec':>6}", flush=True)

    train_loss_history: List[float] = []
    start_time = time.monotonic()

    for epoch in range(1, n_epochs + 1):
        epoch_start = time.monotonic()
        model.train()
        running_loss = 0.0
        n_batches = 0
        for X_batch, len_batch, static_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            static_batch = static_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch, len_batch, static_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        epoch_loss = running_loss / max(n_batches, 1)
        train_loss_history.append(epoch_loss)
        elapsed = time.monotonic() - epoch_start

        if verbose:
            print(f"{epoch:6d}  {epoch_loss:10.4f}  {elapsed:6.2f}", flush=True)

    total_seconds = time.monotonic() - start_time
    if verbose:
        print(
            f"[train_final_model] Done. {n_epochs} epochs, "
            f"total {total_seconds:.1f}s ({total_seconds / 60:.1f} min). "
            f"No held-out metric for this run by design - trust the "
            f"leave-one-round-out pooled result for what to expect from "
            f"this model on unseen 2026 rounds.",
            flush=True,
        )

    return FinalTrainResult(
        model=model,
        train_loss_history=train_loss_history,
        n_epochs=n_epochs,
        total_seconds=total_seconds,
    )
