"""Steps 5-6 reconstruction detectors: shared machinery plus the Step 5 autoencoder.

The reconstruction corner of the five-method comparison. A model is trained
to rebuild ordinary windows of a sensor's own history; whatever it rebuilds
badly is scored as anomalous. Step 5 does this with a plain feed-forward
network over a flattened window and Step 6 will do it with recurrent layers
over the same window as a sequence, so everything that is not the
architecture itself -- windowing, the time-based split, scaling, the
training loop, the error-to-score step, the threshold rule -- lives here and
is shared, and the two steps differ only where they are supposed to differ.

Four decisions carry in from earlier steps rather than being made here:

*The window length is a controlled factor, not a free parameter.* Step 4b
re-ran the Step 4 density methods at a 6h window instead of their 1h one and
changed nothing else; each method's agreement with *itself* fell to 0.053
(LOF) and 0.242 (Isolation Forest) -- for LOF, lower than its agreement with
an entirely different method family. Window length therefore moves a flagged
set as much as method family does, which is why `SEQUENCE_BINS` is shared by
Steps 5 and 6 rather than chosen per step: at mismatched windows, comparing
them would measure the window and not the recurrence that comparison exists
to isolate.

*Raw flattened windows, not summary features.* Step 6's recurrent models must
see the actual sequence, so Step 5 is given the identical numbers in
flattened form. Had Step 5 instead reused Step 4's 4-dimensional window
summaries, the Step 5 vs Step 6 contrast would confound representation with
recurrence -- the same class of error Step 4b caught for window length.

*Per station x sensor.* Step 1 found global lag-0 correlation between the
three sensors to be weak everywhere (|r| < 0.15), and Steps 3-4 both fit per
station x sensor. Keeping that scope means a Step 7 disagreement reflects the
method rather than a change in what was fit to what, and "anomalous" keeps
meaning anomalous *for that station* rather than relative to a network-wide
pool.

*The threshold is a percentile, and that is not equivalent to what Steps 3-4
use.* `|z| > 3`, `LOF > 1.5` and `s(x, n) > 0.5` are published conventions
that let the data decide how much gets flagged. A reconstruction error has no
published cutoff, so a percentile of the validation error distribution is the
honest alternative -- but it fixes the alarm rate by construction rather than
discovering it. This method's native flagged rate is therefore not comparable
to the others' in the way theirs are to each other, which is a fact for Step 7
to report rather than to work around.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch import nn

# 6 hours at 10-minute resolution, matching Step 3's rolling window so the
# project's method roster covers two time scales (Step 4's 1h and this) rather
# than three. Shared with Step 6 -- see the module docstring.
SEQUENCE_BINS = 36

# Second window for this step's self-agreement control, the same device Step 4b
# applied to the density methods. 1h matches Step 4, so the control doubles as a
# window-matched comparison against LOF/Isolation Forest.
CONTROL_BINS = 6

# Compression ratio held fixed across window lengths. The control at a different
# window would otherwise vary two things at once: at 6 inputs a latent of 8 is
# over-complete, and an autoencoder that can copy its input reconstructs
# everything perfectly and scores nothing as anomalous. Holding the ratio keeps
# the window the only thing that changes.
COMPRESSION = 4.5
MIN_BOTTLENECK = 2

# 4.5:1 at the primary window, and the middle of the grid below.
BOTTLENECK = 8

# Four bottleneck sizes and five seeds, the shape of the prior prototype's
# ablation, rescaled to this input size. That prototype's one transferable
# ablation finding was that a too-large bottleneck went bimodal and
# seed-sensitive where training data was scarce, so a single fixed seed would
# hide exactly the instability worth knowing about here.
BOTTLENECK_GRID = (2, 4, 8, 16)
SEEDS = (0, 1, 2, 3, 4)

# Earliest 80% trains, latest 20% validates. Never a random split: consecutive
# 10-minute readings are strongly dependent, so random assignment would put a
# window's near-duplicate neighbours on the other side of the split and the
# validation error would understate reconstruction difficulty.
TRAIN_FRACTION = 0.8

# Percentile of the validation-split reconstruction error used as the flagging
# cutoff, with the two looser values reported alongside it as a sensitivity
# analysis rather than presenting one as validated.
THRESHOLD_PCT = 0.99
THRESHOLD_GRID = (0.90, 0.95, 0.99)

MAX_EPOCHS = 500
BATCH_SIZE = 256

# Rows per forward pass when scoring or evaluating, where no gradient is kept
# and the batch is therefore a pure memory/throughput knob rather than part of
# the method. It has to exist: a recurrent model holds an activation per bin
# per window, so scoring a large cell in one pass asks for several GB and
# fails outright on a 4GB accelerator. Every row's error is computed from its
# own window alone, so chunking changes throughput and nothing else.
EVAL_BATCH_SIZE = 8192
LEARNING_RATE = 1e-3
PATIENCE = 8

# Relative improvement in held-out loss that counts as progress. Patience alone
# is not enough: a run resets its counter on any improvement at all, so on a
# genuine plateau the noise-level gains keep resetting it and training only ends
# when the epoch budget runs out -- which is a cap, not convergence. Requiring
# the gain to clear a floor makes early stopping fire when the model has
# actually stopped learning. It gates the patience counter only; the best state
# is still tracked on any improvement, so nothing is lost by requiring it.
MIN_DELTA = 1e-4


def bottleneck_for(window: int, compression: float = COMPRESSION) -> int:
    """Latent size holding `compression`:1 against a window of `window` inputs."""
    return max(MIN_BOTTLENECK, round(window / compression))


def sliding_windows(series: pd.Series, window: int = SEQUENCE_BINS) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Every complete `window`-length window of `series`, labelled at its center.

    Stride 1, so each reading whose surrounding window is complete gets its own
    window, and a window containing any NaN -- a gap the <=6h fill left open, or
    the edge of the record -- is dropped rather than imputed. That is the same
    completeness rule `uema.density.window_features` applies, so the three
    method families end up scoring the same set of timestamps.

    The center label follows pandas' `rolling(center=True)` convention (offset
    `window // 2` from the window's start) rather than the true midpoint, which
    for an even window falls between two bins. Matching pandas is what matters:
    Steps 3 and 4 both label centered windows that way, and Step 7 aligns the
    methods on the timestamp.

    Assumes `series` is on a strict 10-minute grid (`uema.silver`), since the
    window length is counted in bins.
    """
    values = series.to_numpy(dtype=float)
    if len(values) < window:
        return np.empty((0, window)), series.index[:0]

    view = np.lib.stride_tricks.sliding_window_view(values, window)
    complete = ~np.isnan(view).any(axis=1)
    offset = window // 2
    centers = series.index[offset : offset + len(view)]
    return view[complete], centers[complete]


def time_split(index: pd.DatetimeIndex, fraction: float = TRAIN_FRACTION) -> np.ndarray:
    """Boolean mask selecting the earliest `fraction` of `index` for training.

    Split on time rather than on position so the boundary is a calendar
    instant: windows are dropped unevenly across the record (see
    `sliding_windows`), and a positional split would put the boundary at a
    different date for every station x sensor.
    """
    if len(index) == 0:
        return np.zeros(0, dtype=bool)
    cutoff = index[0] + (index[-1] - index[0]) * fraction
    return np.asarray(index <= cutoff)


def scale_windows(matrix: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Put `matrix` on a common scale, fit on training rows only.

    `RobustScaler` for the same reason Step 4 uses it: it centers on the median
    and scales by the IQR, so the extreme readings the detector exists to find
    do not inflate the scale they are measured against, and it degrades safely
    on `precipitation`, whose IQR is exactly zero.

    Fitting on the training rows alone keeps the validation split genuinely
    unseen -- the percentile threshold is read off that split, so letting it
    influence the scaling would calibrate the cutoff partly against itself.
    Every bin of the window is one column and they are all the same sensor in
    the same units, so the scaler is fit on the pooled values rather than
    per-column, which would otherwise give each lag its own centering.
    """
    scaler = RobustScaler()
    scaler.fit(matrix[train_mask].reshape(-1, 1))
    return scaler.transform(matrix.reshape(-1, 1)).reshape(matrix.shape)


class DenseAutoencoder(nn.Module):
    """Step 5's non-recurrent autoencoder: window -> hidden -> latent -> hidden -> window.

    Deliberately contains no recurrent layer of any kind. Its role in the study
    is to show what nonlinear reconstruction achieves on its own, so that
    Step 7 can read any difference against Step 6's LSTM/GRU variants as the
    contribution of modelling temporal order explicitly, rather than as the
    contribution of reconstruction-based scoring in general. The input is a
    flattened window, so the network has no way to know its inputs are ordered
    in time -- which is exactly the property being withheld.

    The hidden layer sits at the geometric mean of the input and latent sizes,
    giving an even funnel at any window length instead of a width that only
    happens to look reasonable at one.
    """

    def __init__(self, window: int, bottleneck: int):
        super().__init__()
        hidden = max(bottleneck + 1, round((window * bottleneck) ** 0.5))
        self.encoder = nn.Sequential(
            nn.Linear(window, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, window),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_reconstructor(
    model: nn.Module,
    matrix: np.ndarray,
    train_mask: np.ndarray,
    seed: int = 0,
    max_epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    patience: int = PATIENCE,
    min_delta: float = MIN_DELTA,
    device: str = "cpu",
) -> dict[str, list[float]]:
    """Fit `model` to reconstruct the training rows of `matrix`; return loss history.

    Shared by Steps 5 and 6 -- the recurrent models differ in what they do with
    a window, not in how they are trained, and holding the optimizer, batch
    size, stopping rule and seed handling fixed is what lets Step 7 attribute a
    difference between them to the architecture.

    Early stopping restores the best validation state rather than the last one,
    so the scores come from the model at its best generalizing point instead of
    from however many epochs it took to run out of patience. Patience is counted
    from the last improvement that cleared `min_delta` rather than from the last
    improvement of any size -- see that constant. `max_epochs` is meant to be a
    safety bound that early stopping reaches first; if a run ends by exhausting
    it, the model was still learning when it was cut off, which is a training
    defect rather than a converged fit.

    `seed` is set explicitly because both the weight initialization and the
    batch order are random: without it, re-running would produce a slightly
    different flagged set and Step 7's agreement numbers would not be
    reproducible. It is also varied deliberately -- see `SEEDS` -- since one
    seed reports a number while several report whether that number is stable.

    The model's parameters are re-initialized here, after seeding, rather than
    being taken as the caller left them. Seeding only the batch order looks
    like it seeds the run and does not: a model is constructed before this
    function is called, so its weights come from wherever the ambient RNG
    happened to be, which depends on how many models were built earlier in the
    same process. The symptom is narrow enough to miss entirely -- every fit
    after the first in a process reproduces exactly, and only the first one
    differs -- and the effect is not small: on a `precipitation` cell the
    unseeded initializations ended training at 145, 154, 176 and 205 epochs on
    different runs, and stopping one epoch apart moved that cell's flagged set
    by ~20% (Jaccard 0.80). Resetting here makes a fit depend on `seed` alone,
    so a cell reproduces on its own rather than only as part of an identical
    whole-notebook run.

    `device` moves the model and both splits onto an accelerator; the whole
    cell is a few megabytes, so it stays resident for the entire fit and no
    batch is ever transferred. It exists for Step 6: a recurrent layer walks
    the window one bin at a time and is far slower on CPU than the dense
    model, where the reverse holds and dispatch overhead dominates. The
    default stays CPU so Step 5's published fits are reproduced by the same
    call that produced them. Reproducibility under `seed` is a property of the
    device, not of this function -- verify it on whichever one is used rather
    than assuming it.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    train = torch.from_numpy(matrix[train_mask]).float().to(device)
    val = torch.from_numpy(matrix[~train_mask]).float().to(device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)

    history: dict[str, list[float]] = {"train": [], "val": []}
    best_loss = np.inf
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    last_gain = 0

    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        epoch_loss = 0.0
        for start in range(0, len(train), batch_size):
            batch = train[order[start : start + batch_size]]
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        history["train"].append(epoch_loss / max(len(train), 1))

        model.eval()
        val_loss = _evaluate(model, val, loss_fn) if len(val) else np.nan
        history["val"].append(val_loss)

        if val_loss < best_loss:
            if val_loss < best_loss * (1 - min_delta):
                last_gain = epoch
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch - last_gain >= patience:
            break

    model.load_state_dict(best_state)
    return history


def _evaluate(model: nn.Module, data: torch.Tensor, loss_fn: nn.Module) -> float:
    """Mean loss over `data`, chunked only when it does not fit in one pass.

    The single-pass branch is not an optimization, it is exactness: recombining
    chunk means as a weighted average is a multiply and a divide, which can
    land one ulp away from the same quantity computed in one pass. That
    sounds ignorable and is not -- the stopping rule compares this number
    against `best_loss * (1 - min_delta)`, so a last-bit difference can move
    the stopping epoch by one, and one epoch was measured to move a cell's
    flagged set by ~20% (Jaccard 0.80 against the same fit stopped an epoch
    later). Cells small enough to evaluate in one pass therefore keep the
    exact arithmetic Step 5's stored scores were produced with.
    """
    with torch.no_grad():
        if len(data) <= EVAL_BATCH_SIZE:
            return loss_fn(model(data), data).item()
        total = 0.0
        for start in range(0, len(data), EVAL_BATCH_SIZE):
            chunk = data[start : start + EVAL_BATCH_SIZE]
            total += loss_fn(model(chunk), chunk).item() * len(chunk)
    return total / len(data)


def reconstruction_error(
    model: nn.Module, matrix: np.ndarray, index: pd.DatetimeIndex, device: str = "cpu"
) -> pd.Series:
    """Per-window mean squared reconstruction error, indexed at the window centers.

    This is the `score` column of the shared schema: higher means the model
    rebuilt this window less well from what it learned about ordinary ones.
    Averaged over the window's bins so the value does not depend on the window
    length, which the self-agreement control varies.
    """
    if len(matrix) == 0:
        return pd.Series(dtype=float, index=index[:0])

    model = model.to(device)
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(matrix), EVAL_BATCH_SIZE):
            tensor = torch.from_numpy(matrix[start : start + EVAL_BATCH_SIZE]).float().to(device)
            chunks.append(((model(tensor) - tensor) ** 2).mean(dim=1).cpu().numpy())
    return pd.Series(np.concatenate(chunks), index=index)


def percentile_threshold(
    errors: pd.Series, train_mask: np.ndarray, pct: float = THRESHOLD_PCT
) -> float:
    """Flagging cutoff at the `pct` quantile of the *validation* split's errors.

    Read off the held-out split rather than all rows because the training rows'
    errors are optimistically low by construction -- the model was fit to
    minimize them -- so a cutoff taken over everything would sit too low and
    flag the later part of every record preferentially.
    """
    held_out = errors.to_numpy()[~train_mask]
    if len(held_out) == 0:
        return float("nan")
    return float(np.quantile(held_out, pct))


class RecurrentAutoencoder(nn.Module):
    """Step 6's sequence autoencoder: the same window, read as a sequence.

    Encoder RNN over the window one bin at a time, its final hidden state
    projected to the latent; decoder repeats that latent across every bin,
    runs a second RNN over the repetition, and projects each step back to one
    value. `cell` is `nn.LSTM` or `nn.GRU` -- the two variants differ in
    nothing else, which is what makes them comparable to each other.

    The contrast with `DenseAutoencoder` is the entire point of running
    Steps 5 and 6 as separate methods, so everything except the architecture
    is held fixed: the same window, the same latent size and compression
    ratio, the same split, scaler, optimizer and stopping rule. The dense
    model gets the window flattened and so cannot know its inputs are ordered
    in time; this one is given nothing extra, only the same numbers in an
    order it can use.

    Parameter counts are *not* matched and cannot be while window and latent
    are both held fixed -- a recurrent cell simply carries more weights per
    unit of width (dense 1,574, GRU 3,171, LSTM 4,123 at the primary
    configuration). Holding the input representation and the compression
    fixed is the comparison that isolates recurrence; holding parameter count
    fixed instead would force a different latent size and reintroduce exactly
    the confound Step 4b warned about.

    One recurrent layer per side, not the deep funnel a larger sequence model
    would use: the input is 36 bins of a single variable, and Step 5 found
    this data underfits rather than overfits at a few thousand parameters.
    Dropout is omitted for the same reason.

    The hidden width follows the same geometric-mean rule as
    `DenseAutoencoder`, so both architectures funnel identically at any window
    length -- the window control refits at `CONTROL_BINS` and neither model
    should change shape for reasons unrelated to the window.
    """

    def __init__(self, window: int, bottleneck: int, cell: type[nn.RNNBase] = nn.LSTM):
        super().__init__()
        hidden = max(bottleneck + 1, round((window * bottleneck) ** 0.5))
        self.window = window
        self.encoder = cell(1, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, bottleneck)
        self.from_latent = nn.Linear(bottleneck, hidden)
        self.decoder = cell(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, state = self.encoder(x.unsqueeze(-1))
        # LSTM returns (h, c); GRU returns h alone. Either way the last layer's
        # hidden state is the window's fixed-size summary.
        last = state[0][-1] if isinstance(state, tuple) else state[-1]
        latent = self.to_latent(last)
        repeated = self.from_latent(latent).unsqueeze(1).expand(-1, self.window, -1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded).squeeze(-1)


class LSTMAutoencoder(RecurrentAutoencoder):
    """`RecurrentAutoencoder` with LSTM cells; method name `lstm_ae`."""

    def __init__(self, window: int, bottleneck: int):
        super().__init__(window, bottleneck, cell=nn.LSTM)


class GRUAutoencoder(RecurrentAutoencoder):
    """`RecurrentAutoencoder` with GRU cells; method name `gru_ae`.

    Run as a genuinely separate variant rather than assumed equivalent to the
    LSTM: the two are routinely reported as comparable in aggregate, which is
    a statement about averages over datasets and not about this one.
    """

    def __init__(self, window: int, bottleneck: int):
        super().__init__(window, bottleneck, cell=nn.GRU)
