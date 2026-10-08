"""Training, on Ray.

Separate from op.py because the two run in different places: op.py runs in the
operator's own process for every message, while this runs distributed and rarely.

What is trained: the parameters of forecast.forecast() -- how many earlier days
to look at, which quantile of them to take, and how fast older days lose weight
-- chosen by backtesting the 24 hour ahead MAE on hourly means. Two validation
windows: the last 30 days of history, and the same 30 days one year before the
end of history, which is the season-matched stand-in for the month that follows.

What it does not know: weather. There is no irradiance or weather forecast on
this platform for this site, so a cloudy day after a sunny one is an error this
model cannot avoid. That is the first thing to add if the threshold is not met.
"""

import datetime
import typing

import pandas as pd
import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger

import forecast as fc


# How much history one training pass reads: a year for the season-matched
# validation window, plus the longest lookback the grid can ask for, plus slack.
TRAINING_WINDOW = datetime.timedelta(days=400)
VALIDATION_DAYS = 30
# A validation window with fewer scored hours than this is left out of the
# selection rather than allowed to decide it.
MIN_VALIDATION_HOURS = 100


class OdeAblationPvL0Model(PythonModel):
    """The model MLflow registers and op.py later loads.

    Carries the chosen parameters and the hourly sums and counts of the last
    MAX_K + 2 days of history, so that the operator can forecast from its first
    message rather than only after it has observed days of its own.
    """

    def __init__(self, params: typing.Dict[str, float], seed: typing.Dict[int, typing.Tuple[float, int]]) -> None:
        self.params = dict(params)
        self.seed = dict(seed)

    def predict(self, context, model_input=None, params=None):
        # payload: {"target_hour": int, "means": {hour_index: mean}}
        payload = model_input if model_input is not None else context
        means = payload.get("means") or {
            h: s / c for h, (s, c) in self.seed.items() if c
        }
        return fc.forecast(means.get, int(payload["target_hour"]), self.params)


@ray.remote
def _hourly_sums(dataset) -> typing.Dict[int, typing.Tuple[float, int]]:
    """Sum and count of the value per UTC hour, the evaluation's own bucketing.

    A plain mean of the messages in each hour, not a time-weighted one, because
    that is what the evaluation compares a forecast against.
    """
    sums: typing.Dict[int, float] = {}
    counts: typing.Dict[int, int] = {}
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    for batch in dataset.iter_batches(batch_size=65536, batch_format="pandas"):
        if "time" not in batch.columns or "value" not in batch.columns:
            continue
        values = pd.to_numeric(batch["value"], errors="coerce")
        times = pd.to_datetime(batch["time"], utc=True)
        hours = (times - epoch) // pd.Timedelta(hours=1)
        frame = pd.DataFrame({"h": hours, "v": values}).dropna()
        if frame.empty:
            continue
        grouped = frame.groupby("h")["v"].agg(["sum", "count"])
        for h, row in grouped.iterrows():
            h = int(h)
            sums[h] = sums.get(h, 0.0) + float(row["sum"])
            counts[h] = counts.get(h, 0) + int(row["count"])
    return {h: (sums[h], counts[h]) for h in sums}


def _select(means: typing.Dict[int, float], end_hour: int):
    windows = {
        "last30d": (end_hour - VALIDATION_DAYS * fc.DAY_H, end_hour),
        "prev_year": (end_hour - 365 * fc.DAY_H, end_hour - (365 - VALIDATION_DAYS) * fc.DAY_H),
    }
    usable = {}
    for name, (a, b) in windows.items():
        _, n = fc.backtest(means, a, b, fc.DEFAULT_PARAMS)
        if n >= MIN_VALIDATION_HOURS:
            usable[name] = (a, b)

    best = None
    for params in fc.PARAM_GRID:
        maes = {name: fc.backtest(means, a, b, params)[0] for name, (a, b) in usable.items()}
        if not maes or any(v is None for v in maes.values()):
            continue
        score = sum(maes.values()) / len(maes)
        if best is None or score < best[0]:
            best = (score, params, maes)

    persistence = {
        name: fc.backtest(means, a, b, fc.PERSISTENCE_PARAMS) for name, (a, b) in usable.items()
    }
    return best, usable, persistence


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    """Read the history, choose the parameters, and hand back a model."""
    with logger.trace("read history"):
        datasets = provide_historic_data(TRAINING_WINDOW)
    if not datasets:
        return None

    # One input: the series being forecast. A second input would be a feature,
    # which this model does not use yet.
    with logger.trace("hourly means"):
        stats = ray.get(_hourly_sums.remote(datasets[0]))
    if not stats:
        return None
    means = {h: s / c for h, (s, c) in stats.items() if c}
    end_hour = max(means) + 1

    with logger.trace("select parameters"):
        best, usable, persistence = _select(means, end_hour)

    params = best[1] if best is not None else fc.DEFAULT_PARAMS
    seed_from = end_hour - (fc.MAX_K + 2) * fc.DAY_H
    seed = {h: v for h, v in stats.items() if h >= seed_from}

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "validation_windows": ",".join(sorted(usable)) or "none",
        "k": params["k"],
        "q": params["q"],
        "decay": params["decay"],
    })
    metrics = {
        "hours_with_data": float(len(means)),
        "seed_hours": float(len(seed)),
    }
    if best is not None:
        metrics["val_mae"] = best[0]
        for name, value in best[2].items():
            metrics[f"val_mae_{name}"] = value
    for name, (value, n) in persistence.items():
        if value is not None:
            metrics[f"persistence_mae_{name}"] = value
            metrics[f"val_hours_{name}"] = float(n)
    logger.log_metrics(metrics)
    return OdeAblationPvL0Model(params=params, seed=seed)
