"""Training, on Ray.

Separate from op.py because the two run in different places: op.py runs in the
operator's own process for every message, while this runs distributed and rarely.

What is trained: the parameters of forecast.forecast(), in two stages, each
chosen by backtesting the 24 hour ahead MAE on hourly means.

1. The history rule alone -- how many earlier days, which quantile, how fast
   older days lose weight.
2. With that fixed, the weather rule -- how many days the PV/irradiance ratio is
   taken over, which quantile of it, and how much of the forecast it carries.

Validation windows: the last 30 days of history, and the same 30 days one year
before the end of history, the season-matched stand-in for the month that
follows. Both stages log their MAE, so a run shows what the weather input buys.

The weather input is an archived day-ahead irradiance forecast in this
repository (weather/, see forecast.WeatherArchive), not a platform input.
"""

import datetime
import typing

import pandas as pd
import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger

import forecast as fc


# A year for the season-matched validation window, plus the longest lookback
# the grid can ask for, plus slack.
TRAINING_WINDOW = datetime.timedelta(days=400)
VALIDATION_DAYS = 30
# A validation window with fewer scored hours than this is left out of the
# selection rather than allowed to decide it.
MIN_VALIDATION_HOURS = 100


class OdeAblationPvL0Model(PythonModel):
    """The model MLflow registers and op.py later loads.

    Carries the chosen parameters and the hourly sums and counts of the last
    MAX_LOOKBACK_DAYS + 2 days of history, so that the operator can forecast
    from its first message. The weather archive is not part of the model: it is
    read from the repository by whoever runs the model.
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
        return fc.forecast(means.get, int(payload["target_hour"]), self.params,
                           fc.WeatherArchive.load())


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


def _windows(means, end_hour):
    candidates = {
        "last30d": (end_hour - VALIDATION_DAYS * fc.DAY_H, end_hour),
        "prev_year": (end_hour - 365 * fc.DAY_H, end_hour - (365 - VALIDATION_DAYS) * fc.DAY_H),
    }
    usable = {}
    for name, (a, b) in candidates.items():
        _, n = fc.backtest(means, a, b, fc.DEFAULT_PARAMS)
        if n >= MIN_VALIDATION_HOURS:
            usable[name] = (a, b)
    return usable


def _best(means, usable, grid, weather):
    best = None
    for params in grid:
        maes = {}
        for name, (a, b) in usable.items():
            maes[name] = fc.backtest(means, a, b, params, weather)[0]
        if not maes or any(v is None for v in maes.values()):
            continue
        score = sum(maes.values()) / len(maes)
        if best is None or score < best[0]:
            best = (score, params, maes)
    return best


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    """Read the history, choose the parameters, and hand back a model."""
    with logger.trace("read history"):
        datasets = provide_historic_data(TRAINING_WINDOW)
    if not datasets:
        return None

    # One input: the series being forecast.
    with logger.trace("hourly means"):
        stats = ray.get(_hourly_sums.remote(datasets[0]))
    if not stats:
        return None
    means = {h: s / c for h, (s, c) in stats.items() if c}
    end_hour = max(means) + 1

    weather = fc.WeatherArchive.load()

    with logger.trace("select parameters"):
        usable = _windows(means, end_hour)
        history_grid = [dict(p, alpha=0.0) for p in fc.HISTORY_GRID]
        best_hist = _best(means, usable, history_grid, None)
        hist_params = best_hist[1] if best_hist is not None else dict(fc.DEFAULT_PARAMS)

        best_wx = None
        if len(weather):
            weather_grid = [dict(hist_params, **p) for p in fc.WEATHER_GRID]
            best_wx = _best(means, usable, weather_grid, weather)

        if best_wx is not None and (best_hist is None or best_wx[0] < best_hist[0]):
            chosen = best_wx
        else:
            chosen = best_hist
        params = chosen[1] if chosen is not None else dict(fc.DEFAULT_PARAMS)

        persistence = {
            name: fc.backtest(means, a, b, fc.PERSISTENCE_PARAMS) for name, (a, b) in usable.items()
        }

    seed_from = end_hour - (fc.MAX_LOOKBACK_DAYS + 2) * fc.DAY_H
    seed = {h: v for h, v in stats.items() if h >= seed_from}

    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "validation_windows": ",".join(sorted(usable)) or "none",
        "weather_source": "open-meteo previous-runs shortwave_radiation_previous_day1, 51.5N 10.0E, repository file",
        "weather_hours": len(weather),
        "k": params["k"],
        "q": params["q"],
        "decay": params["decay"],
        "alpha": params.get("alpha", 0.0),
        "n": params.get("n", 0),
        "rq": params.get("rq", 0.0),
    })
    metrics = {
        "hours_with_data": float(len(means)),
        "seed_hours": float(len(seed)),
    }
    if chosen is not None:
        metrics["val_mae"] = chosen[0]
        for name, value in chosen[2].items():
            metrics[f"val_mae_{name}"] = value
    if best_hist is not None:
        metrics["val_mae_history_only"] = best_hist[0]
        for name, value in best_hist[2].items():
            metrics[f"val_mae_history_only_{name}"] = value
    if best_wx is not None:
        metrics["val_mae_with_weather"] = best_wx[0]
    for name, (value, n) in persistence.items():
        if value is not None:
            metrics[f"persistence_mae_{name}"] = value
            metrics[f"val_hours_{name}"] = float(n)
    logger.log_metrics(metrics)
    return OdeAblationPvL0Model(params=params, seed=seed)
