"""Day-ahead hourly PV forecast.

Shared by training.py (to backtest and choose parameters) and op.py (to answer
each message), so the rule that is tuned is exactly the rule that runs.

Time is handled as integer hour indices since the Unix epoch, in UTC. That is
the same bucketing Operator Lib's evaluation uses (fixed-width buckets from
1970-01-01T00:00Z), and the sun does not observe daylight saving time.

Two rules, blended:

- history: a weighted quantile of the hourly means at T - 1 day, ..., T - k
  days, weight decay**(lag - 1). Knows no weather.
- weather: the day-ahead irradiance forecast for T times a ratio of PV output
  to forecast irradiance, the ratio being a quantile over the same hour on the
  last n days. The same hour, because the ratio carries the panel orientation
  and the sun's angle, which differ by hour far more than by day.

  With band > 0 the ratio is taken only from those past hours whose forecast
  irradiance was within +-band of the target's -- analogues -- because a tilted
  panel converts a bright, mostly direct hour and a dull, mostly diffuse hour at
  different ratios. Fewer than MIN_ANALOGS analogues fall back to all of them.

prediction = alpha * weather + (1 - alpha) * history, falling back to history
alone where there is no forecast for T or no usable ratio.
"""

import csv
import datetime
import os
import typing

HOUR_S = 3600
DAY_H = 24

# --- history rule -----------------------------------------------------------

# k=1 is plain persistence (same hour yesterday), kept in the grid so the
# backtest always reports it as the baseline to beat.
HISTORY_GRID: typing.List[typing.Dict[str, float]] = [
    {"k": k, "q": q, "decay": decay}
    for k in (1, 3, 5, 7, 10, 14, 21)
    for q in (0.4, 0.5, 0.6)
    for decay in (1.0, 0.85, 0.7)
]
DEFAULT_PARAMS: typing.Dict[str, float] = {"k": 7, "q": 0.5, "decay": 1.0, "alpha": 0.0}
PERSISTENCE_PARAMS: typing.Dict[str, float] = {"k": 1, "q": 0.5, "decay": 1.0, "alpha": 0.0}

# --- weather rule -----------------------------------------------------------

# band 0.0 is the unconditioned ratio of the previous commit, kept so the
# backtest reports what the analogue band is worth.
WEATHER_GRID: typing.List[typing.Dict[str, float]] = [
    {"n": n, "rq": rq, "alpha": alpha, "band": band}
    for n in (7, 14, 21, 30, 60)
    for rq in (0.4, 0.5, 0.6)
    for alpha in (1.0, 0.75, 0.5)
    for band in (0.0, 0.3)
]
# Below this forecast irradiance, in W/m², a ratio PV/irradiance is noise
# divided by almost nothing and is not used.
MIN_GHI = 20.0
MIN_ANALOGS = 3

MAX_LOOKBACK_DAYS = max(
    [int(p["k"]) for p in HISTORY_GRID] + [int(p["n"]) for p in WEATHER_GRID]
)

WEATHER_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "weather",
    "openmeteo_prevday1_51.5N_10.0E.csv",
)


def hour_index(at: datetime.datetime) -> int:
    """The UTC hour bucket `at` falls in, as hours since the epoch. Naive is UTC."""
    if at.tzinfo is None:
        at = at.replace(tzinfo=datetime.timezone.utc)
    return int(at.timestamp() // HOUR_S)


class WeatherArchive:
    """Archived day-ahead irradiance forecasts, read with an availability guard.

    The file is Open-Meteo's Previous Runs API for 51.5 N 10.0 E,
    shortwave_radiation_previous_day1: for each hour, what the model run of one
    day earlier forecast. Its `time` column is the moment a row became usable:
    the start of the forecast hour (Open-Meteo stamps the end of the hour it
    averages) minus 24 hours. A row published at hour p is the forecast for hour
    p + 24.

    get() answers only when the row's publication hour is at or before the hour
    asking, so a replay can never read a forecast before it would have existed.

    This is a file, not a feed. It ends on 2026-10-01; past that, get() answers
    None and the operator falls back to the history rule. A deployment needs a
    live day-ahead forecast in its place.
    """

    def __init__(self, ghi_by_target_hour: typing.Dict[int, float]) -> None:
        self._ghi = ghi_by_target_hour

    @classmethod
    def load(cls, path: str = WEATHER_FILE) -> "WeatherArchive":
        ghi: typing.Dict[int, float] = {}
        if not os.path.exists(path):
            return cls(ghi)
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                value = row.get("ghi_forecast_day1")
                if value in (None, ""):
                    continue
                published = datetime.datetime.strptime(
                    row["time"], "%Y-%m-%dT%H:%M:%SZ"
                ).replace(tzinfo=datetime.timezone.utc)
                ghi[hour_index(published) + DAY_H] = float(value)
        return cls(ghi)

    def __len__(self) -> int:
        return len(self._ghi)

    def get(self, target_hour: int, now_hour: int) -> typing.Optional[float]:
        if target_hour - DAY_H > now_hour:
            return None
        return self._ghi.get(target_hour)

    def get_past(self, hour: int) -> typing.Optional[float]:
        """The forecast that was issued for an hour already in the past -- for
        the ratio, which compares what was forecast with what happened."""
        return self._ghi.get(hour)


def weighted_quantile(values: typing.Sequence[float], weights: typing.Sequence[float], q: float) -> float:
    pairs = sorted(zip(values, weights))
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    target = q * total
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= target - 1e-12:
            return value
    return pairs[-1][0]


def history_forecast(
    lookup: typing.Callable[[int], typing.Optional[float]],
    target_hour: int,
    params: typing.Dict[str, float],
) -> float:
    k = int(params["k"])
    q = float(params["q"])
    decay = float(params["decay"])
    values, weights = [], []
    for lag in range(1, k + 1):
        value = lookup(target_hour - lag * DAY_H)
        if value is None:
            continue
        values.append(value)
        weights.append(decay ** (lag - 1))
    if not values:
        return 0.0
    return max(0.0, float(weighted_quantile(values, weights, q)))


def weather_forecast(
    lookup: typing.Callable[[int], typing.Optional[float]],
    weather: WeatherArchive,
    target_hour: int,
    now_hour: int,
    params: typing.Dict[str, float],
) -> typing.Optional[float]:
    ghi = weather.get(target_hour, now_hour)
    if ghi is None:
        return None
    if ghi < MIN_GHI:
        return 0.0
    candidates = []
    for lag in range(1, int(params["n"]) + 1):
        hour = target_hour - lag * DAY_H
        past_ghi = weather.get_past(hour)
        actual = lookup(hour)
        if past_ghi is None or actual is None or past_ghi < MIN_GHI:
            continue
        candidates.append((actual / past_ghi, past_ghi))
    if not candidates:
        return None
    band = float(params.get("band", 0.0))
    ratios = [r for r, _ in candidates]
    if band > 0.0:
        close = [r for r, g in candidates if abs(g - ghi) <= band * ghi]
        if len(close) >= MIN_ANALOGS:
            ratios = close
    ratio = weighted_quantile(ratios, [1.0] * len(ratios), float(params["rq"]))
    return max(0.0, ratio * ghi)


def forecast(
    lookup: typing.Callable[[int], typing.Optional[float]],
    target_hour: int,
    params: typing.Dict[str, float],
    weather: typing.Optional[WeatherArchive] = None,
    now_hour: typing.Optional[int] = None,
) -> float:
    """Forecast the mean of `target_hour`. `now_hour` defaults to 24 h before it."""
    hist = history_forecast(lookup, target_hour, params)
    alpha = float(params.get("alpha", 0.0))
    if weather is None or alpha <= 0.0:
        return hist
    if now_hour is None:
        now_hour = target_hour - DAY_H
    wx = weather_forecast(lookup, weather, target_hour, now_hour, params)
    if wx is None:
        return hist
    return alpha * wx + (1.0 - alpha) * hist


def backtest(
    means: typing.Dict[int, float],
    start_hour: int,
    end_hour: int,
    params: typing.Dict[str, float],
    weather: typing.Optional[WeatherArchive] = None,
) -> typing.Tuple[typing.Optional[float], int]:
    """MAE over every hour in [start_hour, end_hour) that has an actual.

    Hours without an observation are not scored, as in the evaluation. Each
    hour is forecast as of 24 hours before it, from complete earlier days.
    """
    total, n = 0.0, 0
    for target in range(start_hour, end_hour):
        actual = means.get(target)
        if actual is None:
            continue
        total += abs(forecast(means.get, target, params, weather) - actual)
        n += 1
    return (total / n if n else None), n
