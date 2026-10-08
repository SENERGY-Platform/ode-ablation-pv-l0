"""Day-ahead hourly forecast from the series' own recent days.

Shared by training.py (to backtest and choose parameters) and op.py (to answer
each message), so the rule that is tuned is exactly the rule that runs.

Time is handled as integer hour indices since the Unix epoch, in UTC. That is
the same bucketing Operator Lib's evaluation uses (fixed-width buckets from
1970-01-01T00:00Z), and the sun does not observe daylight saving time.

The rule: the forecast for hour T is a weighted quantile of the hourly means at
T - 1 day, T - 2 days, ..., T - k days, with weight decay**(lag - 1). Hours with
no observation are skipped rather than read as zero; with no observation at all
the forecast is zero. Nothing here knows about weather -- see the README note in
training.py.
"""

import datetime
import typing

HOUR_S = 3600
DAY_H = 24

# Searched in training. k=1 is plain persistence (same hour yesterday), kept in
# the grid so the backtest always reports it as the baseline to beat.
PARAM_GRID: typing.List[typing.Dict[str, float]] = [
    {"k": k, "q": q, "decay": decay}
    for k in (1, 3, 5, 7, 10, 14, 21)
    for q in (0.4, 0.5, 0.6)
    for decay in (1.0, 0.85, 0.7)
]
MAX_K = max(int(p["k"]) for p in PARAM_GRID)
DEFAULT_PARAMS: typing.Dict[str, float] = {"k": 7, "q": 0.5, "decay": 1.0}
PERSISTENCE_PARAMS: typing.Dict[str, float] = {"k": 1, "q": 0.5, "decay": 1.0}


def hour_index(at: datetime.datetime) -> int:
    """The UTC hour bucket `at` falls in, as hours since the epoch. Naive is UTC."""
    if at.tzinfo is None:
        at = at.replace(tzinfo=datetime.timezone.utc)
    return int(at.timestamp() // HOUR_S)


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


def forecast(
    lookup: typing.Callable[[int], typing.Optional[float]],
    target_hour: int,
    params: typing.Dict[str, float],
) -> float:
    """Forecast the mean of `target_hour` from the same hour on earlier days."""
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


def backtest(
    means: typing.Dict[int, float],
    start_hour: int,
    end_hour: int,
    params: typing.Dict[str, float],
) -> typing.Tuple[typing.Optional[float], int]:
    """MAE over every hour in [start_hour, end_hour) that has an actual.

    Hours without an observation are not scored, as in the evaluation. Each
    hour is forecast from complete earlier days only, which is slightly
    stricter than the live operator, whose lag-1 hour is the running mean of
    the hour it is in.
    """
    total, n = 0.0, 0
    for target in range(start_hour, end_hour):
        actual = means.get(target)
        if actual is None:
            continue
        total += abs(forecast(means.get, target, params) - actual)
        n += 1
    return (total / n if n else None), n
