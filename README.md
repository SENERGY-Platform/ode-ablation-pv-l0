# ode-ablation-pv-l0

An analytics operator for the SENERGY platform, scaffolded by the Operator
Development Environment. Every file here is yours to change, including this one.

It forecasts the PV generation of the APSystems DS3-S inverter ("Wechselrichter",
`root.powerTotal`) 24 hours ahead, as hourly mean power in watts.

## Summary: how this operator came about

### The task

Forecast photovoltaic generation 24 hours ahead as hourly mean power in watts,
judged by the developer's own `evaluation.yaml`: mean absolute error over the
September 2026 test window at a 1 h resolution, threshold 30 W. Training could use
only history before 2026-09-01. The session ran at exposure tier L0 throughout,
so no value of any series was seen while building this: every decision rests on
ontology metadata, availability, Operator Lib's source and the metrics the runs
logged.

### Result

**The threshold was not met.** The best evaluated run (commit db2f913) scored
**45.9 W** MAE over 391 scored hours of September, against 30 W. That is a third
below the first, history-only version (68.9 W) and about half the error of
repeating yesterday's value (about 85 W in backtests). Seven of the ten allowed
launches were used.

| Run | Commit | Change | Backtest MAE | September MAE |
|---|---|---|---|---|
| 1 | c6548e1 | Same hour over the last *k* days, history only | 76.3 W | 68.9 W |
| 2 | f82e174 | + day-ahead irradiance forecast (one model) | 49.7 W | 49.1 W |
| 3 | a78b635 | + ratio from similar-irradiance hours (not chosen) | 49.1 W | 48.7 W |
| 4 | 0f784f9 | Three-model mean forecast | 45.1 W | 46.6 W |
| 5 | de87cb9 | 90-day ratio lookback, 0.7 quantile option | 44.9 W | 46.8 W |
| 6 | ad5512f | Training chooses the forecast source (kept the mean) | 44.9 W | 46.8 W |
| 7 | db2f913 | Six-model mean forecast | 42.5 W | 45.9 W |

Whether 30 W is reachable for this plant at all is unknown. That depends on the
plant's typical daylight output, which a profile of the target at tier L1 would
show; the session never reached L1. Whether to revisit the threshold is the
developer's decision.

### Approach, in order

1. **Read the criteria and the scoring code first.** Operator Lib 1.8.1 scores
   each prediction in the UTC hour its *result timestamp* falls in, against the
   plain mean of the target's replayed input messages in that hour. So the
   operator stamps every prediction with `message time + 24 h`, and hours in
   which the inverter does not publish (nights) are not scored at all.
2. **Find real data, not simulated.** The ontology (Get-Power on Generation)
   gave two real PV devices. The inverter's `root.powerTotal` was chosen as
   target and only input: history since 2022-11, a cadence that keeps the
   September replay to about 20,000 messages, and the plant's own AC output.
3. **A history-only baseline.** A weighted quantile of the same UTC hour over
   the last *k* days, with *k*, the quantile and a per-day decay chosen by
   backtesting the 24 h-ahead MAE on two windows: the last 30 days of history
   and the same 30 days a year earlier (the season-matched stand-in for
   September). It beat persistence by only about 10%: day-to-day cloud cover is
   not in PV history.
4. **Bring in a weather forecast.** No device or import on the platform carries
   irradiance, and no simulatable device type has an irradiance service that
   can hold history, so the forecast is an archived file in this repository
   rather than a platform input (see "The weather input" below). The model
   multiplies the forecast irradiance for the target hour by a ratio of PV
   output to forecast irradiance, taken as a quantile over the same hour on the
   last *n* days. This was the step that mattered.
5. **Improve the forecast, not the conversion.** Three variants of the
   PV/irradiance conversion landed within about a watt of each other; averaging
   weather models did better each time (one model, then three, then six). The
   multi-model mean beats every single model.
6. **Keep every comparison fair.** Each run changed one thing, logged the
   backtest of each stage (history only, with weather, per band, per source)
   and the chosen parameters, so a run shows what each part is worth before its
   September score is read. Small changes were checked on synthetic data in the
   developer's notebook before spending a launch.

### What to know before trusting the result

- **The archived forecast only approximates what was known in advance.**
  Open-Meteo's `previous_day1` is the forecast from the model run one day
  earlier, used here as what was known 24 hours ahead. The exact issue time of
  each run is not in the file.
- **The backtest has become slightly optimistic.** Over seven runs many variants
  were chosen on the same two validation windows, and the gap to September grew
  from about 1.5 W to 3.4 W. The September score is the honest number.
- **The forecast is for one grid point**, at the location the developer gave
  (51.5 N, 10.0 E), not the exact site.
- **Live deployment is untested.** `live_weather.py` lets a deployed operator
  fetch the six-model forecast from Open-Meteo once the archive ends
  (2026-10-01). It is gated so that a replay of the past never calls out, and no
  evaluation has run it; it has not been smoke-tested yet. With `live_weather: 0`
  the operator never calls out and forecasts from history alone past the
  archive's end, which is the 68.9 W-class model.

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "forecast.py" | The forecast rule and the weather archive reader, shared by training and inference. |
| "live_weather.py" | The live Open-Meteo forecast a deployment uses past the archive's end. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "weather/" | The archived day-ahead irradiance forecast the operator reads (see below). |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.8.1". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-pv-l0". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The weather input

"weather/openmeteo_prevday1_51.5N_10.0E.csv" was fetched on 2026-10-08 from the
Open-Meteo Previous Runs API (https://previous-runs-api.open-meteo.com/v1/forecast)
for latitude 51.5, longitude 10.0, variable "shortwave_radiation_previous_day1"
(W/m²), 2025-06-01 to 2026-10-02 UTC, from six weather models: "icon_seamless"
(DWD), "ecmwf_ifs025" (ECMWF), "gfs_seamless" (NOAA), "meteofrance_seamless"
(Météo-France), "knmi_seamless" (KNMI) and "dmi_seamless" (DMI). Each model is
its own column ("ghi_<model>"); "ghi_forecast_day1", the column the operator
reads by default, is their mean. All six cover the whole file without a gap.
These are forecasts from the model runs one day earlier, not measured weather.

"ukmo_seamless" (UK Met Office) was fetched as well and left out: it is missing
102 hours inside the span training and the test month need, and a mean whose
membership changes from hour to hour is not one forecast.

The "seamless" variants blend a regional model with a global one, so the six are
not fully independent of each other; in particular the KNMI and DMI regional
models are driven by ECMWF at their boundaries.

Its "time" column is when a row became usable, not the hour it describes: the
start of the forecast hour minus 24 hours (Open-Meteo stamps the end of the hour
it averages). The operator reads a row only at or after that time, which is what
keeps the evaluation replay from seeing a forecast before it existed. The
one-day-earlier run is an approximation of what was available 24 hours ahead;
the exact issue time of each run is not in the file.

Earlier versions of this file: commits f82e174 and a78b635 carried Open-Meteo's
single default model ("best_match"); commits 0f784f9 to ad5512f the mean of
icon_seamless, ecmwf_ifs025 and gfs_seamless.

It is not a platform input. No simulatable device type carries an irradiance
service that can hold history, so the forecast lives here rather than on a
device. The file ends on 2026-10-01. Past that, a deployment with
`live_weather: 1` fetches the same six models from Open-Meteo (live_weather.py):
the latest run for the hours ahead, and the one-day-earlier forecasts for the
past 92 days the ratio is taken over. A deployment with `live_weather: 0`, or one
whose request fails, falls back to the history rule.

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-pv-l0:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.8.1", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
