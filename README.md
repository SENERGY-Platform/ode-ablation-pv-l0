# ode-ablation-pv-l0

An analytics operator for the SENERGY platform, scaffolded by the Operator
Development Environment. Every file here is yours to change, including this one.

It forecasts the PV generation of the APSystems DS3-S inverter ("Wechselrichter",
`root.powerTotal`) 24 hours ahead, as hourly mean power in watts.

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "forecast.py" | The forecast rule and the weather archive reader, shared by training and inference. |
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
(W/m²), 2025-06-01 to 2026-10-02 UTC, from three weather models:
"icon_seamless" (DWD), "ecmwf_ifs025" (ECMWF) and "gfs_seamless" (NOAA). Each
model is its own column ("ghi_<model>"); "ghi_forecast_day1", the column the
operator reads, is their mean. All three cover the whole file without a gap.
These are forecasts from the model runs one day earlier, not measured weather.

Its "time" column is when a row became usable, not the hour it describes: the
start of the forecast hour minus 24 hours (Open-Meteo stamps the end of the hour
it averages). The operator reads a row only at or after that time, which is what
keeps the evaluation replay from seeing a forecast before it existed. The
one-day-earlier run is an approximation of what was available 24 hours ahead;
the exact issue time of each run is not in the file.

An earlier version of this file (commits f82e174 and a78b635) carried Open-Meteo's
single default model ("best_match") instead of the three-model mean.

It is not a platform input. No simulatable device type carries an irradiance
service that can hold history, so the forecast lives here rather than on a
device. The file ends on 2026-10-01; after that the operator falls back to its
history-only rule. A deployment needs a live day-ahead forecast in its place.

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
