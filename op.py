"""The operator: what it infers per message, and when it retrains.

MLOperator is the machine-learning half of Operator Lib. It loads the model
registered under this pipeline and operator from MLflow, calls infer() for every
message that matches a selector, and calls train() when there is no model yet or
when need_retraining() says so. Training runs on Ray.

This operator forecasts PV generation 24 hours ahead as an hourly mean in watts.
Every message updates the running mean of the UTC hour it falls in, and answers
with the forecast for the hour 24 hours later, stamped with that time.
"""

import datetime
import typing

from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

import forecast as fc
from training import train_model


HORIZON = datetime.timedelta(hours=24)


class CustomConfig(Config):
    """Deployment configuration, typed.

    The base Config already carries mlflow_url, ray_url and ts_conn. Anything
    added here arrives from the operator's deployment config under the same name,
    with this value as the default.
    """

    # Retrain at most this often, in seconds. A day, so a deployment does not
    # spend its life training.
    retrain_after_s = 86400


def _python_model(model: PyFuncModel):
    unwrap = getattr(model, "unwrap_python_model", None)
    if unwrap is not None:
        return unwrap()
    return model._model_impl.python_model


class Operator(MLOperator):
    configType = CustomConfig

    selectors = [
        Selector({"name": "value", "args": ["value"]}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns, so anything set after it is missing
        # during the replay and overwrites what train() just recorded.
        self.trained_at: typing.Optional[datetime.datetime] = None
        self._sums: typing.Dict[int, float] = {}
        self._counts: typing.Dict[int, int] = {}
        self._params: typing.Dict[str, float] = dict(fc.DEFAULT_PARAMS)
        self._model_ref: typing.Optional[PyFuncModel] = None
        self._last_hour: typing.Optional[int] = None
        super().init(*args, **kwargs)

    def _adopt(self, model: PyFuncModel) -> None:
        """Take parameters and the seed history from a newly loaded model.

        Seed hours already observed live are kept as observed.
        """
        python_model = _python_model(model)
        self._params = dict(python_model.params)
        for hour, (s, c) in python_model.seed.items():
            if hour not in self._counts:
                self._sums[hour] = s
                self._counts[hour] = c
        self._model_ref = model

    def _mean(self, hour: int) -> typing.Optional[float]:
        count = self._counts.get(hour)
        if not count:
            return None
        return self._sums[hour] / count

    def _prune(self, hour: int) -> None:
        keep_from = hour - (fc.MAX_K + 2) * fc.DAY_H
        for old in [h for h in self._counts if h < keep_from]:
            del self._counts[old]
            self._sums.pop(old, None)

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        value = data.get("value")
        if value is None or model is None or isinstance(value, bool):
            return None, None, None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None, None, None

        if model is not self._model_ref:
            self._adopt(model)

        hour = fc.hour_index(timestamp)
        self._sums[hour] = self._sums.get(hour, 0.0) + value
        self._counts[hour] = self._counts.get(hour, 0) + 1
        if self._last_hour is None or hour > self._last_hour:
            self._prune(hour)
            self._last_hour = hour

        prediction = fc.forecast(self._mean, hour + fc.DAY_H, self._params)
        return timestamp + HORIZON, {"prediction": prediction}, None

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        self.trained_at = datetime.datetime.now(datetime.timezone.utc)
        return train_model(logger)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        """Time-based. Retraining re-chooses the parameters on fresher history;
        the running hourly state is kept across it."""
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(datetime.timezone.utc) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s
