from typing import Dict, List, Optional
from ray.tune.result import DEFAULT_METRIC
import random

from ray.tune.execution import trial_runner
from ray.tune.result import DEFAULT_METRIC
from ray.tune.experiment import Trial
from ray.tune.schedulers.trial_scheduler import FIFOScheduler, TrialScheduler
from ray.util.annotations import PublicAPI

class _PSOTrialState:
    """Internal PSO state tracked per-trial."""
    
    def __init__(self, trial: Trial):
        self.orig_tag = trial.experiment_tag
        self.last_score = None
        self.last_checkpoint = None
        self.last_perturbation_time = 0
        self.last_train_time = 0  # Used for synchronous mode.
        self.last_result = None  # Used for synchronous mode.

    def __repr__(self) -> str:
        return str(
            (
                self.last_score,
                self.last_checkpoint,
                self.last_train_time,
                self.last_perturbation_time,
            )
        )

@PublicAPI
class ParticleSwarmOptimization(FIFOScheduler):
    def __init__(
        self,
        time_attr: str = "time_total_s",
        metric: Optional[str] = None,
        mode: Optional[str] = None,
        step_size: int = 5
        inertia: float = 0.5,
        global_slope: float = 0.5
        local_slope: float = 0.5
    ):
        super.__init__()
        self._time_attr = time_attr #How much train before move
        self._metric = metric #loss function
        self._mode = mode #min or max
        self._inertia = inertia #preserving the position of hyperparameter point
        self._global_slope = global_slope
        self._local_slope = local_slope
    
    def set_search_properties(
        self, metric: Optional[str], mode: Optional[str], **spec
    ) -> bool:
        if self._metric and metric:
            return False
        if self._mode and mode:
            return False

        if metric:
            self._metric = metric
        if mode:
            self._mode = mode

        if self._mode == "max":
            self._metric_op = 1.0
        elif self._mode == "min":
            self._metric_op = -1.0
            
        if self._metric is None and self._mode:
            # If only a mode was passed, use anonymous metric
            self._metric = DEFAULT_METRIC
        
        return True
    
    def _save_trial_state(
        self, state: _PSOTrialState, time: int, result: Dict, trial: Trial
    ):
        """Saves necessary trial information when result is received.
        Args:
            state: The state object for the trial.
            time: The current timestep of the trial.
            result: The trial's result dictionary.
            trial: The trial object.
        """

        # This trial has reached its perturbation interval.
        # Record new state in the state object.
        score = self._metric_op * result[self._metric]
        state.last_score = score
        state.last_train_time = time
        state.last_result = result

        return score
    
    def _global_best(self) -> Trial:
        trials.sort(key=lambda t: self._trial_state[t].last_score)
        return self._trial_state[0]
