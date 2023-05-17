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
        self.orig_tag = trial.experiment_tag,
        self.last_score = None
        self.last_checkpoint = None
        self.last_perturbation_time = 0
        self.last_velocity = 0
        self.best_position = {}
        self.best_result = None

    def __repr__(self) -> str:
        return str(
            (
                self.last_score,
                self.last_checkpoint,
                self.last_velocity = 0,
                self.best_position = {},
                self.best_result = None,
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
        global_slope: float = 0.5,
        local_slope: float = 0.5,
        synch: bool = True,
    ):
        super.__init__()
        self._time_attr = time_attr #How much train before move
        self._metric = metric #loss function
        self._mode = mode #min or max
        self._inertia = inertia #preserving the position of hyperparameter point
        self._global_slope = global_slope
        self._local_slope = local_slope
        self._synch = synch
    
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
        self, state: _PSOTrialState, result: Dict, trial: Trial, velocity
    ):
        """Saves necessary trial information when result is received.
        Args:
            state: The state object for the trial.
            time: The current timestep of the trial.
            result: The trial's result dictionary.
            trial: The trial object.
            수정 필요
        """

        # This trial has reached its perturbation interval.
        # Record new state in the state object.
        score = self._metric_op * result[self._metric]
        state.last_score = score
        state.last_result = result
        state.last_velocity = velocity

        return score
    
    def _save_best_trial_state(
        self, state: _PSOTrialState, trial : Trial
    ):
        state.best_position = "?"
        state.best_result = state.last_result
        return state
    
    def _global_best(self) -> Trial:
        trials.sort(key=lambda t: self._trial_state[t].last_score)
        return self._trial_state[-1]
    
    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner", trial: Trial):
            if trial_runner.search_alg is not None and isinstance(
            trial_runner.search_alg, SearchGenerator
        ):
            raise ValueError(
                "Search algorithms cannot be used with {} "
                "schedulers. Please remove {}.".format(
                    self.__class__.__name__, trial_runner.search_alg
                )
            )

        if not self._metric or not self._metric_op:
            raise ValueError(
                "{} has been instantiated without a valid `metric` ({}) or "
                "`mode` ({}) parameter. Either pass these parameters when "
                "instantiating the scheduler, or pass them as parameters "
                "to `tune.TuneConfig()`".format(
                    self.__class__.__name__, self._metric, self._mode
                )
            )
            
            self._trial_state[trial] = _PSOTrialState(trial)

    def on_trial_result(
        self, trial_runner: "trial_runner.TrialRunner", trial: Trial, result: Dict
    ) -> str:
        
        "main_code: have to write"
        
        
        
        "PBT"
        if self._time_attr not in result:
            time_missing_msg = (
                "Cannot find time_attr {} "
                "in trial result {}. Make sure that this "
                "attribute is returned in the "
                "results of your Trainable.".format(self._time_attr, result)
            )
            if self._require_attrs:
                raise RuntimeError(
                    time_missing_msg
                    + "If this error is expected, you can change this to "
                    "a warning message by "
                    "setting PSO(require_attrs=False)"
                )
            else:
                if log_once("pso-time_attr-error"):
                    logger.warning(time_missing_msg)
        if self._metric not in result:
            metric_missing_msg = (
                "Cannot find metric {} in trial result {}. "
                "Make sure that this attribute is returned "
                "in the "
                "results of your Trainable.".format(self._metric, result)
            )
            if self._require_attrs:
                raise RuntimeError(
                    metric_missing_msg + "If this error is expected, "
                    "you can change this to a warning message by "
                    "setting PSO(require_attrs=False)"
                )
            else:
                if log_once("pso-metric-error"):
                    logger.warning(metric_missing_msg)

        if self._metric not in result or self._time_attr not in result:
            return TrialScheduler.CONTINUE

        time = result[self._time_attr]
        state = self._trial_state[trial]

        # Continue training if burn-in period has not been reached, yet.
        if time < self._burn_in_period:
            return TrialScheduler.CONTINUE

        # Continue training if perturbation interval has not been reached, yet.
        if time - state.last_perturbation_time < self._perturbation_interval:
            return TrialScheduler.CONTINUE  # avoid checkpoint overhead

        self._save_trial_state(state, time, result, trial)

        if not self._synch:
            state.last_perturbation_time = time
            lower_quantile, upper_quantile = self._quantiles()
            decision = TrialScheduler.CONTINUE
            for other_trial in trial_runner.get_trials():
                if other_trial.status in [Trial.PENDING, Trial.PAUSED]:
                    decision = TrialScheduler.PAUSE
                    break
            self._checkpoint_or_exploit(
                trial, trial_runner, upper_quantile, lower_quantile
            )
            return TrialScheduler.NOOP if trial.status == Trial.PAUSED else decision
        else:
            # Synchronous mode.
            if any(
                self._trial_state[t].last_train_time < self._next_perturbation_sync
                and t != trial
                for t in trial_runner.get_live_trials()
            ):
                logger.debug("Pausing trial {}".format(trial))
            else:
                # All trials are synced at the same timestep.
                lower_quantile, upper_quantile = self._quantiles()
                all_trials = trial_runner.get_trials()
                not_in_quantile = []
                for t in all_trials:
                    if t not in lower_quantile and t not in upper_quantile:
                        not_in_quantile.append(t)
                # Move upper quantile trials to beginning and lower quantile
                # to end. This ensures that checkpointing of strong trials
                # occurs before exploiting of weaker ones.
                all_trials = upper_quantile + not_in_quantile + lower_quantile
                for t in all_trials:
                    logger.debug("Perturbing Trial {}".format(t))
                    self._trial_state[t].last_perturbation_time = time
                    self._checkpoint_or_exploit(
                        t, trial_runner, upper_quantile, lower_quantile
                    )

                all_train_times = [
                    self._trial_state[t].last_train_time
                    for t in trial_runner.get_trials()
                ]
                max_last_train_time = max(all_train_times)
                self._next_perturbation_sync = max(
                    self._next_perturbation_sync + self._perturbation_interval,
                    max_last_train_time,
                )
            # In sync mode we should pause all trials once result comes in.
            # Once a perturbation step happens for all trials, they should
            # still all be paused.
            # choose_trial_to_run will then pick the next trial to run out of
            # the paused trials.
            return (
                TrialScheduler.NOOP
                if trial.status == Trial.PAUSED
                else TrialScheduler.PAUSE
        
        
        
        
        

    def on_trial_complete(
        self, trial_runner: "trial_runner.TrialRunner", trial: Trial, result: Dict
    ):
        """Notification for the completion of trial.
        This will only be called when the trial is in the RUNNING state and
        either completes naturally or by manual termination."""

        raise NotImplementedError

    def on_trial_remove(self, trial_runner: "trial_runner.TrialRunner", trial: Trial):
        """Called to remove trial.
        This is called when the trial is in PAUSED or PENDING state. Otherwise,
        call `on_trial_complete`."""

        raise NotImplementedError

    def choose_trial_to_run(
        self, trial_runner: "trial_runner.TrialRunner"
    ) -> Optional[Trial]:
        """Called to choose a new trial to run.
        This should return one of the trials in trial_runner that is in
        the PENDING or PAUSED state. This function must be idempotent.
        If no trial is ready, return None."""

        raise NotImplementedError

    def debug_string(self) -> str:
        return "PopulationBasedTraining: {} checkpoints, {} perturbs".format(
            self._num_checkpoints, self._num_perturbations
        )
