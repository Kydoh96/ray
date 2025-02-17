"""Tune worker fault tolerance test.

This test checks if Tune worker fault tolerance works as expected. Worker fault
tolerance concerns the case where a worker node goes down (e.g. due to spot instance
preemption).
In this test, we start dummy trials that do nothing but sleep and checkpoint. We
also start a node killer actor, that has a chance to kill a random worker node
every N seconds.
The checkpoints are synced to S3.

If a trial is restored, it should restart from the last checkpointed iteration.

The test is succesfull if all trials finish with the expected number of iterations,
and that a checkpoint is always available when restoring.

This test only works on AWS as it uses AWS CLI to terminate nodes.

Test owner: Yard1 (Antoni)

"""

import os
import argparse
import time
import random
import gc

import ray
from ray import tune
from ray.air import session, Checkpoint
from ray.air.config import RunConfig, FailureConfig, CheckpointConfig
from ray.tune.tune_config import TuneConfig
from ray.tune.tuner import Tuner

from terminate_node_aws import create_instance_killer

MAX_ITERS = 40
ITER_TIME_BOUNDS = (60, 90)
WARMUP_TIME_S = 45


def objective(config):
    start_iteration = 0
    checkpoint = session.get_checkpoint()
    # Ensure that after the node killer warmup time, we always have
    # a checkpoint to restore from.
    if (time.monotonic() - config["start_time"]) >= config["warmup_time_s"]:
        assert checkpoint
    if checkpoint:
        start_iteration = checkpoint.to_dict()["iteration"] + 1

    for iteration in range(start_iteration, MAX_ITERS + 1):
        time.sleep(random.uniform(*ITER_TIME_BOUNDS))
        dct = {"iteration": iteration}
        session.report(dct, checkpoint=Checkpoint.from_dict(dct))


def main(bucket_uri: str):
    ray.init(log_to_driver=True, runtime_env={"working_dir": os.path.dirname(__file__)})
    num_samples = int(ray.cluster_resources()["CPU"])

    tuner = Tuner(
        objective,
        param_space={"start_time": time.monotonic(), "warmup_time_s": WARMUP_TIME_S},
        tune_config=TuneConfig(num_samples=num_samples, metric="iteration", mode="max"),
        run_config=RunConfig(
            verbose=1,
            failure_config=FailureConfig(max_failures=-1),
            sync_config=tune.SyncConfig(upload_dir=bucket_uri),
            checkpoint_config=CheckpointConfig(num_to_keep=2),
        ),
    )

    instance_killer = create_instance_killer(
        probability=0.03, time_between_checks_s=10, warmup_time_s=WARMUP_TIME_S
    )
    results = tuner.fit()
    del instance_killer
    gc.collect()

    for result in results:
        checkpoint_dict = result.checkpoint.to_dict()
        assert checkpoint_dict["iteration"] == MAX_ITERS, result.checkpoint
        assert (
            checkpoint_dict["iteration"] == result.metrics["iteration"]
        ), result.checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", type=str, help="Bucket URI")
    args, _ = parser.parse_known_args()

    main(args.bucket or "s3://tune-cloud-tests/worker_fault_tolerance")
