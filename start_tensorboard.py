import argparse
import logging
import os
import sys
from tensorboard import program
import time

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def _parse_args():
    parser = argparse.ArgumentParser("TensorBoard startup script.")
    parser.add_argument(
        "--s3-output-path", type=str, metavar="N", help="S3 TensorBoard output path."
    )
    parser.add_argument(
        "--job-name", type=str, metavar="N", help="SageMaker training job name."
    )
    parser.add_argument(
        "--logs-dir",
        default="logs",
        type=str,
        metavar="N",
        help="Local logs directory.",
    )
    parser.add_argument(
        "--sync-freq",
        default=2,
        type=int,
        metavar="N",
        help="Synchronization frequency, in seconds, between the s3 logs directory and "
        "the local one. Default: 2.",
    )
    return parser.parse_known_args()


def start(args):
    # s3 logs dir
    tb_artifacts_path = os.path.join(
        args.s3_output_path, args.job_name, "tensorboard-output"
    )
    logger.info(f"S3 TensorBoard artifacts path: {tb_artifacts_path}")

    # local tf dir
    local_logdir = os.path.join(args.logs_dir, args.job_name)
    os.makedirs(os.path.join(args.logs_dir, args.job_name), exist_ok=True)
    logger.info(f"Local TensorBoard logs directory: {local_logdir}")

    # startup tf server
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", local_logdir])
    url = tb.launch()
    port = url.split(":")[-1]
    logger.info(
        "TensorBoard URL, replace sXXXXXXX with your user serial number:\n"
        "https://glin-ap35001-dev-smn-sXXXXXXX.notebook.eu-central-1.sagemaker.aws/"
        f"proxy/{port}"
    )

    # sync s3 with local dir
    logger.info(
        "Starting the synchronization between the s3 and local logs directories"
    )
    while True:
        os.system(f"aws s3 sync {tb_artifacts_path} {local_logdir}")
        time.sleep(args.sync_freq)


if __name__ == "__main__":
    args, _ = _parse_args()
    start(args)
