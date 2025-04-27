import jax

jax.distributed.initialize()
# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main file for running the ImageNet example.

This file is intentionally kept short. The majority for logic is in libraries
that can be easily tested and imported in Colab.
"""

import os
from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags

import train
from utils import logging_utils
from utils.logging_utils import log_for_0

logging_utils.supress_checkpt_info()

import warnings

warnings.filterwarnings("ignore")


FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_enum(
    "mode",
    enum_values=["local_debug", "remote_debug", "remote_run"],
    default="remote_run",
    help="Running mode.",
)  # NOTE: This variable isn't used currently, but maintained for future use. This at least ensures that there is no more variable that must be passed in from the command line.

flags.DEFINE_bool("debug", False, "Debugging mode.")
config_flags.DEFINE_config_file(
    "config",
    help_string="File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    log_for_0("JAX process: %d / %d", jax.process_index(), jax.process_count())
    log_for_0("JAX local devices: %r", jax.local_devices())

    c = FLAGS.config

    def f():
        if FLAGS.debug:
            with jax.disable_jit():
                train.train_and_evaluate(FLAGS.config, FLAGS.workdir)
        else:
            train.train_and_evaluate(FLAGS.config, FLAGS.workdir)

    # f()

    if not c.just_evaluate: raise SystemError(f"please modify main.py!!!")
    for guidance in [2.5, 3.5]:
        log_for_0("Guidance: %f", guidance)
        c.fid.guidance = guidance
        c.wandb_name = f"Stu-default-cfg-ma-{guidance}-5k"
        f()


if __name__ == "__main__":
    flags.mark_flags_as_required(["workdir", "mode", "config"])
    app.run(main)