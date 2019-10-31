import os
import sys
import tensorflow as tf
import contextlib
import json
import importlib
from tensorflow.python.eager.context import num_gpus


def get_current_epoch(output_dir):
    try:
        with open(os.path.join(output_dir, "stats.json"), "r") as f:
            return json.load(f)["epoch"]
    except:
        return 0


class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs=logs)
        with open(os.path.join(os.path.dirname(self.filepath), "stats.json"), "w") as f:
            return json.dump({"epoch": epoch + 1}, f)


def get_distribution_scope(batch_size):
    if num_gpus() > 1:
        strategy = tf.distribute.MirroredStrategy()
        assert (
            batch_size % strategy.num_replicas_in_sync == 0
        ), f"Batch size {batch_size} cannot be divided onto {num_gpus()} GPUs"
        distribution_scope = strategy.scope
    else:
        if sys.version_info >= (3, 7):
            distribution_scope = contextlib.nullcontext
        else:
            distribution_scope = contextlib.suppress

    return distribution_scope()


def import_all_sub_modules(module):
    with os.scandir(os.path.join(os.path.dirname(__file__), module)) as it:
        for entry in it:
            name = entry.name
            if name.endswith(".py") and not name.startswith("_") and entry.is_file():
                importlib.import_module(f"bnn_optimization.{module}.{name[:-3]}")


def prepare_registry():
    for module in ("data", "models"):
        import_all_sub_modules(module)
