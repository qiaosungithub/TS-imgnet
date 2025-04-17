from absl import logging
import jax
from flax.training import checkpoints
from .logging_utils import log_for_0
from models.tarflow import get_map_fn


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    log_for_0("Saving checkpoint step %d.", step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=2)


def restore_pretrained(state, path, config):
    pretrained = checkpoints.restore_checkpoint(path, target=None)
    pretrained_params = pretrained["ema_params"]
    log_for_0(f"pretrained model: {pretrained_params.keys()}")
    assert all(key.startswith("blocks_") for key in pretrained_params.keys())
    teacher_nblocks = len(pretrained_params)
    student_nblocks = len(state.params)
    map_fn = get_map_fn(config.load_pretrain_method, teacher_nblocks, student_nblocks)
    for i in range(student_nblocks):
        student_block = state.params[f"blocks_{i}"]
        # example_element = jax.tree_leaves(student_block)[0].reshape(-1)
        # logging.info(f'example_element at layer {i}: {example_element[0]}')
        teacher_block = pretrained_params[f"blocks_{map_fn(i)}"]
        # example_element = jax.tree_leaves(teacher_block)[0].reshape(-1)
        # logging.info(f'example_element from teacher at (student) layer {i}: {example_element[0]}')
        assert jax.tree_structure(student_block) == jax.tree_structure(teacher_block)
        state.params[f"blocks_{i}"] = teacher_block
        logging.info(f"Restored block {i} from teacher block {map_fn(i)}")

    # assert jax.tree_structure(state.params["Encoder"]) == jax.tree_structure(
    #     pretrained["ema_params"]["Encoder"]
    # )
    # assert jax.tree_structure(state.params["Decoder"]) == jax.tree_structure(
    #     pretrained["ema_params"]["Decoder"]
    # )


    # just in case
    # assert jax.tree_structure(state.batch_stats) == \
    #   jax.tree_structure(pretrained['batch_stats'])
    # state = state.replace(batch_stats=pretrained['batch_stats'])
    
    # new_block = state.params["blocks_0"]
    # example_element = jax.tree_leaves(new_block)[0].reshape(-1)
    # logging.info(f'final example_element at layer 0: {example_element[0]}')

    log_for_0("Loaded.")
    # ema
    # state = state.replace(ema_params=jax.tree_map(lambda x: jax.numpy.array(x), state.params))
    return state
