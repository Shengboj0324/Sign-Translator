"""Temporary evaluation mode without changing a caller's mixed module modes."""
from contextlib import contextmanager


@contextmanager
def preserving_eval_mode(model):
    """Restore each existing module's training flag, including on entry failure.

    Recursive ``train(parent_mode)`` cannot restore a mixed-mode hierarchy. Assign
    the saved flags directly so restoration does not recursively overwrite child
    flags. This preserves mode flags only, not arbitrary custom train/eval side
    effects, parameter mutations or changes to the module hierarchy.
    """
    modes = tuple((module, module.training) for module in model.modules())
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training
