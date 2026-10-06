"""Necessary AdamW state domains; these do not prove optimization history."""
from collections.abc import Mapping

import torch


def validate_adam_state(state_dict, parameter_groups, *, global_step):
    """Validate serialized state without mutating receiver tensors or RNG.

    State may be absent for parameters that never received gradients. Present
    counters may differ because support and trainability differ by parameter.
    """
    def fail(reason):
        raise ValueError(f'invalid governed AdamW state: {reason}')

    if not isinstance(state_dict, Mapping) or set(state_dict) != {'state', 'param_groups'}:
        fail('unexpected state dictionary')
    states, groups = state_dict['state'], state_dict['param_groups']
    if not isinstance(states, Mapping) or not isinstance(groups, list) or len(groups) != len(parameter_groups):
        fail('parameter groups or state mapping')
    owned = set()
    for group, parameters in zip(groups, parameter_groups):
        if not isinstance(group, Mapping) or not isinstance(group.get('params'), list):
            fail('parameter identifiers')
        identifiers = group['params']
        if len(identifiers) != len(parameters):
            fail('parameter count')
        for identifier, parameter in zip(identifiers, parameters):
            if type(identifier) is not int or identifier < 0 or identifier in owned:
                fail('duplicate or invalid parameter identifier')
            owned.add(identifier)
            if identifier not in states:
                continue
            state = states[identifier]
            fields = {'step', 'exp_avg', 'exp_avg_sq'}
            if group.get('amsgrad'):
                fields.add('max_exp_avg_sq')
            if not isinstance(state, Mapping) or set(state) != fields:
                fail('missing or unexpected per-parameter fields')
            step = state['step']
            if (not isinstance(step, torch.Tensor) or step.ndim != 0
                    or not step.is_floating_point() or not bool(torch.isfinite(step))):
                fail('step must be a finite floating scalar tensor')
            count = step.item()
            if count < 1 or count > global_step or not count.is_integer():
                fail('step must be an integer in the recorded update range')
            for name in fields - {'step'}:
                value = state[name]
                if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                        or value.shape != parameter.shape or value.dtype != parameter.dtype
                        or not bool(torch.isfinite(value).all())):
                    fail(f'{name} tensor contract')
                # AdamW applies the real-valued update independently to the
                # two components of a complex parameter.
                components = torch.view_as_real(value) if value.is_complex() else value
                if name != 'exp_avg' and bool((components < 0).any()):
                    fail(f'{name} must be nonnegative')
            if 'max_exp_avg_sq' in fields:
                current, maximum = state['exp_avg_sq'], state['max_exp_avg_sq']
                if current.is_complex():
                    current, maximum = torch.view_as_real(current), torch.view_as_real(maximum)
                if bool((maximum < current).any()):
                    fail('AMSGrad maximum is below the current second moment')
    if any(type(key) is not int or key not in owned for key in states):
        fail('state belongs to an unknown parameter')
