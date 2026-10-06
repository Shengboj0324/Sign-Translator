"""Synthetic collation checks; these fixtures confer no source or phase approval."""
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from signtranslator.data.multichannel import collate_multichannel
from test_pose_multichannel import fixture


def short_state():
    state = fixture()
    return replace(state, sample_id="short", clock_id="independent-clock",
                   timestamps=np.array([7., 7.25]),
                   channels={name: replace(channel, **{
                       field: getattr(channel, field)[:2].copy()
                       for field in ("values", "valid", "observed", "confidence")})
                       for name, channel in state.channels.items()})


def test_dataloader_retains_channels_clock_domains_dtypes_and_padding():
    states = [fixture(), short_state()]
    batch = next(iter(DataLoader(states, batch_size=2, collate_fn=collate_multichannel)))
    assert batch.sample_ids == ("fixture", "short")
    assert batch.clock_ids == ("synthetic-clock", "independent-clock")
    assert batch.timestamps.dtype == torch.float64
    assert batch.lengths.tolist() == [3, 2]
    assert batch.frame_valid.tolist() == [[True, True, True], [True, True, False]]
    for row, state in enumerate(states):
        count = len(state.timestamps)
        np.testing.assert_array_equal(batch.timestamps[row, :count].numpy(), state.timestamps)
        for name, source in state.channels.items():
            channel = batch.channels[name]
            for field in ("values", "valid", "observed", "confidence"):
                value = getattr(channel, field)
                np.testing.assert_array_equal(value[row, :count].numpy(), getattr(source, field))
                assert value.numpy().dtype == getattr(source, field).dtype
                assert torch.count_nonzero(value[row, count:]) == 0
            assert channel.layout.labels == source.labels
            assert channel.provenance[row].source_sha256 == source.source_sha256
    # No alias in either direction between source arrays and batched tensors.
    batch.channels["face"].values[0, 0, 0, 0] = 3
    assert states[0].channels["face"].values[0, 0, 0] == 0
    states[0].channels["face"].values[1, 0, 0] = 4
    assert batch.channels["face"].values[0, 1, 0, 0] == 0


def test_rejected_observation_and_valid_inference_stay_distinct():
    state = fixture()
    channel = replace(state.channels["face"], valid=np.array([[False], [True], [True]]),
                      observed=np.array([[True], [False], [True]]),
                      confidence=np.array([[0.], [.5], [.75]], dtype=np.float32),
                      inference_method="synthetic-fit")
    state = replace(state, channels={**state.channels, "face": channel})
    result = collate_multichannel([state, short_state()]).channels["face"]
    assert result.observed[0, 0, 0] and not result.valid[0, 0, 0]
    assert result.supervision_weights()[0, :, 0].tolist() == [0., 0., .75]
    assert result.supervision_weights(allow_inferred=True)[0, :, 0].tolist() == [0., .5, .75]
    assert result.supervision_weights(allow_inferred=True)[1, 2, 0] == 0
    assert result.provenance[0].inference_method == "synthetic-fit"
    with pytest.raises(ValueError, match="boolean"):
        result.supervision_weights(allow_inferred=1)


@pytest.mark.parametrize("change", [
    {"labels": ("different-label",)}, {"coordinate_frame": "different-frame"},
    {"convention": "different-convention"},
    {"values": np.zeros((3, 1, 1), dtype=np.float32)},
    {"confidence": np.ones((3, 1), dtype=np.float64)},
])
def test_incompatible_layout_or_dtype_is_never_silently_coerced(change):
    state = fixture()
    other = replace(state, channels={**state.channels,
                                    "face": replace(state.channels["face"], **change)})
    with pytest.raises(ValueError, match="explicit"):
        collate_multichannel([state, other])


def test_mutation_is_revalidated_before_tensor_conversion():
    state = fixture()
    state.channels["left_gaze"].values[:] = 0
    with pytest.raises(ValueError, match="unit norm"):
        collate_multichannel([state])
    with pytest.raises(ValueError, match="empty"):
        collate_multichannel([])
    with pytest.raises(ValueError, match="typed"):
        collate_multichannel([None])
