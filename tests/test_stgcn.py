"""Tests for the ST-GCN motion encoder."""

import torch

from signtranslator.skeleton import SkeletonGraph
from signtranslator.models import STGCNEncoder, GraphConvolution


def _encoder(channels=(16, 32)):
    g = SkeletonGraph()
    return g, STGCNEncoder(in_channels=3, adjacency=g.adjacency(),
                           channels=channels, temporal_kernel=9)


def test_encoder_output_shape():
    g, enc = _encoder()
    x = torch.randn(4, 3, 32, g.num_nodes)  # (N, C, T, V)
    z = enc(x)
    assert z.shape == (4, 32)
    assert torch.isfinite(z).all()


def test_graph_conv_respects_joint_dim():
    g = SkeletonGraph()
    gc = GraphConvolution(3, 8, g.adjacency())
    x = torch.randn(2, 3, 10, g.num_nodes)
    y = gc(x)
    assert y.shape == (2, 8, 10, g.num_nodes)


def test_encoder_gradients_flow():
    g, enc = _encoder()
    x = torch.randn(3, 3, 24, g.num_nodes, requires_grad=True)
    z = enc(x)
    z.sum().backward()
    # Every trainable parameter should receive a gradient.
    grads = [p.grad for p in enc.parameters() if p.requires_grad]
    assert all(gr is not None for gr in grads)
    assert any(gr.abs().sum() > 0 for gr in grads)


def test_adjacency_is_buffer_not_parameter():
    g = SkeletonGraph()
    gc = GraphConvolution(3, 8, g.adjacency())
    param_names = {n for n, _ in gc.named_parameters()}
    buffer_names = {n for n, _ in gc.named_buffers()}
    assert "A" in buffer_names and "A" not in param_names
