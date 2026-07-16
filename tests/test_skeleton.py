"""Tests for skeleton-graph construction and adjacency normalisation."""

import numpy as np
import pytest

from signtranslator.skeleton import SkeletonGraph, NUM_DEFAULT_JOINTS


def test_default_graph_is_connected_tree():
    g = SkeletonGraph()
    assert g.num_nodes == NUM_DEFAULT_JOINTS
    # A tree on V nodes has exactly V-1 edges and every hop distance is finite.
    assert len(g.edges) == g.num_nodes - 1
    assert (g.hop >= 0).all()
    assert g.hop[g.center] == 0


def test_adjacency_shape_and_partitions():
    g = SkeletonGraph()
    A = g.adjacency()
    assert A.shape == (3, g.num_nodes, g.num_nodes)  # spatial-config partitioning K=3
    assert A.dtype == np.float32
    assert np.isfinite(A).all()


def test_partition_semantics_centripetal_vs_centrifugal():
    """A centripetal neighbour must be strictly closer to the centre; centrifugal farther."""
    g = SkeletonGraph()
    _, cp, cf = g.A  # root, centripetal, centrifugal (pre-normalisation structure preserved)
    for i in range(g.num_nodes):
        for j in range(g.num_nodes):
            if cp[i, j] > 0:
                assert g.hop[j] < g.hop[i]
            if cf[i, j] > 0:
                assert g.hop[j] > g.hop[i]


def test_row_normalisation_is_substochastic_and_bounds_spectrum():
    """Row-normalised D^{-1} A has row sums <= 1, hence |eigenvalue| <= 1
    (Gershgorin), guaranteeing stable stacked graph convolutions."""
    g = SkeletonGraph()
    for k in range(g.num_partitions):
        Ak = g.A[k].astype(np.float64)
        row_sums = Ak.sum(axis=1)
        assert (row_sums <= 1.0 + 1e-6).all()
        # Directed matrices -> use the general (complex) eigenvalues.
        radius = np.abs(np.linalg.eigvals(Ak)).max()
        assert radius <= 1.0 + 1e-6


def test_self_loop_partition_is_identity_normalised():
    g = SkeletonGraph()
    root = g.A[0]
    # The root partition is the identity (self-connections); normalising identity
    # by its own degree leaves the identity.
    assert np.allclose(root, np.eye(g.num_nodes), atol=1e-6)


def test_disconnected_graph_raises():
    with pytest.raises(ValueError):
        SkeletonGraph(num_nodes=4, edges=[(0, 1)], center=0)  # nodes 2,3 unreachable


def test_self_loop_edge_rejected():
    with pytest.raises(ValueError):
        SkeletonGraph(num_nodes=3, edges=[(0, 0), (1, 2)], center=0)
