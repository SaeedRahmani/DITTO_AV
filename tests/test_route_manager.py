"""Near/far route conditioning cursors (pure logic, no carla needed).

Semantics measured from 2.6k annotation frames (2026-07-28): near =
dense-plan node (1-2 m spacing) popped at ~4 m, hovering 4-6 m ahead;
far = sparse command-plan node (20-50 m spacing) popped at ~7.5 m.
"""
import numpy as np

from ditto_av.carla_agent import RouteCursor


def _dense(n=200, cmd=4):
    xy = np.stack([np.arange(n, dtype=float), np.zeros(n)], axis=1)
    return xy, [cmd] * n


def test_near_cursor_hovers_at_pop_radius():
    # ego drives the route; the near node stays 4-5 m ahead (pop 4,
    # spacing 1) and never falls behind
    xy, cmds = _dense()
    cur = RouteCursor(xy, cmds, pop_radius=4.0)
    for x in np.arange(0.0, 150.0, 0.5):
        node, cmd = cur.step(np.array([x, 0.0]))
        ahead = node[0] - x
        assert 4.0 - 0.5 <= ahead <= 5.0 + 1e-9, (x, node)
        assert cmd == 4


def test_far_cursor_sparse_spacing():
    xy = np.stack([np.arange(0.0, 300.0, 30.0), np.zeros(10)], axis=1)
    cur = RouteCursor(xy, [4] * 10, pop_radius=7.5)
    dists = []
    for x in np.arange(0.0, 250.0, 1.0):
        node, _ = cur.step(np.array([x, 0.0]))
        dists.append(node[0] - x)
    # far node distance lives in [7.5, 7.5+spacing], never behind
    assert min(dists) >= 7.5 - 1.0
    assert max(dists) <= 37.5 + 1e-9


def test_cursor_index_is_monotonic():
    # a backwards jump (reversing, loops) must not rewind the target
    xy, cmds = _dense()
    cur = RouteCursor(xy, cmds, pop_radius=4.0)
    for x in np.arange(0.0, 50.0, 0.5):   # drive to x=50
        cur.step(np.array([x, 0.0]))
    i_before = cur.i
    node, _ = cur.step(np.array([10.0, 0.0]))
    assert cur.i == i_before and node[0] >= 50.0


def test_command_propagates_from_node():
    xy, cmds = _dense(100)
    cmds[60:70] = [1] * 10  # LEFT segment ahead
    cur = RouteCursor(xy, cmds, pop_radius=4.0)
    for x in np.arange(0.0, 57.0, 0.5):
        cur.step(np.array([x, 0.0]))
    node, cmd = cur.step(np.array([57.0, 0.0]))
    assert cmd == 1 and 60.0 <= node[0] <= 62.0


def test_end_of_plan_clamps():
    xy, cmds = _dense(20)
    cur = RouteCursor(xy, cmds, pop_radius=4.0)
    for x in np.arange(0.0, 100.0, 0.5):  # drive past the end
        node, cmd = cur.step(np.array([x, 0.0]))
    assert node[0] == 19.0 and cmd == 4


def test_empty_plan():
    cur = RouteCursor(np.zeros((0, 2)), [], pop_radius=4.0)
    node, cmd = cur.step(np.array([0.0, 0.0]))
    assert node is None and cmd == 4
