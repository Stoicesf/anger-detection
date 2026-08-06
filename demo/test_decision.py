"""Unit tests for Phase-1 decision layer (no mic / model needed)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demo"))

from decision import AlarmState, AngerStateMachine, DecisionConfig, EmotionSmoother


def test_ema_smooths_spike():
    s = EmotionSmoother(alpha=0.3)
    vals = [0.2, 0.8, 0.9, 0.3]
    out = [s.update(v) for v in vals]
    assert out[0] == 0.2
    assert out[1] < 0.8  # smoothed down from spike
    assert abs(out[1] - 0.38) < 1e-6


def test_state_machine_requires_consecutive_hits():
    sm = AngerStateMachine(
        DecisionConfig(
            ema_alpha=1.0,  # no smoothing for deterministic test
            threshold=0.7,
            required_hits=3,
            require_argmax=False,
            recover_hold_s=2.0,
            recover_display_s=0.5,
        )
    )
    t = 0.0
    # two hits -> still SUSPECT, not ANGER
    for _ in range(2):
        t += 0.4
        out = sm.update(0.85, now=t)
    assert out.state == AlarmState.SUSPECT
    assert not out.is_alarm
    # third hit -> ANGER
    t += 0.4
    out = sm.update(0.85, now=t)
    assert out.state == AlarmState.ANGER
    assert out.is_alarm


def test_recover_after_low_hold():
    sm = AngerStateMachine(
        DecisionConfig(
            ema_alpha=1.0,
            threshold=0.7,
            clear_threshold=0.35,
            required_hits=2,
            require_argmax=False,
            recover_hold_s=1.0,
            recover_display_s=0.2,
        )
    )
    t = 0.0
    sm.update(0.9, now=t)
    t += 0.4
    out = sm.update(0.9, now=t)
    assert out.state == AlarmState.ANGER
    # stay low for recover_hold_s
    t += 0.3
    out = sm.update(0.1, now=t)
    assert out.state == AlarmState.ANGER  # just started falling
    t += 1.1
    out = sm.update(0.1, now=t)
    assert out.state == AlarmState.RECOVER
    t += 0.3
    out = sm.update(0.1, now=t)
    assert out.state == AlarmState.NORMAL


def test_flicker_does_not_alarm():
    """Alternating high/low should not reach ANGER with required_hits=3."""
    sm = AngerStateMachine(
        DecisionConfig(
            ema_alpha=1.0,
            threshold=0.7,
            required_hits=3,
            require_argmax=False,
        )
    )
    t = 0.0
    seq = [0.8, 0.2, 0.8, 0.2, 0.8, 0.2]
    states = []
    for v in seq:
        t += 0.4
        states.append(sm.update(v, now=t).state)
    assert AlarmState.ANGER not in states


if __name__ == "__main__":
    test_ema_smooths_spike()
    test_state_machine_requires_consecutive_hits()
    test_recover_after_low_hold()
    test_flicker_does_not_alarm()
    print("ALL decision tests passed")
