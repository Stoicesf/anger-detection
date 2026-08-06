"""Realtime decision layer: EMA + consecutive trigger + state machine.

States:
  NORMAL  -> SUSPECT  -> ANGER  -> RECOVER  -> NORMAL

Designed to reduce flicker false alarms without changing the model.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AlarmState(str, Enum):
    NORMAL = "NORMAL"
    SUSPECT = "SUSPECT"
    ANGER = "ANGER"
    RECOVER = "RECOVER"


STATE_LABEL_ZH = {
    AlarmState.NORMAL: "非愤怒",
    AlarmState.SUSPECT: "疑似愤怒",
    AlarmState.ANGER: "愤怒",
    AlarmState.RECOVER: "恢复中",
}


class EmotionSmoother:
    """Exponential moving average over anger probability."""

    def __init__(self, alpha: float = 0.3):
        self.alpha = float(alpha)
        self.score = 0.0
        self._initialized = False

    def reset(self) -> None:
        self.score = 0.0
        self._initialized = False

    def update(self, value: float) -> float:
        v = float(value)
        if not self._initialized:
            self.score = v
            self._initialized = True
        else:
            a = self.alpha
            self.score = a * v + (1.0 - a) * self.score
        return self.score


class HitTrigger:
    """Require N consecutive above-threshold hits before firing."""

    def __init__(self, threshold: float = 0.70, required_hits: int = 3):
        self.threshold = float(threshold)
        self.required_hits = int(required_hits)
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def update(self, score: float) -> bool:
        if score > self.threshold:
            self.count += 1
        else:
            self.count = 0
        return self.count >= self.required_hits


@dataclass
class DecisionConfig:
    ema_alpha: float = 0.30
    threshold: float = 0.70
    clear_threshold: float = 0.35
    required_hits: int = 3
    require_argmax: bool = True
    recover_hold_s: float = 3.0  # low-score hold before leaving ANGER
    recover_display_s: float = 1.0  # brief RECOVER UI state


@dataclass
class DecisionOutput:
    state: AlarmState
    raw_score: float
    smooth_score: float
    hit_count: int
    is_alarm: bool  # True only in ANGER
    detail: str


class AngerStateMachine:
    """Event detector on top of per-frame anger probabilities."""

    ANGRY_IDX = 2

    def __init__(self, cfg: DecisionConfig | None = None):
        self.cfg = cfg or DecisionConfig()
        self.smoother = EmotionSmoother(self.cfg.ema_alpha)
        self.trigger = HitTrigger(self.cfg.threshold, self.cfg.required_hits)
        self.state = AlarmState.NORMAL
        self._low_since: float | None = None
        self._recover_since: float | None = None

    def reset(self) -> None:
        self.smoother.reset()
        self.trigger.reset()
        self.state = AlarmState.NORMAL
        self._low_since = None
        self._recover_since = None

    def configure(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        self.smoother.alpha = self.cfg.ema_alpha
        self.trigger.threshold = self.cfg.threshold
        self.trigger.required_hits = self.cfg.required_hits

    def update(
        self,
        raw_score: float,
        *,
        top_idx: int | None = None,
        now: float,
        silent: bool = False,
    ) -> DecisionOutput:
        cfg = self.cfg

        if silent:
            raw_score = 0.0
            smooth = self.smoother.update(0.0)
            candidate = False
        else:
            smooth = self.smoother.update(raw_score)
            candidate = smooth > cfg.threshold
            if cfg.require_argmax and top_idx is not None:
                candidate = candidate and (top_idx == self.ANGRY_IDX)

        # Drive consecutive counter only with candidate frames
        if candidate:
            self.trigger.count += 1
        else:
            self.trigger.count = 0
        hits = self.trigger.count
        fired = hits >= cfg.required_hits

        detail = ""
        prev = self.state

        if self.state == AlarmState.NORMAL:
            self._low_since = None
            self._recover_since = None
            if candidate:
                self.state = AlarmState.SUSPECT
                detail = "进入疑似（分数过阈值）"
            else:
                detail = "常规（非愤怒）"

        elif self.state == AlarmState.SUSPECT:
            self._low_since = None
            if fired:
                self.state = AlarmState.ANGER
                detail = f"连续 {hits} 次触发 → 愤怒报警"
            elif not candidate:
                self.state = AlarmState.NORMAL
                detail = "疑似解除 → 非愤怒"
            else:
                detail = f"疑似中（连续命中 {hits}/{cfg.required_hits}）"

        elif self.state == AlarmState.ANGER:
            if smooth < cfg.clear_threshold or silent:
                if self._low_since is None:
                    self._low_since = now
                held = now - self._low_since
                if held >= cfg.recover_hold_s:
                    self.state = AlarmState.RECOVER
                    self._recover_since = now
                    self.trigger.reset()
                    detail = f"低分维持 {held:.1f}s → 恢复中"
                else:
                    detail = f"愤怒保持（回落中 {held:.1f}/{cfg.recover_hold_s:.1f}s）"
            else:
                self._low_since = None
                detail = "愤怒报警中"

        elif self.state == AlarmState.RECOVER:
            # Cool-down display, then back to NORMAL
            if self._recover_since is None:
                self._recover_since = now
            if (now - self._recover_since) >= cfg.recover_display_s:
                self.state = AlarmState.NORMAL
                self._low_since = None
                self._recover_since = None
                self.trigger.reset()
                detail = "已恢复非愤怒"
            else:
                # If anger surges again during recover, go back to SUSPECT/ANGER
                if fired:
                    self.state = AlarmState.ANGER
                    self._low_since = None
                    self._recover_since = None
                    detail = "恢复中再次触发 → 愤怒"
                elif candidate:
                    self.state = AlarmState.SUSPECT
                    detail = "恢复中再次升高 → 疑似"
                else:
                    detail = "恢复中"

        if prev != self.state and not detail:
            detail = f"{prev.value} → {self.state.value}"

        return DecisionOutput(
            state=self.state,
            raw_score=float(raw_score),
            smooth_score=float(smooth),
            hit_count=int(hits),
            is_alarm=self.state == AlarmState.ANGER,
            detail=detail,
        )
