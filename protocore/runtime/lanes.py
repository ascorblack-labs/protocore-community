"""Named lanes over shared history. Main always exists; extras take exclusive locks."""
from __future__ import annotations

from dataclasses import dataclass

from protocore.contracts.runtime_constants import RuntimeConstants

MAIN = "main"


@dataclass(slots=True)
class Lane:
    lane_id: str
    cursor: int
    model: str
    toolset: tuple[str, ...]
    locked_by: str | None = None
    diverged: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "lane_id": self.lane_id,
            "cursor": self.cursor,
            "model": self.model,
            "toolset": list(self.toolset),
            "locked_by": self.locked_by,
            "diverged": self.diverged,
        }


def ensure_main(lanes: list[Lane], *, model: str = "", toolset: tuple[str, ...] = ()) -> list[Lane]:
    if any(item.lane_id == MAIN for item in lanes):
        return list(lanes)
    return [Lane(lane_id=MAIN, cursor=0, model=model, toolset=toolset), *lanes]


def refuse_lanes_when_disabled(enabled: bool) -> None:
    if not enabled:
        raise ValueError("lanes_disabled")


def create_lane(
    lanes: list[Lane],
    *,
    lane_id: str,
    cursor: int,
    model: str,
    toolset: tuple[str, ...],
    rc: RuntimeConstants,
) -> list[Lane]:
    refuse_lanes_when_disabled(rc.lanes_enabled)
    current = ensure_main(lanes, model=model, toolset=toolset)
    if any(item.lane_id == lane_id for item in current):
        raise ValueError("lane_exists")
    if len(current) >= rc.lanes_max_per_session:
        raise ValueError("lanes_full")
    return [*current, Lane(lane_id=lane_id, cursor=cursor, model=model, toolset=toolset)]


def acquire_lane(lanes: list[Lane], lane_id: str, owner: str) -> list[Lane]:
    out: list[Lane] = []
    found = False
    for item in lanes:
        if item.lane_id != lane_id:
            out.append(item)
            continue
        found = True
        if item.locked_by and item.locked_by != owner:
            raise ValueError("lane_locked")
        item.locked_by = owner
        out.append(item)
    if not found:
        raise ValueError("unknown_lane")
    return out


def release_lane(lanes: list[Lane], lane_id: str, owner: str) -> list[Lane]:
    for item in lanes:
        if item.lane_id == lane_id and item.locked_by == owner:
            item.locked_by = None
    return lanes


def mark_diverged(lanes: list[Lane], lane_id: str) -> list[Lane]:
    for item in lanes:
        if item.lane_id == lane_id:
            item.diverged = True
    return lanes


def reviewer_blocks_main(lanes: list[Lane]) -> bool:
    """A reviewer lane after diverge must not hold main's lock."""
    main = next((item for item in lanes if item.lane_id == MAIN), None)
    extra = [item for item in lanes if item.lane_id != MAIN]
    if main is None:
        return False
    if any(item.diverged for item in extra) and main.locked_by is None:
        return False
    return bool(main.locked_by and any(item.locked_by == main.locked_by for item in extra))


__all__ = [
    "MAIN",
    "Lane",
    "acquire_lane",
    "create_lane",
    "ensure_main",
    "mark_diverged",
    "refuse_lanes_when_disabled",
    "release_lane",
    "reviewer_blocks_main",
]
