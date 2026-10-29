"""Immutable direction and vehicle geometry for ordinary passage snapshots."""
from dataclasses import dataclass
from plate_ai.evidence.drawing import draw_event_box


DIRECTION_COLORS = {"IN": (0, 180, 0), "OUT": (0, 165, 255)}


@dataclass(frozen=True)
class PassageAnnotation:
    box: tuple
    track_id: int
    direction: str


def annotate_passage(frame, annotation):
    """Draw on a copy using geometry frozen when the vehicle crossed the line."""
    color = DIRECTION_COLORS[annotation.direction]
    image = frame.copy()
    draw_event_box(image, annotation.box, f"{annotation.direction} ID:{annotation.track_id}", color)
    return image
