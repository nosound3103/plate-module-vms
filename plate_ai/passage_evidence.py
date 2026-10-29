"""Immutable direction and vehicle geometry for ordinary passage snapshots."""
from dataclasses import dataclass
import cv2


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
    h, w = image.shape[:2]
    coords = tuple(max(0, min((w-1) if i % 2 == 0 else (h-1),
                             int(round(value*((w-1) if i % 2 == 0 else (h-1))))))
                   for i, value in enumerate(annotation.box))
    x1, y1, x2, y2 = coords
    cv2.rectangle(image, (x1, y1), (x2, y2), color, max(2, round(min(h, w)/300)))
    label = f"{annotation.direction} ID:{annotation.track_id}"
    scale = min(.8, max(.3, w/1600))
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    scale *= min(1., max(1, w-8)/max(1, tw))
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    tx = max(0, min(x1, w-tw-4))
    ty = max(th+3, min(h-baseline-2, y1-6 if y1 > th+8 else y2+th+6))
    cv2.rectangle(image, (tx, max(0, ty-th-3)), (min(w-1, tx+tw+4), min(h-1, ty+baseline+2)), color, -1)
    cv2.putText(image, label, (tx+2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return image
