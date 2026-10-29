"""Shared box/label rendering for separately annotated evidence images."""
import cv2


def draw_event_box(image, box, label, color, *, label_background=None, sort_corners=False):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (
        max(0, min((w-1) if i % 2 == 0 else (h-1),
                   int(round(value*((w-1) if i % 2 == 0 else (h-1))))))
        for i, value in enumerate(box)
    )
    if sort_corners:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, max(2, round(min(h, w)/300)))
    scale = min(.8, max(.3, w/1600))
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    scale *= min(1., max(1, w-8)/max(1, tw))
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    tx = max(0, min(x1, w-tw-4))
    ty = max(th+3, min(h-baseline-2, y1-6 if y1 > th+8 else y2+th+6))
    cv2.rectangle(image, (tx, max(0, ty-th-3)), (min(w-1, tx+tw+4), min(h-1, ty+baseline+2)),
                  color if label_background is None else label_background, -1)
    cv2.putText(image, label, (tx+2, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
