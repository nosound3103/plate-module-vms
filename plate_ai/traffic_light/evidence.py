"""Unmodified crossing image plus separately marked context and audit metadata."""
from dataclasses import dataclass
from concurrent.futures import Future
import hashlib
import json
import threading
from pathlib import Path
import cv2
import numpy as np
from plate_ai.evidence.storage import save_original
from plate_ai.evidence.drawing import draw_event_box
from plate_ai.traffic_light.detector import draw_traffic_light, contour


@dataclass(frozen=True)
class EvidenceFiles:
    original: str
    annotated: str
    manifest: str
    frame_sha256: str
    upload: object = None


def write_manifest(files, metadata):
    path = Path(files.manifest)
    temp = path.with_suffix(".json.tmp")
    data = {**metadata, "original_path": files.original, "annotated_path": files.annotated,
            "frame_sha256": files.frame_sha256}
    temp.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    temp.replace(path)


def prepare_evidence(camera, violation, directory):
    """Save the untouched original and a separate marked viewing copy."""
    frame = violation.frame
    original = save_original(frame, directory)
    root = Path(original)
    digest = hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
    preview = frame.copy()
    h, w = frame.shape[:2]
    for name, points, color in (("STOP-ZONE", violation.stop_points, (255, 180, 0)),
                                ("VIOLATION-ZONE", violation.violation_points, (255, 0, 255))):
        if points:
            polygon = np.rint(contour(points)*[w-1, h-1]).astype(np.int32)
            cv2.polylines(preview, [polygon], True, color, 2)
    draw_vehicle_highlight(preview, violation)
    cv2.putText(preview, f"EVIDENCE CONTEXT track={violation.track_id} frame={violation.frame_sequence}",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
    if violation.light_points and violation.signal:
        draw_traffic_light(preview, violation.light_points, violation.signal)
    annotated = str(root.with_suffix(".annotated.jpg"))
    if not cv2.imwrite(annotated, preview):
        raise OSError("Could not save annotated evidence context")
    files = EvidenceFiles(original, annotated, str(root.with_suffix(".json")), digest)
    write_manifest(files, {"camera_id": camera.id, "entity_id": f"{violation.track_key}:RED_LIGHT",
                           "crossing_timestamp": violation.timestamp,
                           "frame_timestamp": violation.frame_timestamp,
                           "frame_sequence": violation.frame_sequence,
                           "vehicle_bbox": violation.vehicle_box,
                           "traffic_light_state": violation.signal.state if violation.signal else None,
                           "plate_status": "PENDING_VERIFIED_OCR"})
    return files


def draw_vehicle_highlight(image, violation, plate_text=None):
    """Mark only the frozen crossing box; never use the live track's position."""
    label = f"RED_LIGHT ID:{violation.track_id}"
    if plate_text:
        label += f" {plate_text}"
    draw_event_box(image, violation.vehicle_box, label, (0, 0, 255),
                   label_background=(0, 0, 180), sort_corners=True)


def upload_original(camera, files, snapshots, object_key):
    """Load the lossless original off the controller, only after OCR is verified."""
    image = cv2.imread(files.original)
    if image is None or hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest() != files.frame_sha256:
        raise ValueError("Original evidence image is missing or changed")
    return snapshots.submit(camera, image, object_key=object_key)


def upload_evidence(camera, files, snapshots, object_key, violation, plate_text):
    """Upload both assets; retain the original key beside the viewing-copy key."""
    original = upload_original(camera, files, snapshots, object_key+":original")
    image = cv2.imread(files.annotated)
    if image is None:
        raise ValueError("Annotated evidence image is missing")
    draw_vehicle_highlight(image, violation, plate_text)
    if not cv2.imwrite(files.annotated, image):
        raise OSError("Could not label annotated evidence")
    annotated = snapshots.submit(camera, image, object_key=object_key+":annotated")
    combined = Future()
    lock = threading.Lock()

    def complete(_):
        with lock:
            if original.done() and annotated.done() and not combined.done():
                try:
                    combined.set_result({"original": original.result(), "annotated": annotated.result()})
                except Exception as error:
                    combined.set_exception(error)

    original.add_done_callback(complete)
    annotated.add_done_callback(complete)
    return combined
