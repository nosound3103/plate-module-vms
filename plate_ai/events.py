"""Serialize recognition into the VMS PLATE event contract."""
from datetime import datetime, timedelta, timezone
import re

PLATE_TOPIC = "smart_vms/ai_events/PLATE"


def build_plate_event(camera, track):
    recognized = track.best_read()
    plate = recognized.license_plate if recognized is not None else None
    text = re.sub(r"[^A-Z0-9]", "", plate.text.upper()) if plate else ""
    confidence = plate.confidence if text else track.vehicle.confidence
    moment = datetime.fromtimestamp(track.crossed_epoch, timezone(timedelta(hours=7)))
    return {
        "ai_modules": "PLATE",
        "camera_id": camera.id,
        "event_time": moment.isoformat(timespec="milliseconds"),
        "entity_type": "VEHICLE",
        "entity_id": track.object_key,
        "payload": {
            "object_key": track.object_key,
            "direction": track.direction,
            "confidence": round(max(0., min(1., confidence)), 4),
            "vehicle": {
                "license_plate": {"text": text, "status": "DETECTED" if text else "NOT_DETECTED",
                                  "plate_color": plate.plate_color if text else None},
                "vehicle_type": track.vehicle.vehicle_type,
                "car_type": None,
                "manufacturer": recognized.manufacturer if recognized else None,
                "color": recognized.color if recognized else None,
                "total_number_of_seats": None,
                "chassis_number": None,
                "owner_name": None,
            },
        },
        "snapshot_url": None,
        "snapshot_base64": None,
    }
