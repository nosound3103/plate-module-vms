"""Validated camera and zone configuration, independent of MQTT and inference."""
from dataclasses import dataclass, field
import math
from uuid import UUID


def modules(value) -> set[str]:
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {str(v).strip().upper() for v in value}


@dataclass(frozen=True)
class Camera:
    id: str
    code: str
    source: str = field(repr=False)

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict):
            raise ValueError("Camera must be an object")
        if "PLATE" not in modules(data.get("ai_modules")) or data.get("is_active") is False:
            return None
        if str(data.get("status", "ONLINE")).upper() not in {"ONLINE", "ACTIVE"}:
            return None
        identity = str(UUID(str(data.get("id", ""))))
        code = data.get("code")
        streams = data.get("restream_urls") or {}
        source = (streams.get("PLATE") if isinstance(streams, dict) else None) or data.get("rtsp_url")
        if (not isinstance(code, str) or not code or any(c in code for c in "/+#")
                or any(c.isspace() or ord(c) < 32 for c in code)):
            raise ValueError("Camera code is missing or cannot be used in MQTT topics")
        if not isinstance(source, str) or not source.startswith(("rtsp://", "rtsps://")):
            raise ValueError("Camera PLATE stream must be an RTSP URL")
        return cls(identity, code, source)


@dataclass(frozen=True)
class Zone:
    id: str
    points: tuple[tuple[float, float], ...]

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict):
            raise ValueError("AI zone must be an object")
        identity = str(data.get("id") or data.get("code") or "")
        if not identity:
            raise ValueError("AI zone needs an id or code")
        if data.get("is_active") is False or "PLATE" not in modules(data.get("ai_modules", data.get("ai_module"))):
            return None
        points = data.get("points")
        if not isinstance(points, list) or len(points) < 3:
            raise ValueError("PLATE monitoring zone needs at least three normalized points")
        values = tuple((float(p["x"]), float(p["y"])) for p in points)
        if any(not math.isfinite(v) or not 0 <= v <= 1 for point in values for v in point):
            raise ValueError("Zone coordinates must be finite and between 0 and 1")
        area = abs(sum(values[i][0] * values[(i+1) % len(values)][1] -
                       values[(i+1) % len(values)][0] * values[i][1] for i in range(len(values))))
        if area < 1e-8:
            raise ValueError("Zone polygon has no area")
        return cls(identity, values)


@dataclass(frozen=True)
class CrossingLine:
    id: str
    points: tuple[tuple[float, float], tuple[float, float]]
    in_side: int = 1

    def distance(self, point):
        (ax, ay), (bx, by) = self.points
        return ((bx-ax) * (point[1]-ay) - (by-ay) * (point[0]-ax)) / math.hypot(bx-ax, by-ay)

    @classmethod
    def parse(cls, data, *, module_scoped=False):
        if not isinstance(data, dict):
            raise ValueError("Crossing line must be an object")
        identity = str(data.get("id") or data.get("code") or "")
        if not identity:
            raise ValueError("Crossing line needs an id or code")
        if data.get("is_active") is False:
            return None
        configured_modules = data.get("ai_modules", data.get("ai_module"))
        if configured_modules is not None and "PLATE" not in modules(configured_modules):
            return None
        if configured_modules is None and not module_scoped:
            raise ValueError("Crossing line must declare PLATE")
        geometry = data.get("line") or data
        raw = geometry.get("points")
        if not isinstance(raw, list) or len(raw) != 2:
            raise ValueError("A crossing line needs two endpoints")
        points = tuple((float(p["x"]), float(p["y"])) for p in raw)
        if any(not math.isfinite(v) or not 0 <= v <= 1 for p in points for v in p):
            raise ValueError("Line endpoints must be normalized")
        (ax, ay), (bx, by) = points
        if math.hypot(bx-ax, by-ay) < 1e-6:
            raise ValueError("Crossing line has no length")
        reference = data.get("direction_point", geometry.get("direction_point"))
        vector = data.get("direction_vector", geometry.get("direction_vector"))
        if reference is not None:
            dx, dy = float(reference["x"])-ax, float(reference["y"])-ay
        elif vector is not None:
            dx, dy = map(float, vector)
        else:
            # Left-hand side of the directed endpoint pair is IN by default.
            dx, dy = -(by-ay), bx-ax
        cross = (bx-ax)*dy - (by-ay)*dx
        if not math.isfinite(cross) or abs(cross) < 1e-9:
            raise ValueError("IN reference must point to one side of the line")
        return cls(identity, points, 1 if cross > 0 else -1)


class ZoneStore:
    """Full snapshots replace state; a zone object updates only that zone."""
    def __init__(self):
        self._zones = {}
        self._topic_ids = {}
        self._line_ids = set()

    @property
    def zones(self):
        return tuple(self._zones.values())

    def apply(self, topic_suffix: str, payload):
        line_topic = topic_suffix == "PLATE/lines" or topic_suffix.startswith("PLATE/lines/")
        if payload is None:
            if not topic_suffix:
                self._zones.clear()
                self._topic_ids.clear()
                self._line_ids.clear()
            elif topic_suffix == "PLATE/lines":
                self._zones = {k: z for k, z in self._zones.items() if not isinstance(z, CrossingLine)}
                self._topic_ids.pop(topic_suffix, None)
                self._line_ids.clear()
            else:
                identities = self._topic_ids.pop(topic_suffix, {topic_suffix.split("/")[-1]})
                for identity in identities:
                    self._zones.pop(identity, None)
                    self._line_ids.discard(identity)
            return
        if isinstance(payload, list) and line_topic:
            payload = {"lines": payload}
        if not isinstance(payload, dict):
            raise ValueError("Zone message must be an object")
        snapshot = "ai_zones" in payload
        line_snapshot = "lines" in payload
        entries = payload["ai_zones"] if snapshot else payload["lines"] if line_snapshot else [payload]
        if not isinstance(entries, list):
            raise ValueError("ai_zones must be an array")
        parsed = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("AI zone entries must be objects")
            is_line = line_topic or line_snapshot or (isinstance(entry, dict) and (
                entry.get("shape_type") == "line_crossing" or "line" in entry))
            zone = CrossingLine.parse(entry, module_scoped=line_topic) if is_line else Zone.parse(entry)
            identity = str(entry.get("id") or entry.get("code"))
            if identity in parsed:
                raise ValueError("Duplicate AI zone identity")
            parsed[identity] = zone
        if snapshot:
            # Retained polygon snapshots can arrive after the dedicated lines
            # topic. Preserve that authoritative line configuration.
            dedicated = {k: z for k, z in self._zones.items() if k in self._line_ids}
            self._zones = {key: value for key, value in parsed.items() if value is not None}
            self._zones.update(dedicated)
            self._topic_ids = {topic: identities for topic, identities in self._topic_ids.items()
                               if topic == "PLATE/lines" or topic.startswith("PLATE/lines/")}
        else:
            if line_snapshot:
                self._zones = {key: value for key, value in self._zones.items() if not isinstance(value, CrossingLine)}
                self._line_ids.clear()
            for key, value in parsed.items():
                if value is None:
                    self._zones.pop(key, None)
                    self._line_ids.discard(key)
                else:
                    self._zones[key] = value
                    if line_topic:
                        self._line_ids.add(key)
        if topic_suffix:
            self._topic_ids[topic_suffix] = set(parsed)
