"""Validated camera and zone configuration, independent of MQTT and inference."""
from dataclasses import dataclass, field
import math
from uuid import UUID

TRAFFIC_ZONE_NAMES = {"TRAFFICLIGHT", "STOP-ZONE", "VIOLATION-ZONE"}


@dataclass(frozen=True)
class TrafficZone:
    """Named VMS polygon. Invalid updates replace old geometry (fail closed)."""
    id: str
    name: str
    points: tuple = ()

    @classmethod
    def parse(cls, data):
        name = str(data.get("name", "")).strip().upper()
        identity = str(data.get("id") or data.get("code") or name)
        if data.get("is_active") is False:
            return None
        configured = data.get("ai_modules", data.get("ai_module"))
        if configured is not None and "PLATE" not in modules(configured):
            return None
        try:
            raw = data.get("zone", data.get("points"))
            if isinstance(raw, dict):
                raw = raw.get("points")
            points = tuple((float(p["x"]), float(p["y"])) if isinstance(p, dict)
                           else (float(p[0]), float(p[1])) for p in raw)
            if len(points) > 3 and points[0] == points[-1]:
                points = points[:-1]
            if len(points) < 3 or len(set(points)) != len(points):
                raise ValueError("Invalid vertices")
            if any(not math.isfinite(v) or not 0 <= v <= 1 for p in points for v in p):
                raise ValueError("Coordinates must be normalized to source frame")
            area = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(points, points[1:]+points[:1]))
            if abs(area) < 1e-8:
                raise ValueError("No polygon area")
            def orient(a, b, c):
                return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
            edges = list(zip(points, points[1:]+points[:1]))
            for i, (a, b) in enumerate(edges):
                for j, (c, d) in enumerate(edges):
                    if j <= i or j == i+1 or (i == 0 and j == len(edges)-1):
                        continue
                    if (orient(a, b, c)*orient(a, b, d) <= 0
                            and orient(c, d, a)*orient(c, d, b) <= 0
                            and max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0]))
                            and max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1]))):
                        raise ValueError("Self intersecting polygon")
            return cls(identity, name, points)
        except (TypeError, ValueError, KeyError, IndexError, OverflowError):
            return cls(identity, name)


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
    def parse(cls, data, *, module_scoped=False):
        if not isinstance(data, dict):
            raise ValueError("AI zone must be an object")
        identity = str(data.get("id") or data.get("code") or "")
        if not identity:
            raise ValueError("AI zone needs an id or code")
        if data.get("is_active") is False:
            return None
        configured_modules = data.get("ai_modules", data.get("ai_module"))
        if configured_modules is not None and "PLATE" not in modules(configured_modules):
            return None
        if configured_modules is None and not module_scoped:
            raise ValueError("Monitoring zone must declare PLATE")
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
    """Apply atomic PLATE snapshots and individual retained-topic updates."""
    def __init__(self):
        self._zones = {}
        self._topic_ids = {}
        self._line_ids = set()

    @property
    def zones(self):
        return tuple(self._zones[key] for key in sorted(self._zones))

    def _remove(self, keys):
        keys = set(keys)
        for key in keys:
            self._zones.pop(key, None)
        self._line_ids.difference_update(keys)
        for topic, identities in tuple(self._topic_ids.items()):
            remaining = identities - keys
            if remaining:
                self._topic_ids[topic] = remaining
            else:
                del self._topic_ids[topic]

    def _clear(self, suffix):
        if suffix in {"", "PLATE"}:
            self._remove(self._zones)
        elif suffix == "PLATE/lines":
            self._remove(key for key, zone in self._zones.items() if isinstance(zone, CrossingLine))
        else:
            kind = "line" if suffix.startswith("PLATE/lines/") else "polygon"
            identities = self._topic_ids.get(suffix, {(kind, suffix.split("/")[-1])})
            self._remove(identities)

    def apply(self, topic_suffix: str, payload):
        if topic_suffix and topic_suffix.split("/")[0] != "PLATE":
            return
        module_scoped = bool(topic_suffix)
        line_topic = topic_suffix == "PLATE/lines" or topic_suffix.startswith("PLATE/lines/")
        collection_topic = topic_suffix in {"", "PLATE", "PLATE/lines"}
        if payload is None or payload == {}:
            self._clear(topic_suffix)
            return
        if isinstance(payload, list):
            if not collection_topic:
                raise ValueError("Individual geometry topic must contain an object")
            payload = {"lines" if line_topic else "ai_zones": payload}
        if not isinstance(payload, dict):
            raise ValueError("Zone message must be an object")
        snapshot = "ai_zones" in payload
        line_snapshot = "lines" in payload
        if (snapshot or line_snapshot) and not collection_topic:
            raise ValueError("Individual geometry topic cannot replace a collection")
        if line_topic and snapshot:
            raise ValueError("Line topic cannot contain a polygon snapshot")
        groups = []
        if snapshot:
            groups.append((payload["ai_zones"], False))
        if line_snapshot:
            groups.append((payload["lines"], True))
        if not groups:
            groups.append(([payload], line_topic))
        parsed = {}
        for entries, force_line in groups:
            if not isinstance(entries, list):
                raise ValueError("Geometry snapshot must be an array")
            for entry in entries:
                if not isinstance(entry, dict):
                    raise ValueError("AI zone entries must be objects")
                if not collection_topic and not entry.get("id") and not entry.get("code"):
                    entry = {**entry, "id": topic_suffix.split("/")[-1]}
                if not collection_topic and str(entry.get("id") or entry.get("code")) != topic_suffix.split("/")[-1]:
                    raise ValueError("Geometry identity must match its topic")
                named = str(entry.get("name", "")).strip().upper() in TRAFFIC_ZONE_NAMES
                if named and not entry.get("id") and not entry.get("code"):
                    entry = {**entry, "id": str(entry["name"]).strip().upper()}
                is_line = not named and (force_line or entry.get("shape_type") == "line_crossing" or "line" in entry)
                zone = (TrafficZone.parse(entry) if named else
                        CrossingLine.parse(entry, module_scoped=module_scoped) if is_line
                        else Zone.parse(entry, module_scoped=module_scoped))
                identity = str(entry.get("id") or entry.get("code"))
                key = ("line" if is_line else "polygon", identity)
                if key in parsed:
                    if named:
                        parsed[key] = TrafficZone(identity, str(entry["name"]).strip().upper())
                        continue
                    raise ValueError("Duplicate AI geometry identity")
                parsed[key] = zone
        # Parse the whole update before modifying live state. Polygon and line
        # IDs belong to separate namespaces.
        if snapshot:
            self._remove(key for key in self._zones if key not in self._line_ids)
        if line_snapshot:
            self._remove(key for key, zone in self._zones.items() if isinstance(zone, CrossingLine))
        if not collection_topic:
            self._remove(self._topic_ids.get(topic_suffix, set()) - set(parsed))
        owned = set()
        for key, zone in parsed.items():
            if snapshot and not line_snapshot and key in self._line_ids:
                continue
            self._remove({key})
            if zone is not None:
                self._zones[key] = zone
                owned.add(key)
                if line_topic or line_snapshot:
                    self._line_ids.add(key)
        if owned:
            self._topic_ids[topic_suffix] = owned
        else:
            self._topic_ids.pop(topic_suffix, None)
