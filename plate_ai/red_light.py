"""Masked HSV observations and causal per-track STOP -> VIOLATION transitions."""
from dataclasses import dataclass, field
import math
import logging
import cv2
import numpy as np
from .tracking import foot

COLORS = {"RED": (0, 0, 255), "GREEN": (0, 255, 0),
          "YELLOW": (0, 255, 255), "UNKNOWN": (128, 128, 128)}
LABELS = {"RED": "R", "GREEN": "G", "YELLOW": "Y", "UNKNOWN": "U"}
LOGGER = logging.getLogger(__name__)


def contour(points):
    return np.asarray(points, dtype=np.float32)


@dataclass(frozen=True)
class LightObservation:
    state: str
    confidence: float
    scores: dict
    timestamp: float


class TrafficLightStateDetector:
    def __init__(self, settings):
        self.settings = settings

    def detect(self, frame, points, timestamp):
        s = self.settings
        scores = {name: 0. for name in ("RED", "YELLOW", "GREEN")}
        unknown = lambda: LightObservation("UNKNOWN", 0., scores, timestamp)
        if len(points) < 3:
            return unknown()
        h, w = frame.shape[:2]
        polygon = np.rint(contour(points)*[w-1, h-1]).astype(np.int32)
        x, y, bw, bh = cv2.boundingRect(polygon)
        roi = frame[y:y+bh, x:x+bw]
        mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon-[x, y]], 255)
        valid = cv2.countNonZero(mask)
        if valid < s.min_roi_pixels:
            return unknown()
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        confidences = {}
        for name, ranges in (("RED", s.red_hues), ("YELLOW", s.yellow_hues), ("GREEN", s.green_hues)):
            color = np.zeros_like(mask)
            for low, high in ranges:
                color |= cv2.inRange(hsv, (low, s.saturation_min, s.value_min), (high, 255, 255))
            color &= mask
            # LED lamps/countdowns contain dark gaps between bright diodes.
            # Close only for component grouping; all signal energy, brightness
            # and minimum-area checks still use original HSV-matched pixels.
            grouped = cv2.morphologyEx(color, cv2.MORPH_CLOSE,
                                      np.ones((s.component_close_size, s.component_close_size), np.uint8))
            grouped &= mask
            count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
            best, confidence = 0., 0.
            for i in range(1, count):
                _, _, cw, ch, grouped_area = stats[i]
                source = (labels == i) & (color > 0)
                area = int(np.count_nonzero(source))
                ratio, fill = area/valid, grouped_area/(cw*ch)
                support = area/grouped_area
                if (area < s.min_component_pixels or ratio < s.min_component_ratio
                        or fill < s.min_component_fill or support < s.min_component_support):
                    continue
                brightness = float(hsv[:, :, 2][source].mean())/255
                compactness = math.sqrt(fill)
                score = ratio*compactness*brightness
                if score > best:
                    best = score
                    # Signal quality estimate, not a calibrated probability.
                    confidence = min(1., ratio/s.min_component_ratio)*compactness*brightness
            scores[name], confidences[name] = best, confidence
        ordered = sorted(scores, key=scores.get, reverse=True)
        winner, runner = ordered[:2]
        if (scores[winner] <= 0 or confidences[winner] < s.min_confidence
                or scores[runner] >= scores[winner]*s.ambiguity_ratio):
            return unknown()
        return LightObservation(winner, confidences[winner], scores, timestamp)


class TrafficLightStateTracker:
    """Strict crossing signal plus bounded display hysteresis, both causal.

    ``update`` returns the conservative signal used for violations. ``display``
    may bridge a brief dropout, but is never evidence that the light stayed red.
    Display timestamps identify actual supporting observations, not held frames.
    """
    def __init__(self, settings):
        self.settings = settings
        self.last = None
        self.candidate, self.hits = None, 0
        self.display = None
        self.display_candidate, self.display_hits = None, 0

    def _update_display(self, sample, gap):
        s = self.settings
        if gap > s.signal_max_age:
            self.display = None
            self.display_candidate, self.display_hits = None, 0
        if sample.state == "UNKNOWN":
            self.display_candidate, self.display_hits = None, 0
        else:
            if sample.state != self.display_candidate:
                self.display_candidate, self.display_hits = sample.state, 0
            self.display_hits += 1
        if (self.display is not None and sample.state == self.display.state
                and self.display.state != "UNKNOWN"):
            self.display = sample
        elif sample.state != "UNKNOWN" and self.display_hits >= s.preview_confirm_frames:
            self.display = sample
        elif (self.display is None
              or sample.timestamp-self.display.timestamp > s.preview_hold_seconds):
            self.display = LightObservation("UNKNOWN", 0., sample.scores, sample.timestamp)

    def update(self, sample):
        gap = sample.timestamp-self.last.timestamp if self.last else math.inf
        if gap <= 0:
            return LightObservation("UNKNOWN", 0., sample.scores, sample.timestamp)
        self._update_display(sample, gap)
        if gap > self.settings.signal_max_age or sample.state != self.candidate:
            self.candidate, self.hits = sample.state, 0
        self.hits += 1
        state = sample.state if self.hits >= self.settings.signal_confirm_frames else "UNKNOWN"
        self.last = LightObservation(state, sample.confidence if state != "UNKNOWN" else 0.,
                                     sample.scores, sample.timestamp)
        return self.last


@dataclass
class _VehicleState:
    point: tuple
    timestamp: float
    signal: LightObservation
    stop_hits: int = 0
    armed: bool = False
    completed: bool = False
    candidate: object = None
    violation_hits: int = 0
    approach_point: tuple | None = None
    identity_epoch: int = 0


@dataclass(frozen=True)
class Violation:
    track: object
    timestamp: float
    confidence: float
    frame: object = field(repr=False)
    frame_timestamp: float | None = None
    frame_sequence: int = 0
    signal: object = None
    light_points: tuple = ()
    stop_points: tuple = ()
    violation_points: tuple = ()
    vehicle_box: tuple | None = None
    vehicle: object = None
    track_key: str | None = None
    track_id: int | None = None
    identity_epoch: int | None = None
    read_at_crossing: object = None
    direction: str | None = None

    def __post_init__(self):
        if self.frame_timestamp is None:
            object.__setattr__(self, "frame_timestamp", self.timestamp)
        if self.vehicle_box is not None:
            return  # dataclasses.replace must preserve the crossing snapshot.
        for name, value in (("vehicle_box", tuple(self.track.box)), ("vehicle", self.track.vehicle),
                            ("track_key", self.track.object_key), ("track_id", self.track.id),
                            ("identity_epoch", self.track.identity_epoch), ("direction", self.track.direction or "IN")):
            object.__setattr__(self, name, value)
        read = self.track.best_read()
        proof = self.track.read_evidence.get(read.license_plate.text) if read else None
        if proof and proof.identity_epoch != self.track.identity_epoch:
            proof = None
        object.__setattr__(self, "read_at_crossing", proof)


def intersections(start, end, points):
    """Motion parameters at finite polygon boundary intersections."""
    motion = np.asarray(end)-start
    cross = lambda a, b: a[0]*b[1]-a[1]*b[0]
    for a, b in zip(points, points[1:]+points[:1]):
        edge = np.asarray(b)-a
        denominator = cross(motion, edge)
        if abs(denominator) < 1e-12:
            continue
        offset = np.asarray(a)-start
        t, u = cross(offset, edge)/denominator, cross(offset, motion)/denominator
        if 0 <= t <= 1 and 0 <= u <= 1:
            yield float(t)


class RedLightViolationDetector:
    def __init__(self, stop, violation, settings, *, light_points=()):
        self.stop, self.violation, self.settings = stop, violation, settings
        self.light_points = light_points
        self.states = {}

    def update(self, tracks, signal, timestamp, frame, *, frame_sequence=0):
        events, s = [], self.settings
        evidence_frame = None
        active = {t.object_key for t in tracks}
        # Missing even one observation invalidates motion continuity. Tracker ID
        # reuse alone is not proof that a crossing occurred during a detection gap.
        self.states = {k: v for k, v in self.states.items() if k in active}
        for track in tracks:
            point = foot(track.box)
            stop = cv2.pointPolygonTest(contour(self.stop), point, True)
            violation = cv2.pointPolygonTest(contour(self.violation), point, True)
            old = self.states.get(track.object_key)
            if (old is None or not 0 < timestamp-old.timestamp <= s.max_crossing_seconds
                    or old.identity_epoch != track.identity_epoch or not track.association_reliable):
                if old is not None and (old.armed or old.candidate is not None) and not old.completed:
                    reason = ("IDENTITY_UNCERTAIN" if old.identity_epoch != track.identity_epoch
                              or not track.association_reliable else "OBSERVATION_GAP")
                    LOGGER.info("RED_LIGHT object_key=%s track=%s result=TRANSITION_RESET reason=%s",
                                track.object_key, track.id, reason)
                old = _VehicleState(point, timestamp, signal, completed=violation >= 0,
                                    identity_epoch=track.identity_epoch)
                self.states[track.object_key] = old
            if not old.completed:
                if stop > s.boundary_margin and violation < 0:
                    old.stop_hits += 1
                    old.armed |= old.stop_hits >= s.zone_confirm_frames
                    if old.armed and old.approach_point is None:
                        old.approach_point = point
                else:
                    old.stop_hits = 0
                if old.candidate is not None:
                    if violation > s.boundary_margin:
                        old.violation_hits += 1
                        travel = (cv2.pointPolygonTest(contour(self.violation), point, True)
                                  - cv2.pointPolygonTest(contour(self.violation), old.approach_point, True))
                        if old.violation_hits >= s.zone_confirm_frames and travel >= s.min_crossing_displacement:
                            events.append(old.candidate)
                            old.completed, old.candidate = True, None
                    elif violation < 0:
                        old.completed, old.candidate = True, None
                elif old.armed and violation > 0:
                    old.completed = True  # A non-red crossing can never become a later RED event.
                    reason = "NO_SHARED_BOUNDARY_CROSSING"
                    entries = list(intersections(old.point, point, self.violation))
                    exits = list(intersections(old.point, point, self.stop))
                    if entries and exits:
                        entry, exit_ = min(entries), max(exits)
                        gap = abs(entry-exit_)*math.dist(old.point, point)
                        reliable = (cv2.pointPolygonTest(contour(self.stop), old.point, False) >= 0
                                    and cv2.pointPolygonTest(contour(self.violation), old.point, False) <= 0
                                    and gap <= s.max_zone_gap
                                    and (violation-cv2.pointPolygonTest(contour(self.violation), old.approach_point, True)
                                         >= s.min_crossing_displacement))
                        reason = ("GEOMETRY_OR_DISPLACEMENT" if not reliable else
                                  "SIGNAL_NOT_RED" if old.signal.state != "RED" or signal.state != "RED" else
                                  "STALE_SIGNAL" if timestamp-old.signal.timestamp > s.signal_max_age else
                                  "MISSING_FRAME")
                        if (reliable and old.signal.state == signal.state == "RED"
                                and timestamp-old.signal.timestamp <= s.signal_max_age and frame is not None):
                            crossing = old.timestamp+entry*(timestamp-old.timestamp)
                            if evidence_frame is None:
                                evidence_frame = frame.copy()
                            old.candidate = Violation(track, crossing,
                                                      min(old.signal.confidence, signal.confidence,
                                                          track.vehicle.confidence), evidence_frame, timestamp,
                                                      frame_sequence, signal, self.light_points,
                                                      self.stop, self.violation)
                            old.completed = False
                            old.violation_hits = int(violation > s.boundary_margin)
                    if old.candidate is None:
                        LOGGER.info("RED_LIGHT object_key=%s track=%s result=CROSSING_REJECTED reason=%s "
                                    "previous_signal=%s signal=%s timestamp=%.3f", track.object_key, track.id,
                                    reason, old.signal.state, signal.state, timestamp)
            old.point, old.timestamp, old.signal = point, timestamp, signal
        return events


def draw_traffic_light(frame, points, sample, *, debug=False):
    """Draw on the caller's preview copy, after HSV analysis."""
    h, w = frame.shape[:2]
    polygon = np.rint(contour(points)*[w-1, h-1]).astype(np.int32)
    x, y, bw, bh = cv2.boundingRect(polygon)
    color = COLORS[sample.state]
    cv2.rectangle(frame, (x, y), (min(w-1, x+bw-1), min(h-1, y+bh-1)), color, 2)
    label = LABELS[sample.state]
    if debug:
        label += " " + " ".join(f"{k}={v:.3f}" for k, v in sample.scores.items())
    scale, thickness = .5, 1
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    scale *= min(1., max(1, w-4)/max(1, tw), max(1, h-4)/(th+baseline))
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    tx = max(0, min(x, w-tw-1))
    ty = min(h-baseline-1, max(th+1, y-5))
    cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    return frame
