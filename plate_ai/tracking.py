"""Vehicle association, finite line crossings and per-track recognition votes."""
from dataclasses import dataclass, field, replace
import math
from uuid import uuid4
from .detection import VehicleDetection


def iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union > 0 else 0.


def assignment(costs):
    """Rectangular minimum-cost assignment with rows <= columns."""
    if not costs:
        return []
    n, m = len(costs), len(costs[0])
    u, v, p, way = [0.]*(n+1), [0.]*(m+1), [0]*(m+1), [0]*(m+1)
    for i in range(1, n+1):
        p[0], j0 = i, 0
        minimum, used = [math.inf]*(m+1), [False]*(m+1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m+1):
                if not used[j]:
                    cost = costs[i0-1][j-1]-u[i0]-v[j]
                    if cost < minimum[j]:
                        minimum[j], way[j] = cost, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m+1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(p[j]-1, j-1) for j in range(1, m+1) if p[j]]


def foot(box):
    return ((box[0]+box[2])/2, box[3])


def intersects(start, end, line):
    def cross(a, b):
        return a[0]*b[1]-a[1]*b[0]

    a, b = line.points
    motion, segment = (end[0]-start[0], end[1]-start[1]), (b[0]-a[0], b[1]-a[1])
    denominator = cross(motion, segment)
    if abs(denominator) < 1e-12:
        return False
    offset = (a[0]-start[0], a[1]-start[1])
    t, s = cross(offset, segment)/denominator, cross(offset, motion)/denominator
    return 0 <= t <= 1 and 0 <= s <= 1


@dataclass
class Track:
    id: int
    object_key: str
    box: tuple
    vehicle: VehicleDetection
    last_seen: float
    hits: int = 1
    velocity: tuple = (0., 0.)
    sides: dict = field(default_factory=dict)
    votes: dict = field(default_factory=dict)
    direction: str | None = None
    crossed_epoch: float | None = None
    crossed_monotonic: float | None = None
    line_id: str | None = None
    event: dict | None = None
    retry_at: float = 0.
    sent: bool = False
    plate_relative: tuple | None = None
    plate_seen_at: float = 0.
    plate_text: str = ""
    plate_confidence: float = 0.
    text_confidence: float = 0.

    def remember(self, read, received_monotonic=None):
        if received_monotonic is not None:
            plate = read.license_plate
            self.plate_relative = None
            self.plate_text = plate.text
            self.plate_confidence = plate.detection_confidence
            self.text_confidence = plate.confidence
            self.plate_seen_at = received_monotonic
            x1, y1, x2, y2 = read.vehicle.bbox
            if plate.bbox is not None and x2 > x1 and y2 > y1:
                px1, py1, px2, py2 = plate.bbox
                self.plate_relative = ((px1-x1)/(x2-x1), (py1-y1)/(y2-y1),
                                       (px2-x1)/(x2-x1), (py2-y1)/(y2-y1))
        # A frozen passage event stays unchanged, but a later RED_LIGHT event
        # for this track must still be able to consume asynchronous OCR.
        if not read.license_plate.text:
            return
        text = read.license_plate.text
        count, total, previous = self.votes.get(text, (0, 0., None))
        best = read if previous is None or read.license_plate.confidence >= previous.license_plate.confidence else previous
        other = previous if best is read else read
        if other is not None:
            best = replace(best, manufacturer=best.manufacturer or other.manufacturer, color=best.color or other.color,
                           license_plate=replace(best.license_plate,
                                                 plate_color=best.license_plate.plate_color or other.license_plate.plate_color))
        self.votes[text] = (count+1, total+read.license_plate.confidence, best)
        if len(self.votes) > 16:
            del self.votes[min(self.votes, key=lambda k: self.votes[k][:2])]

    def best_read(self):
        if not self.votes:
            return None
        count, score, read = max(self.votes.values(), key=lambda value: value[:2])
        return replace(read, license_plate=replace(read.license_plate, confidence=score/count))


class VehicleTracker:
    def __init__(self, camera_code, settings):
        self.settings = settings
        self.tracks = {}
        self._prefix = f"{camera_code}_{uuid4().hex}"
        self._next_id = 1

    def expire(self, now):
        self.tracks = {k: t for k, t in self.tracks.items() if now-t.last_seen <= self.settings.track_ttl}

    def update(self, metadata):
        now = metadata.received_monotonic
        self.expire(now)
        observations = metadata.detections[:self.settings.max_tracks]
        boxes = [tuple(v/(metadata.width if i % 2 == 0 else metadata.height) for i, v in enumerate(o.bbox))
                 for o in observations]
        old, costs = list(self.tracks.values()), []
        for track in old:
            dt = max(.001, now-track.last_seen)
            predicted = tuple(v+track.velocity[i % 2]*dt for i, v in enumerate(track.box))
            row = []
            for observation, box in zip(observations, boxes):
                distance, overlap = math.dist(foot(predicted), foot(box)), iou(predicted, box)
                if observation.vehicle_type != track.vehicle.vehicle_type or (distance > .15 and overlap < .1):
                    row.append(1e6)
                else:
                    row.append(.65*(1-overlap)+.35*min(1., distance/.15))
            costs.append(row+[1.1]*len(old))
        matched, visible = set(), {}
        for row, col in assignment(costs):
            if col >= len(observations) or costs[row][col] >= 1.1:
                continue
            track, observation, box = old[row], observations[col], boxes[col]
            previous, current = foot(track.box), foot(box)
            dt = max(.001, now-track.last_seen)
            track.velocity = tuple(max(-1., min(1., (current[i]-previous[i])/dt)) for i in range(2))
            track.box, track.vehicle, track.last_seen = box, observation, now
            track.hits += 1
            visible[col] = track
            matched.add(col)
        for index, (observation, box) in enumerate(zip(observations, boxes)):
            if index in matched or len(self.tracks) >= self.settings.max_tracks:
                continue
            track = Track(self._next_id, f"{self._prefix}_{self._next_id}", box, observation, now)
            self._next_id += 1
            self.tracks[track.id] = track
            visible[index] = track
        return visible
