"""Resolve checkpoint class paths without changing global Python imports."""
import pickle
from pathlib import Path

load, loads = pickle.load, pickle.loads
dump, dumps = pickle.dump, pickle.dumps
Pickler = pickle.Pickler


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pathlib" and name in {"WindowsPath", "PosixPath"}:
            return Path
        if module in {"models.common", "models.yolo", "vehicle.yolo.models.common", "vehicle.yolo.models.yolo"}:
            module = "plate_ai.models.yolov5"
        return super().find_class(module, name)
