"""Isolated TensorRT INT8 experiment: collect camera frames, export and compare.

Never publishes PLATE events or modifies the selected production model.
Calibration/validation images are diagnostics, not violation evidence.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np


def letterbox(image, size):
    height, width = image.shape[:2]
    scale = min(size/width, size/height)
    resized = cv2.resize(image, (round(width*scale), round(height*scale)))
    dx, dy = (size-resized.shape[1])/2, (size-resized.shape[0])/2
    return cv2.copyMakeBorder(resized, round(dy-.1), round(dy+.1), round(dx-.1), round(dx+.1),
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


def array_for(image, size):
    image = letterbox(image, size)
    return np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)/255.


def collect(args):
    from plate_ai.config import Settings
    from plate_ai.detection_config import DetectionSettings
    from plate_ai.service import CameraSyncService
    from plate_ai.rtsp import LatestFrameStream
    settings = replace(Settings.from_env(), client_id="tts_vehicle_calibration_01")
    root = Path(args.output)
    if (root/"manifest.json").exists() or any(root.glob("*/*.png")):
        raise FileExistsError("Use an empty output directory for a new calibration capture")
    root.mkdir(parents=True, exist_ok=True)
    records, seen = [], None
    with CameraSyncService(settings) as sync:
        deadline = time.monotonic()+30
        config = None
        while time.monotonic() < deadline:
            cameras = sync.snapshot().cameras
            config = next((c for c in cameras if not args.camera or c.camera.code == args.camera), None)
            if config:
                break
            time.sleep(.1)
        if config is None:
            raise RuntimeError("No eligible camera received from VMS")
        stream = LatestFrameStream(config.camera, DetectionSettings.from_env())
        stream.start()
        try:
            total = args.calibration+args.validation
            deadline = time.monotonic()+total*args.interval+45
            next_at = 0.
            while len(records) < total and time.monotonic() < deadline:
                frame = stream.latest()
                if (frame is None or frame.sequence == seen or time.monotonic() < next_at
                        or time.monotonic()-frame.received_monotonic > 1.):
                    time.sleep(.01)
                    continue
                seen, next_at = frame.sequence, time.monotonic()+args.interval
                split = "calibration" if len(records) < args.calibration else "validation"
                relative = Path(split)/f"{len(records):05d}.png"
                target = root/relative
                target.parent.mkdir(exist_ok=True)
                if not cv2.imwrite(str(target), letterbox(frame.image, args.size)):
                    raise RuntimeError("Could not save calibration image")
                records.append({"file":relative.as_posix(), "camera_id":config.camera.id,
                                "sequence":frame.sequence, "captured_at":frame.captured_at})
                if len(records) % 20 == 0:
                    print(f"Captured {len(records)}/{total}", flush=True)
            if len(records) != total:
                raise RuntimeError(f"Only captured {len(records)}/{total} fresh frames")
        finally:
            stream.stop(10.)
    (root/"manifest.json").write_text(json.dumps({"size":args.size, "records":records}, indent=2), encoding="utf-8")
    print(f"DATASET={root}", flush=True)


def build_engine(onnx_path, output, calibration, shape, *, int8, workspace_mb=128):
    import tensorrt as trt
    import torch
    logger = trt.Logger(trt.Logger.WARNING)

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.index = 0
            self.buffer = torch.empty(shape, dtype=torch.float32, device="cuda:0")
            self.cache = output.with_suffix(".calibration.cache")

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.index >= len(calibration):
                return None
            image = cv2.imread(str(calibration[self.index]))
            self.buffer.copy_(torch.from_numpy(array_for(image, shape[-1])))
            torch.cuda.synchronize()
            self.index += 1
            if self.index % 32 == 0:
                print(f"Calibration {self.index}/{len(calibration)}", flush=True)
            return [self.buffer.data_ptr()]

        def read_calibration_cache(self):
            # Recalibrate each build rather than reuse scales from another dataset.
            return None

        def write_calibration_cache(self, cache):
            self.cache.write_bytes(cache)

    builder = trt.Builder(logger)
    builder.max_threads = 1
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("ONNX parse failed: "+"; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb*1024*1024)
    config.builder_optimization_level = 0
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        if not builder.platform_has_fast_int8:
            raise RuntimeError("This GPU does not report fast TensorRT INT8 support")
        config.set_flag(trt.BuilderFlag.INT8)
        calibrator = Calibrator()
        config.int8_calibrator = calibrator
    print(f"Building {output.name}", flush=True)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    output.write_bytes(serialized)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    info = engine.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    output.with_suffix(".layers.json").write_text(info, encoding="utf-8")
    mentions = info.lower().count("int8")
    if int8 and not mentions:
        raise RuntimeError("INT8 was requested but engine inspector reports no INT8 tensors")
    return {"path":str(output), "bytes":output.stat().st_size,
            "sha256":hashlib.sha256(output.read_bytes()).hexdigest(), "int8_tensor_mentions":mentions}


class Engine:
    def __init__(self, path):
        import tensorrt as trt
        import torch
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        self.context = self.engine.create_execution_context()
        self.input = None
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input = name
            else:
                dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
                self.outputs[name] = torch.empty(tuple(self.engine.get_tensor_shape(name)), dtype=dtype, device="cuda:0")
        if self.input is None or len(self.outputs) != 1:
            raise RuntimeError("Expected one input and one decoded detection output")

    def __call__(self, tensor):
        import torch
        self.context.set_tensor_address(self.input, tensor.data_ptr())
        for name, value in self.outputs.items():
            self.context.set_tensor_address(name, value.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return next(iter(self.outputs.values()))


def postprocess(prediction, settings):
    import torch
    from torchvision.ops import batched_nms
    if isinstance(prediction, (tuple, list)):
        prediction = prediction[0]
    prediction = prediction[0]
    prediction = prediction[prediction[:, 4] > settings.confidence]
    if not len(prediction):
        return np.empty((0, 6))
    scores, classes = (prediction[:, 5:]*prediction[:, 4:5]).max(dim=1)
    valid = scores > settings.confidence
    prediction, scores, classes = prediction[valid], scores[valid], classes[valid]
    center, size = prediction[:, :2], prediction[:, 2:4]
    boxes = torch.cat((center-size/2, center+size/2), dim=1)
    keep = batched_nms(boxes.float(), scores.float(), classes, settings.iou)[:settings.max_detections]
    return torch.cat((boxes[keep], scores[keep, None], classes[keep, None]), dim=1).float().cpu().numpy()


def statistics(values):
    return {"n":len(values), "mean":round(float(np.mean(values)), 3),
            "p50":round(float(np.median(values)), 3), "p95":round(float(np.percentile(values,95)), 3)}


def experiment(args):
    import torch
    from plate_ai.detection_config import DetectionSettings
    from plate_ai.detection import VehicleDetector
    from plate_ai.tracking import iou
    settings = replace(DetectionSettings.from_env(), device="cuda:0", image_size=args.size)
    torch.set_num_threads(settings.torch_threads)
    detector = VehicleDetector(settings)
    cv2.setNumThreads(1)

    class DecodedModel(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            result = self.model(images)
            return result[0] if isinstance(result, (tuple, list)) else result

    calibration = sorted((Path(args.dataset)/"calibration").glob("*.png"))
    validation = sorted((Path(args.dataset)/"validation").glob("*.png"))
    if len(calibration) < 32 or len(validation) < 8:
        raise ValueError("Need >=32 calibration and >=8 separate validation images")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    onnx = root/(Path(settings.vehicle_weight).stem+".onnx")
    shape = (1, 3, args.size, args.size)
    sample = torch.zeros(shape, device="cuda:0")
    if args.reuse_engines:
        previous = json.loads((root/"report.json").read_text(encoding="utf-8"))
        if previous["source_sha256"] != settings.vehicle_sha256 or previous["input_shape"] != list(shape):
            raise ValueError("Existing engines do not match this checkpoint/input shape")
        artifacts = previous["artifacts"]
        for artifact in artifacts.values():
            if hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]:
                raise ValueError("Existing engine checksum mismatch")
    else:
        with torch.inference_mode():
            torch.onnx.export(DecodedModel(detector.model).eval(), sample, str(onnx), input_names=["images"], output_names=["predictions"],
                              opset_version=13, dynamo=False)
        artifacts = {}
        for precision in ("fp16", "int8"):
            target = root/(onnx.stem+"_"+precision+".engine")
            artifacts[precision] = build_engine(onnx, target, calibration, shape, int8=precision == "int8",
                                                workspace_mb=args.workspace_mb)
    # Keep BatchNorm in evaluation mode, including after ONNX exporter restores modes.
    detector.model.eval()
    images = [cv2.imread(str(path)) for path in validation]
    tensors = [torch.from_numpy(array_for(image, args.size)).to("cuda:0") for image in images]
    metrics, detections = {}, {}
    with torch.inference_mode():
        for name, runner in (("pytorch_fp32", detector.model), ("tensorrt_fp16", Engine(artifacts["fp16"]["path"])),
                             ("tensorrt_int8", Engine(artifacts["int8"]["path"]))):
            for _ in range(30):
                runner(tensors[0])
                torch.cuda.synchronize()
                time.sleep(args.cooldown)
            torch.cuda.synchronize()
            for _ in range(3):
                postprocess(runner(tensors[0]), settings)
                time.sleep(args.cooldown)
            forward, total = [], []
            for i in range(args.iterations):
                started = time.perf_counter()
                runner(tensors[i % len(tensors)])
                torch.cuda.synchronize()
                forward.append((time.perf_counter()-started)*1000)
                time.sleep(args.cooldown)
            results = []
            for image in images:
                started = time.perf_counter()
                tensor = torch.from_numpy(array_for(image, args.size)).to("cuda:0")
                output = postprocess(runner(tensor), settings)
                total.append((time.perf_counter()-started)*1000)
                results.append(output.copy())
                time.sleep(args.cooldown)
            metrics[name] = {"forward_ms":statistics(forward), "preprocess_forward_nms_ms":statistics(total)}
            detections[name] = results
            print(name+" "+json.dumps(metrics[name]), flush=True)
    agreement = {}
    for name in ("tensorrt_fp16", "tensorrt_int8"):
        matched, reference_count, candidate_count, overlaps, deltas = 0, 0, 0, [], []
        for reference, candidate in zip(detections["pytorch_fp32"], detections[name]):
            reference_count += len(reference)
            candidate_count += len(candidate)
            used = set()
            for original in reference:
                choices = [(iou(original[:4], row[:4]), j) for j,row in enumerate(candidate)
                           if j not in used and row[5] == original[5]]
                overlap, index = max(choices, default=(0., -1))
                if overlap >= .5:
                    matched += 1
                    used.add(index)
                    overlaps.append(overlap)
                    deltas.append(abs(float(original[4]-candidate[index,4])))
        agreement[name] = {"reference_detections":reference_count, "candidate_detections":candidate_count,
                           "matched_iou_0_5":matched, "precision_vs_fp32":matched/candidate_count if candidate_count else None,
                           "recall_vs_fp32":matched/reference_count if reference_count else None,
                           "matched_mean_iou":float(np.mean(overlaps)) if overlaps else None,
                           "matched_mean_confidence_delta":float(np.mean(deltas)) if deltas else None}
    report = {"source_weight":settings.vehicle_weight, "source_sha256":settings.vehicle_sha256,
              "gpu":torch.cuda.get_device_name(), "torch":torch.__version__, "input_shape":shape,
              "calibration_images":len(calibration), "held_out_images":len(validation),
              "workspace_mb":args.workspace_mb, "builder_optimization_level":0,
              "cooldown_seconds":args.cooldown,
              "artifacts":artifacts, "latency":metrics, "prediction_agreement":agreement,
              "limitations":"Single-camera unlabeled PTQ experiment; agreement with FP32 is not ground-truth mAP. Static square input; runtime selection unchanged."}
    (root/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("collect")
    capture.add_argument("--output", required=True)
    capture.add_argument("--camera")
    capture.add_argument("--calibration", type=int, default=128)
    capture.add_argument("--validation", type=int, default=32)
    capture.add_argument("--interval", type=float, default=.5)
    capture.add_argument("--size", type=int, default=640)
    build = commands.add_parser("experiment")
    build.add_argument("--dataset", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--size", type=int, default=640)
    build.add_argument("--iterations", type=int, default=60)
    build.add_argument("--workspace-mb", type=int, default=128)
    build.add_argument("--cooldown", type=float, default=.05)
    build.add_argument("--reuse-engines", action="store_true", help="Benchmark existing checksum-verified engines without rebuilding")
    args = parser.parse_args()
    if args.size < 32 or args.size % 32:
        parser.error("size must be a positive multiple of 32")
    if args.command == "collect":
        if min(args.calibration, args.validation, args.interval) <= 0:
            parser.error("counts and interval must be positive")
        collect(args)
    else:
        if args.iterations <= 0 or args.workspace_mb <= 0 or args.cooldown < 0:
            parser.error("iterations/workspace must be positive and cooldown nonnegative")
        experiment(args)


if __name__ == "__main__":
    main()
