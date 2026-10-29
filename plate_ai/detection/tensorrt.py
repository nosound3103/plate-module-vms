"""Verified static TensorRT engine adapter for the shared vehicle detector."""
import hashlib


class TensorRTEngine:
    def __init__(self, path, sha256, device, size, class_count):
        import numpy as np
        import torch
        import tensorrt as trt
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != sha256.lower():
                raise ValueError("Vehicle engine checksum mismatch")
            handle.seek(0)
            data = handle.read()
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.device = device
        with torch.cuda.device(device):
            self.engine = self.runtime.deserialize_cuda_engine(data)
            if self.engine is None:
                raise RuntimeError("Cannot load TensorRT engine for this GPU/runtime")
            self.context = self.engine.create_execution_context()
            inputs, outputs = [], []
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                (inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else outputs).append(name)
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError("Vehicle engine must have one input and one decoded output")
            self.input, self.output = inputs[0], outputs[0]
            self.shape = tuple(self.engine.get_tensor_shape(self.input))
            output_shape = tuple(self.engine.get_tensor_shape(self.output))
            if self.shape != (1, 3, size, size) or self.engine.get_tensor_dtype(self.input) != trt.float32:
                raise ValueError("Vehicle engine input does not match configured FP32 square input")
            if len(output_shape) != 3 or output_shape[0] != 1 or output_shape[2] != class_count+5:
                raise ValueError("Vehicle engine output does not match checkpoint classes")
            dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(self.output)))).dtype
            self.buffer = torch.empty(output_shape, dtype=dtype, device=device)
            self.stream = torch.cuda.Stream(device=device)

    def __call__(self, tensor):
        import torch
        if tuple(tensor.shape) != self.shape or tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise ValueError("Invalid TensorRT vehicle input")
        current = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(current)
        with torch.cuda.stream(self.stream):
            tensor.record_stream(self.stream)
            self.context.set_tensor_address(self.input, tensor.data_ptr())
            self.context.set_tensor_address(self.output, self.buffer.data_ptr())
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT vehicle inference failed")
        current.wait_stream(self.stream)
        return self.buffer
