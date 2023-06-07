import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from pathlib import Path

PREALLOC_OUTPUTS = 8

class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        del self.device
        del self.host        

class TensorrtInferer:

    def _get_input_shape(self):
        """Get input shape of the TensorRT YOLO engine."""
        binding = self._engine[0]
        assert self._engine.binding_is_input(binding)
        binding_dims = self._engine.get_binding_shape(binding)
        if len(binding_dims) == 4:
            return tuple(binding_dims[2:])
        elif len(binding_dims) == 3:
            return tuple(binding_dims[1:])
        else:
            raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))


    def _allocate_buffers(self):
        """Allocates all host/device in/out buffers required for an engine."""
        inputs = []
        outputs = []
        bindings = []
        output_idx = 0
        stream = cuda.Stream()
        for binding in self._engine:
            binding_dims = self._engine.get_binding_shape(binding)
            if len(binding_dims) == 4:
                # explicit batch case (TensorRT 7+)
                size = trt.volume(binding_dims)
            elif len(binding_dims) == 3:
                # implicit batch case (TensorRT 6 or older)
                size = trt.volume(binding_dims) * self._engine.max_batch_size
            else:
                raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self._engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                # each grid has 3 anchors, each anchor generates a detection
                # output of 7 float32 values
                assert size % 7 == 0
                outputs.append(HostDeviceMem(host_mem, device_mem))
                output_idx += 1
        assert len(inputs) == 1
        assert len(outputs) == 1
        return inputs, outputs, bindings, stream

    def _do_inference_v2(self):
        """do_inference_v2 (for TensorRT 7.0+)

        This function is generalized for multiple inputs/outputs for full
        dimension networks.
        Inputs and outputs are expected to be lists of HostDeviceMem objects.
        """
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]
        # Run inference.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]
        # Synchronize the stream
        self.stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in self.outputs]

    def _load_engine(self, engine_path: Path):
        with open(engine_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def __init__(self, engine_path: Path):
        self.cuda_ctx = None

        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()

        self.input_shape = self._get_input_shape(self.engine)

        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = \
                self._allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

    def __del__(self):
        """Free CUDA memories."""
        del self.outputs
        del self.inputs
        del self.stream

    def detect(self, infer_data: np.array):
        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        self.inputs[0].host = np.ascontiguousarray(infer_data)
        if self.cuda_ctx:
            self.cuda_ctx.push()
        trt_outputs = self._do_inference_v2()
        if self.cuda_ctx:
            self.cuda_ctx.pop()

        return trt_outputs[0].copy()
