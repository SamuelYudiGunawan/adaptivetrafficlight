import onnxruntime as ort

providers = ort.get_available_providers()
print("Available ONNX Runtime providers:", providers)

if "CUDAExecutionProvider" in providers:
    print("ONNX Runtime is using the GPU!")
else:
    print("ONNX Runtime is not using the GPU.")