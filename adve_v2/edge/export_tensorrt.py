import torch
import torch.onnx
from pathlib import Path

# Try to import tensorrt, but don't crash if it is not installed (e.g. on host development CPU environments)
try:
    import tensorrt as trt
except ImportError:
    trt = None


def export_clip_to_onnx(output_path: str = "edge/clip_vitb32.onnx"):
    """Export CLIP vision encoder to ONNX."""
    import clip

    model, _ = clip.load("ViT-B/32", device="cpu")
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model.visual,
        dummy,
        output_path,
        input_names  = ["image"],
        output_names = ["embedding"],
        dynamic_axes = {"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version = 17,
    )
    print(f"Exported CLIP to ONNX: {output_path}")


def build_tensorrt_engine(
    onnx_path:   str,
    engine_path: str,
    fp16:        bool = True,
):
    """Build TensorRT engine from ONNX model."""
    if trt is None:
        print("TensorRT not available on this platform. Skipping engine compilation.")
        return

    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser  = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    engine = builder.build_serialized_network(network, config)

    with open(engine_path, "wb") as f:
        f.write(engine)

    print(f"TensorRT engine saved: {engine_path}")


if __name__ == "__main__":
    Path("edge").mkdir(exist_ok=True)
    export_clip_to_onnx("edge/clip_vitb32.onnx")
    build_tensorrt_engine(
        "edge/clip_vitb32.onnx",
        "edge/clip_vitb32_fp16.engine",
        fp16=True,
    )
    print("Edge deployment ready.")
