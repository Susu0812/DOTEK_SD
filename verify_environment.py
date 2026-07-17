"""Verify the local hose-recognition Anaconda environment."""

from pathlib import Path

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import PIL
import scipy
import sklearn
import torch
import torchvision

from model.model import parsingNet


ROOT = Path(__file__).resolve().parent
VISUAL_ROOT = ROOT.parent / "水带识别可视化代码"


def verify_pytorch() -> None:
    """Load the delivered checkpoint and execute one deployment forward pass."""
    checkpoint_path = (
        ROOT / "logs" / "0209_1509_lr_1e-04_b_64" / "latest_model.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint["model"].items()
    }
    training_model = parsingNet(
        pretrained=False,
        cls_dim=(51, 18, 1),
        use_aux=True,
    )
    incompatible = training_model.load_state_dict(state_dict, strict=False)
    print(
        f"checkpoint_missing_keys={len(incompatible.missing_keys)}; "
        f"checkpoint_unexpected_keys={len(incompatible.unexpected_keys)}"
    )

    model = parsingNet(
        pretrained=False,
        cls_dim=(51, 18, 1),
        use_aux=False,
    ).eval()
    sample = torch.randn(1, 3, 288, 384)
    with torch.no_grad():
        output = model(sample)
    print(f"project_torch_output={tuple(output.shape)}")


def verify_onnx_models() -> None:
    """Load every delivered ONNX model and execute one inference pass."""
    model_paths = [
        ROOT / "resnet_288_384.onnx",
        ROOT / "resnet_288_384_int8.onnx",
        VISUAL_ROOT / "resnet_288_384.onnx",
        VISUAL_ROOT / "resnet_288_384_int8.onnx",
    ]
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    failures = []

    for model_path in model_paths:
        relative_path = model_path.relative_to(ROOT.parent)
        try:
            onnx.checker.check_model(onnx.load(model_path))
            session = ort.InferenceSession(str(model_path), providers=providers)
            input_meta = session.get_inputs()[0]
            input_data = np.random.randn(1, 3, 288, 384).astype(np.float32)
            output = session.run(None, {input_meta.name: input_data})[0]
            print(
                f"onnx={relative_path}; input={input_meta.shape}; "
                f"output={tuple(output.shape)}; providers={session.get_providers()}"
            )
        except Exception as exc:
            failures.append((relative_path, exc))
            print(f"onnx_failed={relative_path}; error={type(exc).__name__}: {exc}")

    print(f"onnx_failed_count={len(failures)}")


def main() -> None:
    print(f"python={'.'.join(map(str, __import__('sys').version_info[:3]))}")
    print(f"numpy={np.__version__}")
    print(f"opencv={cv2.__version__}")
    print(f"pillow={PIL.__version__}")
    print(f"scipy={scipy.__version__}")
    print(f"scikit_learn={sklearn.__version__}")
    print(f"torch={torch.__version__}")
    print(f"torchvision={torchvision.__version__}")
    print(f"torch_cuda_build={torch.version.cuda}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"torch_cuda_device={torch.cuda.get_device_name(0)}")
    print(f"onnx={onnx.__version__}")
    print(f"onnxruntime={ort.__version__}")
    print(f"onnxruntime_available_providers={ort.get_available_providers()}")

    verify_pytorch()
    verify_onnx_models()


if __name__ == "__main__":
    main()
