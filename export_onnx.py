import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from model.model import parsingNet
from scripts.onnx_export_utils import prepare_inference_state, resolve_output_paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="Export a fine-tuned hose model to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--max-abs-error", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    standard_path, best_path = resolve_output_paths(output_dir)

    net = parsingNet(
        pretrained=False,
        cls_dim=(51, 18, 1),
        use_aux=False,
    )
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    checkpoint_state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    inference_state = prepare_inference_state(checkpoint_state, net.state_dict())
    net.load_state_dict(inference_state, strict=True)
    net.eval()

    generator = torch.Generator(device="cpu").manual_seed(20260716)
    dummy_input = torch.randn(1, 3, 288, 384, generator=generator)
    with torch.no_grad():
        torch_output = net(dummy_input).cpu().numpy()

    torch.onnx.export(
        net,
        dummy_input,
        str(standard_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    model = onnx.load(str(standard_path))
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(standard_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(["output"], {"input": dummy_input.numpy()})[0]
    max_abs_error = float(np.max(np.abs(torch_output - ort_output)))
    if not np.isfinite(max_abs_error) or max_abs_error > args.max_abs_error:
        raise RuntimeError(
            f"ONNX parity failed: max_abs_error={max_abs_error:.8g}, "
            f"limit={args.max_abs_error:.8g}"
        )

    shutil.copy2(standard_path, best_path)
    if sha256(standard_path) != sha256(best_path):
        raise RuntimeError("the conventional and explicit-best ONNX files differ")

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "opset": args.opset,
        "input_shape": [1, 3, 288, 384],
        "output_shape": list(ort_output.shape),
        "max_abs_error": max_abs_error,
        "max_abs_error_limit": args.max_abs_error,
        "onnx": str(standard_path),
        "onnx_best": str(best_path),
        "onnx_sha256": sha256(standard_path),
    }
    report_path = output_dir / "onnx_export_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
