"""PT → ONNX 导出脚本

用法:
  python pt_to_onnx.py <model.pt> [imgsz]

示例:
  python pt_to_onnx.py zuhe5.pt 640
  python pt_to_onnx.py best.pt
"""
import sys
import os

if len(sys.argv) < 2:
    print("用法: python pt_to_onnx.py <model.pt> [imgsz]")
    sys.exit(1)

pt_path = sys.argv[1]
imgsz = int(sys.argv[2]) if len(sys.argv) > 2 else 640

if not os.path.exists(pt_path):
    print(f"错误: 文件不存在 {pt_path}")
    sys.exit(1)

from ultralytics import YOLO

print(f"=== PT → ONNX 导出 ===")
print(f"输入: {pt_path}")
print(f"图像尺寸: {imgsz}")
print()

model = YOLO(pt_path)
onnx_path = model.export(
    format="onnx",
    imgsz=imgsz,
    opset=12,
    simplify=True,
)

print(f"\n导出完成: {onnx_path}")
print(f"文件大小: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")
