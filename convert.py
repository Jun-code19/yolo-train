"""ONNX → RKNN 转换脚本

用法:
  python convert.py <model.onnx> [output.rknn] [--quantize DATASET.txt]

示例:
  python convert.py zuhe5.onnx
  python convert.py zuhe5.onnx output.rknn
  python convert.py zuhe5.onnx output.rknn --quantize dataset.txt
"""
import sys
import os

def main():
    if len(sys.argv) < 2:
        print("用法: python convert.py <model.onnx> [output.rknn] [--quantize DATASET.txt]")
        print("示例:")
        print("  python convert.py zuhe5.onnx")
        print("  python convert.py zuhe5.onnx output.rknn")
        print("  python convert.py zuhe5.onnx output.rknn --quantize dataset.txt")
        sys.exit(1)

    onnx_path = sys.argv[1]

    # 输出路径：指定则用，否则同名 .rknn
    if len(sys.argv) > 2 and not sys.argv[2].startswith("--"):
        rknn_path = sys.argv[2]
    else:
        rknn_path = onnx_path.replace(".onnx", ".rknn")

    # 量化
    do_quant = "--quantize" in sys.argv
    dataset = None
    if do_quant:
        idx = sys.argv.index("--quantize")
        if idx + 1 < len(sys.argv):
            dataset = sys.argv[idx + 1]
        else:
            print("错误: --quantize 需要指定 dataset.txt 路径")
            sys.exit(1)

    if not os.path.exists(onnx_path):
        print(f"错误: 文件不存在 {onnx_path}")
        sys.exit(1)

    from rknn.api import RKNN

    print(f"=== ONNX → RKNN 转换 ===")
    print(f"输入: {onnx_path}")
    print(f"输出: {rknn_path}")
    print(f"目标平台: rk3588")
    print(f"量化: {'是' if do_quant else '否'}")
    if do_quant:
        print(f"校准数据集: {dataset}")
    print()

    rknn = RKNN(verbose=True)

    # mean=0, std=255 → 归一化到 [0,1]，与 YOLO 默认预处理一致
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
    )

    print("=== 加载 ONNX ===")
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"load_onnx 失败: {ret}")
        sys.exit(ret)

    print("=== 构建 RKNN ===")
    if do_quant:
        ret = rknn.build(do_quantization=True, dataset=dataset)
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        print(f"build 失败: {ret}")
        sys.exit(ret)

    print("=== 导出 RKNN ===")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f"export_rknn 失败: {ret}")
        sys.exit(ret)

    rknn.release()
    print(f"\n转换完成: {rknn_path}")
    print(f"文件大小: {os.path.getsize(rknn_path) / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
