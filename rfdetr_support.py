"""RF-DETR 训练 / 推理适配层。

设计原则：
1. 本模块只在函数内部导入 rfdetr，未安装 rfdetr 时不影响 app.py 启动；
2. 模型统一存放在 models/rfdetr/*.pth，用 .pth 后缀与 YOLO 的 .pt 区分；
3. 数据集直接复用现有 YOLO 目录结构（rfdetr>=1.12 原生支持），仅需补一个 data.yaml。
"""

import os
import json
import glob
import shutil
import datetime

# 变体 -> 类名 / 默认分辨率 / 推荐 batch / 推荐梯度累积（RTX 3060 12GB，有效 batch 约 16）
# 分辨率必须是 patch_size(16) * num_windows(2) 的整数倍
RF_VARIANTS = {
    "RF-DETR Nano": {"cls": "RFDETRNano", "resolution": 384, "batch": 8, "grad_accum": 2},
    "RF-DETR Small": {"cls": "RFDETRSmall", "resolution": 512, "batch": 4, "grad_accum": 4},
    "RF-DETR Medium": {"cls": "RFDETRMedium", "resolution": 576, "batch": 2, "grad_accum": 8},
    "RF-DETR Large": {"cls": "RFDETRLarge", "resolution": 704, "batch": 1, "grad_accum": 16},
}

RF_MODEL_DIR = os.path.join("models", "rfdetr")

# 已加载模型缓存：{ (绝对路径, mtime): model_instance }
_model_cache = {}


def is_available():
    """rfdetr 是否已安装。"""
    try:
        import rfdetr  # noqa: F401
        return True
    except Exception:
        return False


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def list_models():
    """列出 models/rfdetr 下的全部 RF-DETR 权重（返回文件名列表）。"""
    if not os.path.isdir(RF_MODEL_DIR):
        return []
    return sorted(
        f for f in os.listdir(RF_MODEL_DIR)
        if f.endswith(".pth") and os.path.isfile(os.path.join(RF_MODEL_DIR, f))
    )


def find_model(name):
    """按文件名定位权重完整路径，找不到返回 None。"""
    if not name:
        return None
    path = os.path.join(RF_MODEL_DIR, name)
    return path if os.path.isfile(path) else None


def _read_class_names(data_path):
    """从 classes/classes.txt 读取类别名；缺失时回退解析 dataset.yaml。"""
    classes_file = os.path.join(data_path, "classes", "classes.txt")
    if os.path.isfile(classes_file):
        for enc in ("utf-8", "gbk", "gb2312", "iso-8859-1"):
            try:
                with open(classes_file, "r", encoding=enc) as f:
                    names = [line.strip() for line in f if line.strip()]
                if names:
                    return names
            except UnicodeDecodeError:
                continue

    for yaml_name in ("data.yaml", "dataset.yaml"):
        yaml_path = os.path.join(data_path, yaml_name)
        if not os.path.isfile(yaml_path):
            continue
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
            names = content.get("names")
            if isinstance(names, list):
                return [str(n) for n in names]
            if isinstance(names, dict):
                return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        except Exception:
            pass
    return []


def _ensure_data_yaml(data_path):
    """RF-DETR 只认 data.yaml / data.yml，本项目生成的是 dataset.yaml，这里补一份。"""
    data_yaml = os.path.join(data_path, "data.yaml")
    if os.path.isfile(data_yaml):
        return data_yaml

    src = os.path.join(data_path, "dataset.yaml")
    if os.path.isfile(src):
        shutil.copy2(src, data_yaml)
        return data_yaml

    # 两者都不存在时，根据目录结构和类别文件现场生成
    import yaml
    names = _read_class_names(data_path)
    content = {
        "path": os.path.abspath(data_path),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(content, f, sort_keys=False, allow_unicode=True)
    return data_yaml


def _make_log_callback(log_cb, stop_flag, counter):
    """构造 Lightning 回调，把每个 epoch 的进度转发给前端日志。"""
    from pytorch_lightning.callbacks import Callback

    class _RFLogCallback(Callback):
        def on_train_epoch_end(self, trainer, pl_module):
            if stop_flag is not None and stop_flag.is_set():
                trainer.should_stop = True
                log_cb("⏹️ 收到停止信号，正在保存当前结果并结束训练...")
                return
            epoch = int(trainer.current_epoch) + 1
            total = int(trainer.max_epochs)
            counter[0] = epoch
            msg = f"Epoch {epoch}/{total}"
            try:
                metrics = trainer.callback_metrics
                parts = []
                for key, label in (
                    ("val/mAP_50_95", "mAP50-95"),
                    ("val/mAP_50", "mAP50"),
                    ("train/loss", "loss"),
                ):
                    if key in metrics:
                        parts.append(f"{label}={float(metrics[key]):.4f}")
                if parts:
                    msg += " | " + " | ".join(parts)
            except Exception:
                pass
            log_cb(msg)

    return _RFLogCallback()


def train(data_path, variant, epochs, batch_size, lr, resolution,
          grad_accum_steps, device, log_cb, stop_flag):
    """执行一次 RF-DETR 训练，返回结果文本。

    参数:
        data_path: 数据集目录（含 images/ 与 labels/）
        variant: RF_VARIANTS 的键
        log_cb: 单参数回调，用于向前端推送日志
    """
    import rfdetr
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge

    classes = {
        "RF-DETR Nano": RFDETRNano,
        "RF-DETR Small": RFDETRSmall,
        "RF-DETR Medium": RFDETRMedium,
        "RF-DETR Large": RFDETRLarge,
    }
    if variant not in RF_VARIANTS:
        return f"❌ 不支持的 RF-DETR 变体: {variant}"

    data_path = os.path.abspath(data_path)
    if not os.path.isdir(os.path.join(data_path, "images", "train")):
        return "❌ 数据集缺少 images/train，请先完成数据集分割（勾选「包含测试集」并启动一次训练即可生成）"

    class_names = _read_class_names(data_path)
    if not class_names:
        return "❌ 未读取到类别名称，请检查 classes/classes.txt"

    _ensure_data_yaml(data_path)

    resolution = int(resolution)
    model_cls = classes[variant]
    log_cb(f"🧩 加载 {variant}（num_classes={len(class_names)}, resolution={resolution}）...")

    model = model_cls(
        num_classes=len(class_names),
        resolution=resolution,
        device=device,
    )

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = _ensure_dir(os.path.join("runs", f"rfdetr_{stamp}"))
    counter = [0]

    log_cb("🚀 开始 RF-DETR 训练...")

    # rfdetr 1.12 的 model.train() 不接受 callbacks 参数（TrainConfig 里没有该字段），
    # 但内部会调用 rfdetr.training.build_trainer(config, model_config, extra_callbacks=...)，
    # 这里临时包装 build_trainer 注入 epoch 日志/停止回调，训练结束后还原。
    cb = _make_log_callback(log_cb, stop_flag, counter)
    patched = False
    _orig_build_trainer = None
    try:
        import rfdetr.training as _rft
        _orig_build_trainer = getattr(_rft, "build_trainer", None)
        if _orig_build_trainer is not None:
            def _build_trainer_with_cb(*args, **kwargs):
                extra = list(kwargs.get("extra_callbacks") or ())
                extra.append(cb)
                kwargs["extra_callbacks"] = extra
                return _orig_build_trainer(*args, **kwargs)

            _rft.build_trainer = _build_trainer_with_cb
            patched = True
    except Exception as e:
        log_cb(f"⚠️ 未能挂载训练进度回调（不影响训练本身）: {e}")

    try:
        model.train(
            dataset_dir=data_path,
            dataset_file="yolo",
            epochs=int(epochs),
            batch_size=int(batch_size),
            grad_accum_steps=max(1, int(grad_accum_steps)),
            lr=float(lr),
            resolution=resolution,
            output_dir=output_dir,
            device=device,
            notes={"class_names": class_names, "variant": variant, "resolution": resolution},
        )
    finally:
        if patched:
            _rft.build_trainer = _orig_build_trainer

    best = os.path.join(output_dir, "checkpoint_best_total.pth")
    if not os.path.isfile(best):
        # 兜底：取 output_dir 下最新的 .pth
        candidates = glob.glob(os.path.join(output_dir, "**", "*.pth"), recursive=True)
        best = max(candidates, key=os.path.getmtime) if candidates else None

    if not best or not os.path.isfile(best):
        return f"⚠️ 训练结束，但未找到最佳权重（输出目录: {output_dir}）"

    slug = variant.lower().replace("rf-detr ", "rfdetr-").replace(" ", "-")
    _ensure_dir(RF_MODEL_DIR)
    target_name = f"{slug}_{stamp}.pth"
    target_path = os.path.join(RF_MODEL_DIR, target_name)
    shutil.copy2(best, target_path)

    # 侧车元数据：加载权重时必须知道类别名与分辨率
    meta = {
        "variant": variant,
        "class_names": class_names,
        "num_classes": len(class_names),
        "resolution": resolution,
        "trained_at": stamp,
        "epochs": int(epochs),
    }
    with open(os.path.splitext(target_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    result = "✅ RF-DETR 训练完成。\n"
    result += f"模型名称: {target_name}（可在图片预测/模型管理中选择）\n"
    result += f"类别数: {len(class_names)} · 分辨率: {resolution} · 完成轮次: {counter[0]}/{int(epochs)}\n"
    result += f"训练输出目录: {output_dir}"
    return result


def _load_model(weight_path, device):
    """加载并缓存 RF-DETR 权重。"""
    import torch
    from rfdetr import from_checkpoint

    weight_path = os.path.abspath(weight_path)
    key = (weight_path, os.path.getmtime(weight_path))
    if key in _model_cache:
        return _model_cache[key]

    if device in (None, "", "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = from_checkpoint(weight_path, device=device)
    _model_cache.clear()  # 只保留最新一个，避免显存堆积
    _model_cache[key] = model
    return model


def predict(weight_path, image_rgb, conf=0.5, translate=None):
    """单张图片推理。

    返回 (标注后的 RGB 图像, json 结果)。json 结果格式与 YOLO 分类/检测分支保持兼容。
    """
    import numpy as np
    import supervision as sv

    model = _load_model(weight_path, "auto")

    detections = model.predict(image_rgb, threshold=float(conf), include_source_image=False)

    names = None
    class_name_data = getattr(detections, "data", {}) or {}
    if "class_name" in class_name_data:
        names = [str(n) for n in np.asarray(class_name_data["class_name"]).ravel()]

    if names is None:
        meta_path = os.path.splitext(weight_path)[0] + ".json"
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                cls_names = meta.get("class_names") or []
                ids = np.asarray(detections.class_id).ravel()
                names = [str(cls_names[int(i)]) if 0 <= int(i) < len(cls_names) else str(int(i)) for i in ids]
            except Exception:
                names = None

    if names is None:
        names = [str(int(i)) for i in np.asarray(detections.class_id).ravel()]

    if translate:
        names = [translate(n) for n in names]

    labels = [f"{n} {c:.2f}" for n, c in zip(names, np.asarray(detections.confidence).ravel())]

    annotated = image_rgb.copy()
    if len(detections) > 0:
        annotated = sv.BoxAnnotator().annotate(scene=annotated, detections=detections)
        annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections, labels=labels)

    result = {
        "framework": "RF-DETR",
        "detections": [
            {
                "class": n,
                "confidence": round(float(c), 4),
                "box": [round(float(v), 2) for v in b],
            }
            for n, c, b in zip(
                names,
                np.asarray(detections.confidence).ravel(),
                np.asarray(detections.xyxy).reshape(-1, 4),
            )
        ],
    }
    return annotated, result
