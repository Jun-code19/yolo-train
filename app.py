import sys
import os
import shutil
import json
import zipfile
import tarfile
import tempfile

# 动态获取 FFmpeg 路径
def get_ffmpeg_path():
    # 优先使用系统环境变量或在 PATH 中查找
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    # 回退到本地路径 (Windows)
    local_path = os.path.join(os.path.dirname(__file__), "FFmpeg", "bin", "ffmpeg.exe")
    if os.path.exists(local_path):
        return local_path
    return "ffmpeg" # 最终回退,依赖于系统PATH

import dill

import gradio as gr
from ultralytics import YOLO
import os
import cv2
import numpy as np
import torch
import tempfile
import subprocess
import shutil
from torchvision.ops import nms
import yaml
from sklearn.model_selection import train_test_split
import json
import logging
import platform
import glob
import threading
import queue as queue_module
import time
import datetime

logging.basicConfig(level=logging.INFO)

# 训练状态管理
_training_stop_flag = threading.Event()
_training_log_queue = queue_module.Queue()
_training_all_logs = []          # 全部日志（供刷新恢复）
_training_active = False         # 训练是否进行中
_training_result = None          # 训练最终结果
_training_lock = threading.Lock()  # 保护状态变量

# 训练任务追踪
_tasks = []            # list[dict]，新任务追加在末尾
_task_id_counter = 0
_schedule_timers = {}  # task_id -> threading.Timer


def _fmt_time(ts):
    return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S") if ts else "—"


def _add_task(status, task, model, dataset, epochs, schedule_at=None):
    """新增任务并返回 task_id。"""
    global _task_id_counter
    with _training_lock:
        _task_id_counter += 1
        tid = _task_id_counter
        task_obj = {
            "id": tid,
            "status": status,              # scheduled / running / completed / stopped / failed
            "task": task,                  # 检测 / 分割 / 姿态 / 分类 / OBB
            "model": model,
            "dataset": dataset,
            "epochs": int(epochs),
            "current_epoch": 0,
            "create_at": time.time(),
            "start_at": None if status == "scheduled" else time.time(),
            "end_at": None,
            "schedule_at": schedule_at,    # 预约时间戳
            "result_msg": None,
        }
        _tasks.append(task_obj)
        return tid


def _update_task(tid, **kwargs):
    with _training_lock:
        for t in _tasks:
            if t["id"] == tid:
                t.update(kwargs)
                return t
    return None


def _list_tasks_html():
    """返回任务列表的 HTML（供首页展示）。"""
    with _training_lock:
        items = list(reversed(_tasks)) if _tasks else []

    if not items:
        return (
            '<div class="task-panel"><div class="task-empty">暂无训练任务，前往「🎯 训练」标签页开始训练或预约任务。</div></div>'
        )

    status_badge = {
        "scheduled": ("📅 已预约", "badge-scheduled"),
        "running": ("🔄 运行中", "badge-running"),
        "completed": ("✅ 已完成", "badge-completed"),
        "stopped": ("⏹️ 已停止", "badge-stopped"),
        "failed": ("❌ 失败", "badge-failed"),
    }
    rows_html = []
    for t in items:
        label, cls = status_badge.get(t["status"], ("—", "badge-running"))
        dataset_name = os.path.basename(t["dataset"]) if t["dataset"] else "—"
        schedule_str = f"预约：{_fmt_time(t['schedule_at'])}" if t["status"] == "scheduled" and t["schedule_at"] else ""
        end_str = _fmt_time(t["end_at"]) if t["end_at"] else "—"
        ep_str = f"{t['current_epoch']}/{t['epochs']}" if t["status"] == "running" else str(t["epochs"])
        rows_html.append(f"""
        <div class="task-row">
          <div class="task-row-head">
            <span class="task-id">#{t['id']}</span>
            <span class="badge {cls}">{label}</span>
            <span class="task-title">{t['task']} · {t['model']}</span>
          </div>
          <div class="task-row-body">
            <span>📊 数据集：{dataset_name}</span>
            <span>🗂️ Epochs：{ep_str}</span>
            <span>⏱️ 开始：{_fmt_time(t['start_at'])}</span>
            <span>🏁 结束：{end_str}</span>
            {"<span>" + schedule_str + "</span>" if schedule_str else ""}
          </div>
          {f'<div class="task-result">{t["result_msg"]}</div>' if t["result_msg"] else ""}
        </div>""")

    return (
        '<div class="task-panel">'
        + "".join(rows_html)
        + "</div>"
    )

# 检查CUDA是否可用
cuda_available = torch.cuda.is_available()

# 初始化YOLO模型
models = {}

ONLINE_MODELS = {
    "检测": ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"],
    "分割": ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt"],
    "姿态": ["yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt"],
    "分类": ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt", "yolo11l-cls.pt", "yolo11x-cls.pt"],
    "OBB": ["yolo11n-obb.pt", "yolo11s-obb.pt", "yolo11m-obb.pt", "yolo11l-obb.pt", "yolo11x-obb.pt"]
}

def get_available_models(task):
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    task_folders = {
        "检测": "detection",
        "分割": "segmentation",
        "姿态": "pose",
        "分类": "classification",
        "OBB": "obb"
    }
    
    task_folder = task_folders.get(task)
    if not task_folder:
        return []
    
    task_path = os.path.join(models_dir, task_folder)
    if not os.path.exists(task_path):
        os.makedirs(task_path)
    
    local_models = set(f for f in os.listdir(task_path) if f.endswith('.pt'))
    online_models = set(ONLINE_MODELS.get(task, []))
    
    all_models = local_models.union(online_models)
    return sorted(list(all_models))

def get_model(task, model_name, device):
    task_folders = {
        "检测": "detection",
        "分割": "segmentation",
        "姿态": "pose",
        "分类": "classification",
        "OBB": "obb"
    }
    task_folder = task_folders.get(task)
    if not task_folder:
        raise ValueError(f"不支持的任务类型: {task}")
    
    model_path = os.path.join("models", task_folder, model_name)
    if not os.path.exists(model_path):
        model_path = download_model(task, model_name)
    
    if (task, model_name) not in models:
        models[(task, model_name)] = YOLO(model_path)
    if device == "cuda" and cuda_available:
        models[(task, model_name)].to(device)
    return models[(task, model_name)]

def split_dataset(data_path, test_size=0.2, val_size=0.2):
    images_dir = os.path.join(data_path, 'images')
    labels_dir = os.path.join(data_path, 'labels')
    
    if not os.path.exists(images_dir):
        raise ValueError(f"图像目录不存在: {images_dir}")
    
    all_images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if len(all_images) == 0:
        raise ValueError(f"在 {images_dir} 中没有找到图像文件")
    
    # 首先分割出验证集和测试集
    train_images, val_test_images = train_test_split(all_images, test_size=(test_size + val_size), random_state=42)
    val_images, test_images = train_test_split(val_test_images, test_size=val_size/(test_size + val_size), random_state=42)
    
    # 创建新的目录结构
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(data_path, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(data_path, 'labels', split), exist_ok=True)
    
    # 复制文件
    for split, image_list in [('train', train_images), ('val', val_images), ('test', test_images)]:
        for img in image_list:
            shutil.copy2(os.path.join(images_dir, img), os.path.join(data_path, 'images', split, img))
            label = img.rsplit('.', 1)[0] + '.txt'
            if os.path.exists(os.path.join(labels_dir, label)):
                shutil.copy2(os.path.join(labels_dir, label), os.path.join(data_path, 'labels', split, label))

def create_dataset_yaml(data_path, include_test=False):
    class_names = get_class_names(data_path)
    yaml_content = {
        'path': os.path.abspath(data_path),  # 使用绝对路径
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(class_names),
        'names': class_names
    }
    
    if include_test:
        yaml_content['test'] = 'images/test'
    
    yaml_path = os.path.join(data_path, 'dataset.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as file:
        yaml.dump(yaml_content, file, sort_keys=False, allow_unicode=True)
    
    return yaml_path

def get_class_names(data_path):
    classes_file = os.path.join(data_path, 'classes', 'classes.txt')
    if os.path.exists(classes_file):
        encodings = ['utf-8', 'gbk', 'gb2312', 'iso-8859-1']
        for encoding in encodings:
            try:
                with open(classes_file, 'r', encoding=encoding) as f:
                    class_names = [line.strip() for line in f.readlines() if line.strip()]
                return class_names
            except UnicodeDecodeError:
                continue
        logging.warning(f"无法以支持的编码读取类别文件: {classes_file}")
        return []
    else:
        logging.warning(f"未找到类别文件: {classes_file}")
        return []

def get_ultralytics_settings():
    settings_path = os.path.join(os.getcwd(), 'ultralytics_settings.json')
    if os.path.exists(settings_path):
        with open(settings_path, 'r') as f:
            return json.load(f)
    return {'datasets_dir': './datasets'}

def check_dataset(data_path):
    images_dir = os.path.join(data_path, 'images')
    labels_dir = os.path.join(data_path, 'labels')
    classes_file = os.path.join(data_path, 'classes', 'classes.txt')

    # 检查目录结构
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        logging.error(f"数据集结构不正确。请确保 {data_path} 目录下有 'images' 和 'labels' 子目录。")
        return False

    # 检查类别文件
    if not os.path.exists(classes_file):
        logging.error(f"未找到类别文件: {classes_file}")
        return False

    # 检查图像和标签文件的对应关系
    image_files = set(f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg')))
    label_files = set(f for f in os.listdir(labels_dir) if f.endswith('.txt'))
    
    if len(image_files) == 0:
        logging.error(f"在 {images_dir} 中没有找到图像文件")
        return False

    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + '.txt'
        if label_file not in label_files:
            logging.warning(f"图像文件 {img_file} 没有对应的标签文件")

    # 检查标签文件格式
    class_names = get_class_names(data_path)
    for label_file in label_files:
        with open(os.path.join(labels_dir, label_file), 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    logging.error(f"标签文件 {label_file} 格式不正确")
                    return False
                class_id = int(parts[0])
                if class_id >= len(class_names):
                    logging.error(f"标签文件 {label_file} 中的类别ID {class_id} 超出范围")
                    return False
                x, y, w, h = map(float, parts[1:])
                if not all(0 <= v <= 1 for v in (x, y, w, h)):
                    logging.error(f"标签文件 {label_file} 中的坐标不在0-1范围内")
                    return False

    logging.info("数据集检查通过")
    return True

def train_model(data_path, epochs, task, model_name, device, include_test, test_size, val_size, 
                batch_size, imgsz, patience, optimizer, lr0, lrf, momentum, weight_decay, 
                warmup_epochs, warmup_momentum, warmup_bias_lr, box, cls, dfl, pose, kobj, 
                label_smoothing, nbs, overlap_mask, mask_ratio, dropout,
                # 新增的参数
                augment, mixup, copy_paste, mosaic, amp):
    try:
        if not check_dataset(data_path):
            return "数据集检查失败，请查看日志以获取详细信息。"

        settings = get_ultralytics_settings()
        datasets_dir = settings.get('datasets_dir', '.')
        
        if not os.path.isabs(data_path):
            data_path = os.path.abspath(data_path)
        
        if not os.path.exists(data_path):
            return f"错误：数据集路径不存在 - {data_path}"
        
        if not (os.path.exists(os.path.join(data_path, 'images')) and 
                os.path.exists(os.path.join(data_path, 'labels'))):
            return f"错误：数据集结构不正确。请确保 {data_path} 目录下有 'images' 和 'labels' 子目录。"
        
        split_dataset(data_path, test_size=float(test_size), val_size=float(val_size))
        
        yaml_path = create_dataset_yaml(data_path, include_test=include_test)
        
        model = get_model(task, model_name, device)
        
        # 注册停止 + 日志回调
        def _on_train_epoch_end(trainer):
            if _training_stop_flag.is_set():
                raise KeyboardInterrupt("用户主动停止训练")
            epoch = trainer.epoch + 1
            total = trainer.epochs
            try:
                loss_val = float(trainer.loss_items.mean()) if hasattr(trainer, 'loss_items') else 0.0
            except Exception:
                loss_val = 0.0
            msg = f"Epoch {epoch}/{total} | loss: {loss_val:.4f}"
            _training_log_queue.put(msg)
            _training_all_logs.append(msg)
        
        def _on_fit_epoch_end(trainer):
            m = trainer.metrics
            parts = []
            for k in ('metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)'):
                if k in m:
                    parts.append(f"{k.split('/')[-1].replace('(B)', '')}={float(m[k]):.3f}")
            if parts:
                msg = "  -> " + " | ".join(parts)
                _training_log_queue.put(msg)
                _training_all_logs.append(msg)
        
        model.add_callback("on_train_epoch_end", _on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
        
        torch.cuda.empty_cache()  # 在训练开始前清理GPU缓存
        
        results = model.train(
            data=yaml_path, 
            epochs=int(epochs), 
            device=device,
            batch=int(batch_size),
            imgsz=int(imgsz),
            patience=int(patience),
            optimizer=optimizer,
            lr0=float(lr0),
            lrf=float(lrf),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
            warmup_epochs=float(warmup_epochs),
            warmup_momentum=float(warmup_momentum),
            warmup_bias_lr=float(warmup_bias_lr),
            box=float(box),
            cls=float(cls),
            dfl=float(dfl),
            pose=float(pose),
            kobj=float(kobj),
            label_smoothing=float(label_smoothing),
            nbs=int(nbs),
            overlap_mask=overlap_mask,
            mask_ratio=int(mask_ratio),
            dropout=float(dropout),
            cache='disk',
            amp=amp,
            augment=augment,
            mixup=float(mixup),
            copy_paste=float(copy_paste),
            mosaic=float(mosaic),
        )
        
        # 提取关键信息
        metrics = results.results_dict
        best_fitness = metrics.get('fitness', 'N/A')
        mAP50 = metrics.get('metrics/mAP50(B)', 'N/A')
        mAP50_95 = metrics.get('metrics/mAP50-95(B)', 'N/A')
        
        # 获取模型保存路径
        save_dir = results.save_dir
        best_model_path = os.path.join(save_dir, 'weights', 'best.pt')
        
        # 自动复制训练结果到 models/ 目录，使其在各下拉框中可见
        task_folders = {
            "检测": "detection", "分割": "segmentation", "姿态": "pose",
            "分类": "classification", "OBB": "obb"
        }
        task_folder = task_folders.get(task, "detection")
        models_target_dir = os.path.join("models", task_folder)
        os.makedirs(models_target_dir, exist_ok=True)
        
        # 生成不冲突的文件名
        base_name = model_name.replace('.pt', '')
        trained_name = f"{base_name}_trained.pt"
        trained_path = os.path.join(models_target_dir, trained_name)
        counter = 1
        while os.path.exists(trained_path):
            trained_name = f"{base_name}_trained_{counter}.pt"
            trained_path = os.path.join(models_target_dir, trained_name)
            counter += 1
        
        copied = False
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, trained_path)
            copied = True
            # 清除模型缓存，确保下次加载的是新文件
            cache_key = (task, trained_name)
            if cache_key in models:
                del models[cache_key]
        
        # 构建简化的结果字符串
        result_str = f"训练完成。\n"
        result_str += f"最佳fitness: {best_fitness}\n"
        result_str += f"mAP50: {mAP50}\n"
        result_str += f"mAP50-95: {mAP50_95}\n"
        if copied:
            result_str += f"最佳模型已保存至: {trained_path}\n"
            result_str += f"模型名称: {trained_name}（可在图片预测/模型管理中选择）"
        else:
            result_str += f"最佳模型保存路径: {best_model_path}"
        
        return result_str
    except KeyboardInterrupt:
        _training_stop_flag.clear()
        # 尝试复制已保存的 best.pt
        try:
            runs_dir = os.path.join("runs")
            best_pt = None
            if os.path.exists(runs_dir):
                for root, dirs, files in os.walk(runs_dir):
                    if "best.pt" in files:
                        candidate = os.path.join(root, "best.pt")
                        if best_pt is None or os.path.getmtime(candidate) > os.path.getmtime(best_pt):
                            best_pt = candidate
            if best_pt:
                task_folders = {"检测": "detection", "分割": "segmentation", "姿态": "pose",
                                "分类": "classification", "OBB": "obb"}
                task_folder = task_folders.get(task, "detection")
                models_target_dir = os.path.join("models", task_folder)
                os.makedirs(models_target_dir, exist_ok=True)
                base_name = model_name.replace('.pt', '')
                trained_name = f"{base_name}_trained.pt"
                trained_path = os.path.join(models_target_dir, trained_name)
                counter = 1
                while os.path.exists(trained_path):
                    trained_name = f"{base_name}_trained_{counter}.pt"
                    trained_path = os.path.join(models_target_dir, trained_name)
                    counter += 1
                shutil.copy2(best_pt, trained_path)
                return f"⏹️ 训练已被用户停止。已完成的 epoch 模型已保存: {trained_name}（可在图片预测/模型管理中选择）"
        except Exception:
            pass
        return "⏹️ 训练已被用户停止。已完成的 epoch 结果已保存。"
    except Exception as e:
        logging.error(f"训练过程中出现错误: {str(e)}", exc_info=True)
        return f"训练过程中出现错误: {str(e)}"
    finally:
        _training_stop_flag.clear()

def translate_label(label):
    # 扩展翻译字典,包含更多常见物体
    translations = {
        'person': '人',
        'bicycle': '自行车',
        'car': '汽车',
        'motorcycle': '摩托车',
        'airplane': '飞机',
        'bus': '公交车',
        'train': '火车',
        'truck': '卡车',
        'boat': '船',
        'traffic light': '交通灯',
        'fire hydrant': '消防栓',
        'stop sign': '停止标志',
        'parking meter': '停车计时器',
        'bench': '长凳',
        'bird': '鸟',
        'cat': '猫',
        'dog': '狗',
        'horse': '马',
        'sheep': '羊',
        'cow': '牛',
        'elephant': '大象',
        'bear': '熊',
        'zebra': '斑马',
        'giraffe': '长颈鹿',
        'backpack': '背包',
        'umbrella': '雨伞',
        'handbag': '手提包',
        'tie': '领带',
        'suitcase': '行李箱',
        'frisbee': '飞盘',
        'skis': '滑雪板',
        'snowboard': '单板滑雪',
        'sports ball': '运动球',
        'kite': '风筝',
        'baseball bat': '棒球棒',
        'baseball glove': '棒球手套',
        'skateboard': '滑板',
        'surfboard': '冲浪板',
        'tennis racket': '网球拍',
        'bottle': '瓶子',
        'wine glass': '酒杯',
        'cup': '杯子',
        'fork': '叉子',
        'knife': '刀',
        'spoon': '勺子',
        'bowl': '碗',
        'banana': '香蕉',
        'apple': '苹果',
        'sandwich': '三明治',
        'orange': '橙子',
        'broccoli': '西兰花',
        'carrot': '胡萝卜',
        'hot dog': '热狗',
        'pizza': '披萨',
        'donut': '甜甜圈',
        'cake': '蛋糕',
        'chair': '椅子',
        'couch': '沙发',
        'potted plant': '盆栽',
        'bed': '床',
        'dining table': '餐桌',
        'toilet': '马桶',
        'tv': '电视',
        'laptop': '笔记本电脑',
        'mouse': '鼠标',
        'remote': '遥控器',
        'keyboard': '键盘',
        'cell phone': '手机',
        'microwave': '微波炉',
        'oven': '烤箱',
        'toaster': '烤面包机',
        'sink': '水槽',
        'refrigerator': '冰箱',
        'book': '书',
        'clock': '时钟',
        'vase': '花瓶',
        'scissors': '剪刀',
        'teddy bear': '泰迪熊',
        'hair drier': '吹风机',
        'toothbrush': '牙刷',
    }
    return translations.get(label.lower(), label)  # 如果没有翻译,返回原标签

def predict_image(image, task, model_name, device, conf, iou, max_det, classes, agnostic_nms, augment):
    model = get_model(task, model_name, device)
    
    # 确保输入图像是RGB格式
    if image.shape[-1] == 3:  # 如果是彩色图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image  # 如果是灰度图像,保持不变
    
    # 处理 classes 参数
    if classes:
        try:
            classes = [int(c.strip()) for c in classes.split(',') if c.strip()]
        except ValueError:
            return None, {"error": "类别过滤参数格式不正确。请使用逗号分隔的整数。"}
    else:
        classes = None
    
    try:
        results = model(image_rgb, device=device, conf=conf, iou=iou, max_det=max_det, classes=classes, agnostic_nms=agnostic_nms, augment=augment)
    
        plot_result = None
        json_result = None
        
        if task in ["检测", "分割", "姿态", "OBB"]:
            # 修改标签为中文
            for r in results:
                r.names = {k: translate_label(v) for k, v in r.names.items()}
                
                # 对于检测任务,我们需要更新每个检测框的标签
                if task == "检测":
                    for box in r.boxes:
                        if box.cls.numel() > 0:
                            class_id = int(box.cls[0])
                            box.label = translate_label(r.names[class_id])
            
            plot_result = results[0].plot()
        elif task == "分类":
            names = results[0].names
            probs = results[0].probs.data.tolist()
            json_result = {translate_label(names[i]): probs[i] for i in range(len(names))}
        
        if plot_result is not None:
            if isinstance(plot_result, torch.Tensor):
                plot_result = plot_result.cpu().numpy()
            
            # 确保输出图像是RGB格式
            if plot_result.shape[-1] == 3:
                plot_result = cv2.cvtColor(plot_result, cv2.COLOR_BGR2RGB)
        
        return plot_result, json_result
    except Exception as e:
        return None, {"error": f"预测过程中出现错误: {str(e)}"}

def export_model(format, task, model_name, device, imgsz, batch, half, int8, dynamic, simplify, opset):
    # 导出原模型文件（.pt），直接返回路径，不需要调用 model.export()
    if format == "pt":
        sub = TASK_FOLDER_MAP.get(task)
        if not sub:
            return "❌ 不支持的任务类型", None
        pt_path = os.path.join("models", sub, model_name)
        if not os.path.exists(pt_path):
            return f"❌ 模型文件不存在: {pt_path}", None
        return f"✅ 原始模型文件: {model_name}\n路径: {pt_path}", pt_path

    model = get_model(task, model_name, device)
    export_path = model.export(format=format, imgsz=imgsz, batch=batch, half=half, int8=int8, dynamic=dynamic, simplify=simplify, opset=opset)
    if export_path:
        return f"✅ 模型导出成功！\n导出文件: {export_path}", export_path
    else:
        return "❌ 模型导出失败", None


EXPORTED_EXTS = ('.onnx', '.engine', '.pb', '.tflite', '.mlpackage', '.saved_model',
                 '.ptc', '.pmf', '.json', '.tar', '.gz', '.zip', '.txt', '.cfg', '.xml')


def list_exported_files():
    """扫描 models/ 目录下所有非 .pt 的导出文件，返回 (rows, summary, file_dict)。
    file_dict: {显示名: 文件路径} 供下载组件使用。
    """
    rows = []
    file_dict = {}
    models_dir = "models"
    if not os.path.exists(models_dir):
        return [["—", "—", "—", 0]], "暂无导出文件。", file_dict

    for sub, label in [
        ("detection", "检测"),
        ("segmentation", "分割"),
        ("pose", "姿态"),
        ("classification", "分类"),
        ("obb", "OBB"),
    ]:
        sub_path = os.path.join(models_dir, sub)
        if not os.path.exists(sub_path) or not os.path.isdir(sub_path):
            continue
        try:
            for f in sorted(os.listdir(sub_path)):
                fpath = os.path.join(sub_path, f)
                if not os.path.isfile(fpath):
                    continue
                if f.endswith('.pt'):
                    continue
                ext = os.path.splitext(f)[1].lower()
                # 只列出已知的导出格式文件
                if ext not in EXPORTED_EXTS and not any(f.endswith(e) for e in EXPORTED_EXTS):
                    continue
                size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
                display = f"[{label}] {f}"
                rows.append([label, f, ext.lstrip('.'), size_mb])
                file_dict[display] = fpath
        except Exception:
            continue

    if not rows:
        return [["—", "—", "—", 0]], "暂无导出文件。导出后会自动记录在此处。", file_dict

    total = round(sum(r[3] for r in rows), 2)
    summary = f"共 **{len(rows)}** 个导出文件，总占用 **{total} MB**。"
    return rows, summary, file_dict


# ==================== ONNX → RKNN 转换 ====================
RKNN_OUTPUT_DIR = os.path.join("models", "rknn")


def _uploaded_path(f):
    """兼容 Gradio 上传组件返回的路径字符串 / 文件对象。"""
    if f is None:
        return None
    if isinstance(f, str):
        return f
    return getattr(f, "name", None) or getattr(f, "path", None)


def convert_onnx_to_rknn(onnx_file, target_platform, do_quantize, calib_files):
    """上传的 ONNX 转换为 RKNN。返回 (日志字符串, rknn文件路径或None)。"""
    logs = []

    def log(msg):
        logs.append(str(msg))

    try:
        from rknn.api import RKNN
    except ImportError:
        return ("❌ 当前镜像未安装 rknn-toolkit2，无法转换。\n"
                "请在 Dockerfile 中添加 rknn-toolkit2==2.3.2 后重建镜像。"), None

    onnx_path = _uploaded_path(onnx_file)
    if not onnx_path or not os.path.exists(onnx_path):
        return "❌ 请先上传有效的 ONNX 模型文件。", None
    if not onnx_path.lower().endswith(".onnx"):
        return f"❌ 文件格式不是 ONNX: {os.path.basename(onnx_path)}", None

    os.makedirs(RKNN_OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(onnx_path))[0]
    rknn_path = os.path.join(RKNN_OUTPUT_DIR, f"{base_name}.rknn")

    # 量化校准数据集
    dataset_txt = None
    tmp_dir = None
    if do_quantize:
        calib_paths = [_uploaded_path(c) for c in (calib_files or [])]
        calib_paths = [p for p in calib_paths if p and os.path.exists(p)]
        if not calib_paths:
            return "❌ 启用量化时必须上传校准图片（建议 100 张以上场景图片）。", None
        tmp_dir = tempfile.mkdtemp(prefix="rknn_calib_")
        abs_lines = []
        for src in calib_paths:
            dst = os.path.join(tmp_dir, os.path.basename(src))
            shutil.copy(src, dst)
            abs_lines.append(dst)
        dataset_txt = os.path.join(tmp_dir, "dataset.txt")
        with open(dataset_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(abs_lines))
        log(f"校准图片数量: {len(abs_lines)}")

    log(f"输入文件: {os.path.basename(onnx_path)}")
    log(f"输出文件: models/rknn/{base_name}.rknn")
    log(f"目标平台: {target_platform}")
    log(f"量化: {'是 (INT8)' if do_quantize else '否 (浮点，精度最高)'}")
    log("")

    try:
        rknn = RKNN(verbose=False)
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform=target_platform,
        )

        log("加载 ONNX ...")
        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            rknn.release()
            return "\n".join(logs) + f"\n❌ load_onnx 失败，错误码: {ret}", None

        log("构建 RKNN ...")
        if do_quantize:
            ret = rknn.build(do_quantization=True, dataset=dataset_txt)
        else:
            ret = rknn.build(do_quantization=False)
        if ret != 0:
            rknn.release()
            return "\n".join(logs) + f"\n❌ build 失败，错误码: {ret}", None

        log("导出 RKNN ...")
        ret = rknn.export_rknn(rknn_path)
        rknn.release()
        if ret != 0:
            return "\n".join(logs) + f"\n❌ export_rknn 失败，错误码: {ret}", None

        size_mb = os.path.getsize(rknn_path) / (1024 * 1024)
        log("")
        log(f"✅ 转换完成！文件大小: {size_mb:.1f} MB")
        return "\n".join(logs), rknn_path
    except Exception as e:
        return "\n".join(logs) + f"\n❌ 转换异常: {type(e).__name__}: {e}", None
    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def list_rknn_files():
    """扫描 models/rknn 目录，返回 (rows, summary, file_dict)。"""
    rows = []
    file_dict = {}
    if not os.path.exists(RKNN_OUTPUT_DIR):
        return [["—", "—", 0]], "暂无 RKNN 文件。", file_dict
    for f in sorted(os.listdir(RKNN_OUTPUT_DIR)):
        fpath = os.path.join(RKNN_OUTPUT_DIR, f)
        if os.path.isfile(fpath) and f.lower().endswith(".rknn"):
            size_mb = round(os.path.getsize(fpath) / (1024 * 1024), 2)
            rows.append([f, "rknn", size_mb])
            file_dict[f] = fpath
    if not rows:
        return [["—", "—", 0]], "暂无 RKNN 文件。转换后会显示在此处。", file_dict
    total = round(sum(r[2] for r in rows), 2)
    summary = f"共 **{len(rows)}** 个 RKNN 文件，总占用 **{total} MB**。"
    return rows, summary, file_dict

def process_video(video_path, task, model_name, device, conf, iou, max_det, classes, agnostic_nms, augment):
    try:
        logging.info(f"开始处理视频: {video_path}")

        if isinstance(video_path, dict) and 'name' in video_path:
            video_path = video_path['name']
        elif not isinstance(video_path, str):
            raise ValueError(f"无效的视频路径类型: {type(video_path)}")

        model = get_model(task, model_name, device)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("无法打开视频文件")
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            output_path = temp_file.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if classes:
            try:
                classes = [int(c.strip()) for c in classes.split(',') if c.strip()]
            except ValueError:
                logging.error("类别过滤参数格式不正确。请使用逗号分隔的整数。")
                return None

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb, device=device, conf=conf, iou=iou, max_det=max_det, classes=classes, agnostic_nms=agnostic_nms, augment=augment)
            
            if task in ["检测", "分割", "姿态", "OBB"]:
                # 修改标签为中文
                for r in results:
                    r.names = {k: translate_label(v) for k, v in r.names.items()}
                    
                    # 对于检测任务,我们需要更新每个检测框的标签
                    if task == "检测":
                        for box in r.boxes:
                            if box.cls.numel() > 0:
                                class_id = int(box.cls[0])
                                box.label = translate_label(r.names[class_id])
                
                annotated_frame = results[0].plot()
            else:
                raise ValueError(f"不支持的任务类型: {task}")
            
            annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
            out.write(annotated_frame_bgr)
            frame_count += 1
        
        cap.release()
        out.release()
        logging.info("视频处理完成，开始进行 FFmpeg 编码")
        
        ffmpeg_path = get_ffmpeg_path()
        if not os.path.exists(ffmpeg_path) and not ffmpeg_path.endswith("ffmpeg"):
            raise ValueError(f"FFmpeg 可执行文件不存在: {ffmpeg_path}")

        final_output_path = output_path.replace('.mp4', '_final.mp4')
        
        ffmpeg_command = [
            ffmpeg_path,
            "-i", output_path,
            "-i", video_path,
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "22",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            final_output_path
        ]
        
        subprocess.run(ffmpeg_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        os.remove(output_path)
        
        logging.info("视频处理完成")
        return final_output_path
    except Exception as e:
        logging.error(f"处理视频时发生错误: {str(e)}")
        return None

def on_video_button_click(video_path, task, model_name, device, conf, iou, max_det, classes, agnostic_nms, augment):
    if video_path is None:
        return None, "请上传视频文件。"
    
    if isinstance(video_path, dict) and 'name' in video_path:
        video_path = video_path['name']
    elif not isinstance(video_path, str):
        return None, f"无效的视频路径类型: {type(video_path)}。请确保上传了正确的视频文件。"
    
    # 处理 classes 参数
    if classes:
        try:
            classes = [int(c.strip()) for c in classes.split(',') if c.strip()]
        except ValueError:
            return None, "类别过滤参数格式不正确。请使用逗号分隔的整数。"
    else:
        classes = None
    
    result = process_video(video_path, task, model_name, device, conf, iou, max_det, classes, agnostic_nms, augment)
    if result is None:
        return None, f"视频处理失败: {task}任务出错，请检查控制台输出以获取更多信息。"
    return result, None

def download_model(task, model_name):
    task_folders = {
        "检测": "detection",
        "分割": "segmentation",
        "姿态": "pose",
        "分类": "classification",
        "OBB": "obb"
    }
    task_folder = task_folders.get(task)
    if not task_folder:
        raise ValueError(f"不支持的任务类型: {task}")
    
    model_path = os.path.join("models", task_folder, model_name)
    if not os.path.exists(model_path):
        print(f"正在下载模型: {model_name}")
        temp_model = YOLO(model_name)
        
        # 确保目标文件夹存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 移动下载的模型文件到指定位置
        shutil.move(model_name, model_path)
        
        print(f"模型已保存到: {model_path}")
        
        # 删除可能存在的额外文件
        for file in os.listdir():
            if file.startswith(model_name.split('.')[0]) and file.endswith('.pt'):
                os.remove(file)
    
    return model_path

def update_model_choices(task):
    return gr.Dropdown(choices=get_available_models(task))

# ==================== 辅助函数 ====================

# 任务类型 -> 本地子目录 映射
TASK_FOLDER_MAP = {
    "检测": "detection",
    "分割": "segmentation",
    "姿态": "pose",
    "分类": "classification",
    "OBB": "obb",
}

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


def get_system_info():
    """返回系统状态信息字典。"""
    info = {
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无（仅 CPU）",
        "python_version": platform.python_version(),
        "model_count": 0,
        "dataset_count": 0,
    }

    # 统计已下载模型数量
    models_dir = "models"
    if os.path.exists(models_dir):
        for sub in ["detection", "segmentation", "pose", "classification", "obb"]:
            sub_path = os.path.join(models_dir, sub)
            if os.path.exists(sub_path) and os.path.isdir(sub_path):
                info["model_count"] += len([
                    f for f in os.listdir(sub_path)
                    if f.endswith('.pt') and os.path.isfile(os.path.join(sub_path, f))
                ])

    # 统计数据集数量（my_dataset 下的有效数据集）
    count = 0
    if os.path.exists(DATASETS_ROOT) and os.path.isdir(DATASETS_ROOT):
        try:
            for name in os.listdir(DATASETS_ROOT):
                full = os.path.join(DATASETS_ROOT, name)
                if os.path.isdir(full) and os.path.exists(os.path.join(full, "images")):
                    count += 1
        except Exception:
            pass
    info["dataset_count"] = count
    return info


def preview_dataset(data_path, page=0, page_size=12):
    """预览数据集，返回 (gallery_images, info_text, corrected_page)。"""
    try:
        page = int(page) if page else 0
        page_size = int(page_size) if page_size else 12
        if page < 0:
            page = 0
        if page_size < 1:
            page_size = 12

        if not data_path:
            return [], "请选择数据集。", 0

        images_dir = os.path.join(data_path, 'images')
        if not os.path.exists(images_dir) or not os.path.isdir(images_dir):
            return [], f"错误：图像目录不存在: {images_dir}", 0

        # 收集 images/ 下的图片，包括子目录 (train/val/test)
        all_images = []
        for root, dirs, files in os.walk(images_dir):
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    rel = os.path.relpath(os.path.join(root, f), images_dir)
                    all_images.append(rel)
        all_images = sorted(all_images)

        total = len(all_images)
        if total == 0:
            return [], "未找到图像文件。请确认 images/ 目录下有图片。", 0

        total_pages = (total + page_size - 1) // page_size
        if page >= total_pages:
            page = max(0, total_pages - 1)

        start = page * page_size
        end = start + page_size
        page_images = all_images[start:end]

        gallery = [(os.path.join(images_dir, img), os.path.basename(img)) for img in page_images]

        # 统计标签
        labels_dir = os.path.join(data_path, 'labels')
        label_count = 0
        if os.path.exists(labels_dir) and os.path.isdir(labels_dir):
            for root, dirs, files in os.walk(labels_dir):
                for f in files:
                    if f.endswith('.txt'):
                        label_count += 1

        class_names = get_class_names(data_path)

        info = f"**图像总数**: {total}  |  **标签数**: {label_count}  |  **类别数**: {len(class_names)}\n\n"
        info += f"当前页: **{page + 1} / {total_pages}** （每页 {page_size} 张）\n\n"
        if class_names:
            shown = class_names[:30]
            info += "**类别列表**: " + ", ".join(shown)
            if len(class_names) > 30:
                info += f"  ...（共 {len(class_names)} 类）"
        else:
            info += "**类别列表**: 未找到 classes/classes.txt 文件。"
        return gallery, info, page
    except Exception as e:
        logging.error(f"预览数据集出错: {str(e)}")
        return [], f"预览数据集时出错: {str(e)}", 0


def list_all_models():
    """列出所有已下载的本地模型，返回 [(task_type, model_name, file_size_mb), ...]。"""
    result = []
    models_dir = "models"
    if not os.path.exists(models_dir):
        return result
    for sub, label in [
        ("detection", "检测"),
        ("segmentation", "分割"),
        ("pose", "姿态"),
        ("classification", "分类"),
        ("obb", "OBB"),
    ]:
        sub_path = os.path.join(models_dir, sub)
        if os.path.exists(sub_path) and os.path.isdir(sub_path):
            try:
                for f in os.listdir(sub_path):
                    if f.endswith('.pt') and os.path.isfile(os.path.join(sub_path, f)):
                        size_mb = round(os.path.getsize(os.path.join(sub_path, f)) / (1024 * 1024), 2)
                        result.append((label, f, size_mb))
            except Exception:
                continue
    return result


def delete_model_func(task_type, model_name):
    """删除指定模型文件，返回操作结果字符串。"""
    if not task_type or not model_name:
        return "请选择要删除的任务类型和模型。"
    sub = TASK_FOLDER_MAP.get(task_type)
    if not sub:
        return f"不支持的任务类型: {task_type}"
    file_path = os.path.join("models", sub, model_name)
    if not os.path.exists(file_path):
        return f"模型文件不存在: {file_path}（可能是在线模型，无法删除。）"
    try:
        # 从内存缓存中移除
        for key in list(models.keys()):
            if key[0] == task_type and key[1] == model_name:
                del models[key]
        os.remove(file_path)
        return f"已成功删除模型: {model_name} （{task_type}）"
    except Exception as e:
        return f"删除模型失败: {str(e)}"


def import_model_func(task_type, uploaded_files, custom_name=""):
    """
    导入上传的 .pt 模型文件到对应目录。
    uploaded_files: gr.File(file_count="multiple") 返回的列表，每项为 dict 或 字符串路径。
    custom_name: 可选自定义文件名（仅单个文件时生效），不带扩展名。
    返回操作结果字符串。
    """
    if not task_type:
        return "请先选择任务类型。"
    sub = TASK_FOLDER_MAP.get(task_type)
    if not sub:
        return f"不支持的任务类型: {task_type}"
    if not uploaded_files:
        return "请上传至少一个 .pt 模型文件。"

    target_dir = os.path.join("models", sub)
    os.makedirs(target_dir, exist_ok=True)

    success_msgs = []
    fail_msgs = []

    single_file = len(uploaded_files) == 1

    for idx, item in enumerate(uploaded_files):
        # gr.File 返回的可能是 dict {name: path, ...} 或字符串路径
        if isinstance(item, dict):
            src = item.get("name") or item.get("path") or str(item)
        else:
            src = str(item)

        if not os.path.exists(src):
            fail_msgs.append(f"[文件{idx+1}] 找不到源文件: {src}")
            continue
        if not src.lower().endswith(".pt"):
            fail_msgs.append(f"[文件{idx+1}] 仅支持 .pt 格式: {os.path.basename(src)}")
            continue

        # 确定目标文件名
        if single_file and custom_name.strip():
            safe_name = custom_name.strip()
            if not safe_name.lower().endswith(".pt"):
                safe_name += ".pt"
        else:
            safe_name = os.path.basename(src)

        dst = os.path.join(target_dir, safe_name)
        try:
            # 避免覆盖同名文件，如已存在则加后缀
            if os.path.exists(dst):
                stem, ext = os.path.splitext(safe_name)
                i = 1
                while os.path.exists(os.path.join(target_dir, f"{stem}_{i}{ext}")):
                    i += 1
                safe_name = f"{stem}_{i}{ext}"
                dst = os.path.join(target_dir, safe_name)

            shutil.copy2(src, dst)
            size_mb = round(os.path.getsize(dst) / (1024 * 1024), 2)
            success_msgs.append(f"✅ {safe_name} ({size_mb} MB)")
        except Exception as e:
            fail_msgs.append(f"[文件{idx+1}] {os.path.basename(src)} 导入失败: {str(e)}")

    parts = []
    if success_msgs:
        parts.append("导入成功:\n" + "\n".join(success_msgs))
    if fail_msgs:
        parts.append("导入失败:\n" + "\n".join(fail_msgs))
    return "\n\n".join(parts) if parts else "未处理任何文件。"


# ==================== 数据集管理辅助函数 ====================

DATASETS_ROOT = "my_dataset"

IMAGE_EXTS_DS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff')


def get_dataset_choices():
    """列出 my_dataset/ 下所有有效数据集，返回 (choices_list, value)。"""
    choices = []
    if not os.path.exists(DATASETS_ROOT):
        return [], None
    try:
        for name in sorted(os.listdir(DATASETS_ROOT)):
            full = os.path.join(DATASETS_ROOT, name)
            if os.path.isdir(full) and os.path.exists(os.path.join(full, "images")):
                choices.append((name, full))
    except Exception:
        pass
    if not choices:
        return [], None
    return choices, choices[0][1]


def get_dataset_yaml_choices():
    """列出有 dataset.yaml 的数据集，返回 (choices_list, value)。"""
    choices = []
    if not os.path.exists(DATASETS_ROOT):
        return [], None
    try:
        for name in sorted(os.listdir(DATASETS_ROOT)):
            full = os.path.join(DATASETS_ROOT, name)
            yaml_path = os.path.join(full, "dataset.yaml")
            if os.path.isdir(full) and os.path.exists(yaml_path):
                choices.append((name, yaml_path))
    except Exception:
        pass
    if not choices:
        return [], None
    return choices, choices[0][1]


def list_datasets_info():
    """返回数据集列表 (dataframe rows, summary markdown)。"""
    rows = []
    if not os.path.exists(DATASETS_ROOT):
        return [["—", 0, 0, "—"]], "暂无数据集。可在下方导入。"
    try:
        for name in sorted(os.listdir(DATASETS_ROOT)):
            full = os.path.join(DATASETS_ROOT, name)
            if not os.path.isdir(full):
                continue
            img_dir = os.path.join(full, "images")
            lbl_dir = os.path.join(full, "labels")
            img_count = len([f for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTS_DS)]) if os.path.exists(img_dir) else 0
            lbl_count = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')]) if os.path.exists(lbl_dir) else 0
            cls_file = os.path.join(full, "classes", "classes.txt")
            classes = get_class_names(full) if os.path.exists(cls_file) else []
            cls_str = ", ".join(classes[:5]) + ("..." if len(classes) > 5 else "") if classes else "—"
            rows.append([name, img_count, lbl_count, cls_str])
    except Exception:
        pass
    if not rows:
        return [["—", 0, 0, "—"]], "暂无数据集。可在下方导入。"
    summary = f"共找到 **{len(rows)}** 个数据集。"
    return rows, summary


def import_dataset_func(dataset_name, image_files, label_files, classes_file):
    """
    导入数据集到 my_dataset/{dataset_name}/ 下。
    自动创建 images/、labels/、classes/ 子目录。
    返回 (msg, ds_choices, ds_value, yaml_choices, yaml_value, rows, summary)
    """
    if not dataset_name or not dataset_name.strip():
        return "请输入数据集名称。", [], None, [], None, [["—", 0, 0, "—"]], "暂无数据集。"
    safe_name = dataset_name.strip().replace("/", "_").replace("\\", "_")
    ds_path = os.path.join(DATASETS_ROOT, safe_name)
    img_dir = os.path.join(ds_path, "images")
    lbl_dir = os.path.join(ds_path, "labels")
    cls_dir = os.path.join(ds_path, "classes")

    if os.path.exists(ds_path) and os.listdir(ds_path):
        return f"数据集 '{safe_name}' 已存在且非空，请使用其他名称或先删除。", [], None, [], None, [["—", 0, 0, "—"]], "暂无数据集。"

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    os.makedirs(cls_dir, exist_ok=True)

    success, fail = [], []

    # 复制图片
    if image_files:
        for item in image_files:
            src = item.get("name") if isinstance(item, dict) else str(item)
            if not src or not os.path.exists(src):
                continue
            fname = os.path.basename(src)
            try:
                shutil.copy2(src, os.path.join(img_dir, fname))
                success.append(f"图片: {fname}")
            except Exception as e:
                fail.append(f"图片 {fname}: {str(e)}")

    # 复制标签
    if label_files:
        for item in label_files:
            src = item.get("name") if isinstance(item, dict) else str(item)
            if not src or not os.path.exists(src):
                continue
            fname = os.path.basename(src)
            try:
                shutil.copy2(src, os.path.join(lbl_dir, fname))
                success.append(f"标签: {fname}")
            except Exception as e:
                fail.append(f"标签 {fname}: {str(e)}")

    # 复制 classes.txt
    if classes_file:
        src = classes_file.get("name") if isinstance(classes_file, dict) else str(classes_file)
        if src and os.path.exists(src):
            try:
                shutil.copy2(src, os.path.join(cls_dir, "classes.txt"))
                success.append("类别文件: classes.txt")
            except Exception as e:
                fail.append(f"classes.txt: {str(e)}")

    # 刷新下拉框和列表
    ds_choices, ds_value = get_dataset_choices()
    yaml_choices, yaml_value = get_dataset_yaml_choices()
    rows, summary = list_datasets_info()

    parts = []
    if success:
        parts.append(f"✅ 成功导入 {len(success)} 个文件:\n" + "\n".join(success[:10]) + ("..." if len(success) > 10 else ""))
    if fail:
        parts.append(f"❌ 失败 {len(fail)} 个:\n" + "\n".join(fail[:10]))
    if not success and not fail:
        parts.append("未导入任何文件（请上传图片和标签）。")
    msg = "\n\n".join(parts)
    return msg, ds_choices, ds_value, yaml_choices, yaml_value, rows, summary


def import_dataset_zip(zip_file):
    """
    从压缩包导入数据集。支持 .zip / .tar.gz / .tgz / .tar。
    支持的压缩包结构：
      1) 顶层目录 → images/ + labels/ + classes.txt
      2) 顶层目录 → images/train、images/val + labels/train、labels/val + classes.txt
    返回 (msg, ds_choices, ds_value, yaml_choices, yaml_value, rows, summary)
    """
    if not zip_file:
        return "请先上传压缩包。", [], None, [], None, [["—", 0, 0, "—"]], "暂无数据集。"

    src = zip_file.get("name") if isinstance(zip_file, dict) else str(zip_file)
    if not src or not os.path.exists(src):
        return "压缩包文件无效。", [], None, [], None, [["—", 0, 0, "—"]], "暂无数据集。"

    fname = os.path.basename(src).lower()
    if not (fname.endswith(".zip") or fname.endswith(".tar.gz") or fname.endswith(".tgz") or fname.endswith(".tar")):
        return "仅支持 .zip / .tar.gz / .tgz / .tar 格式。", [], None, [], None, [["—", 0, 0, "—"]], "暂无数据集。"

    def _safe_extract(archive, dest):
        """安全解压：过滤绝对路径和父目录跳转。"""
        abs_dest = os.path.abspath(dest)
        for member in archive.getmembers() if hasattr(archive, "getmembers") else archive.infolist():
            member_path = member.name if hasattr(member, "name") else member.filename
            abs_member = os.path.abspath(os.path.join(dest, member_path))
            if not abs_member.startswith(abs_dest):
                continue
            try:
                archive.extract(member, dest)
            except Exception:
                pass

    # 创建临时目录解压
    tmp_dir = tempfile.mkdtemp(prefix="yolo_import_")
    try:
        if fname.endswith(".zip"):
            with zipfile.ZipFile(src, "r") as zf:
                _safe_extract(zf, tmp_dir)
        else:
            with tarfile.open(src, "r:*") as tf:
                _safe_extract(tf, tmp_dir)

        # 查找 images / labels / classes 路径
        img_root, lbl_root, found_classes = None, None, None
        found_sub_imgs, found_sub_lbls = {}, {}

        for root, dirs, files in os.walk(tmp_dir):
            dirname = os.path.basename(root).lower()
            if dirname == "images":
                # 过滤非图片子目录
                img_sub_dirs = [d for d in dirs if d.lower() in ("train", "val", "test", "valid")]
                if not img_sub_dirs:
                    img_root = root
                for sub in img_sub_dirs:
                    sub_path = os.path.join(root, sub)
                    if os.path.isdir(sub_path):
                        found_sub_imgs[sub] = sub_path
            if dirname == "labels":
                lbl_sub_dirs = [d for d in dirs if d.lower() in ("train", "val", "test", "valid")]
                if not lbl_sub_dirs:
                    lbl_root = root
                for sub in lbl_sub_dirs:
                    sub_path = os.path.join(root, sub)
                    if os.path.isdir(sub_path):
                        found_sub_lbls[sub] = sub_path
            for f in files:
                if f.lower() == "classes.txt":
                    found_classes = os.path.join(root, f)
            # 也兼容顶层即 classes.txt 的情况
            if dirname == "dataset" or root == tmp_dir:
                for f in files:
                    if f.lower() == "classes.txt":
                        found_classes = os.path.join(root, f)

        # 自动确定数据集名称
        top_items = os.listdir(tmp_dir)
        ds_name = None
        for item in top_items:
            item_path = os.path.join(tmp_dir, item)
            if os.path.isdir(item_path) and not item.startswith("."):
                try:
                    sub_dirs = os.listdir(item_path)
                    if "images" in sub_dirs or "labels" in sub_dirs:
                        ds_name = item
                        break
                except Exception:
                    pass
        if not ds_name:
            base = os.path.splitext(os.path.basename(src))[0]
            if base.endswith(".tar"):
                base = base[:-4]
            ds_name = base

        # 如果数据集目录已存在，加后缀
        safe_name = ds_name.replace("/", "_").replace("\\", "_").strip()
        if not safe_name:
            safe_name = "dataset"
        final_name = safe_name
        counter = 1
        while os.path.exists(os.path.join(DATASETS_ROOT, final_name)):
            final_name = f"{safe_name}_{counter}"
            counter += 1

        ds_path = os.path.join(DATASETS_ROOT, final_name)
        os.makedirs(ds_path, exist_ok=True)

        # 判断目录结构类型
        is_split_structure = bool(found_sub_imgs or found_sub_lbls)
        success, fail = [], []

        if is_split_structure:
            # 保持 train/val/test 子目录结构
            for split_name, img_src in found_sub_imgs.items():
                img_dst = os.path.join(ds_path, "images", split_name)
                os.makedirs(img_dst, exist_ok=True)
                for fname in os.listdir(img_src):
                    fpath = os.path.join(img_src, fname)
                    if os.path.isfile(fpath) and fname.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
                    ):
                        try:
                            shutil.copy2(fpath, os.path.join(img_dst, fname))
                            success.append(f"图片[{split_name}]: {fname}")
                        except Exception as e:
                            fail.append(f"图片[{split_name}] {fname}: {str(e)}")
            for split_name, lbl_src in found_sub_lbls.items():
                lbl_dst = os.path.join(ds_path, "labels", split_name)
                os.makedirs(lbl_dst, exist_ok=True)
                for fname in os.listdir(lbl_src):
                    fpath = os.path.join(lbl_src, fname)
                    if os.path.isfile(fpath) and fname.lower().endswith(".txt"):
                        try:
                            shutil.copy2(fpath, os.path.join(lbl_dst, fname))
                            success.append(f"标签[{split_name}]: {fname}")
                        except Exception as e:
                            fail.append(f"标签[{split_name}] {fname}: {str(e)}")
        else:
            # 扁平结构（把 images/labels 下所有文件复制到对应位置）
            if img_root and os.path.isdir(img_root):
                img_dst = os.path.join(ds_path, "images")
                os.makedirs(img_dst, exist_ok=True)
                for fname in os.listdir(img_root):
                    fpath = os.path.join(img_root, fname)
                    if os.path.isfile(fpath) and fname.lower().endswith(
                        (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
                    ):
                        try:
                            shutil.copy2(fpath, os.path.join(img_dst, fname))
                            success.append(f"图片: {fname}")
                        except Exception as e:
                            fail.append(f"图片 {fname}: {str(e)}")
            if lbl_root and os.path.isdir(lbl_root):
                lbl_dst = os.path.join(ds_path, "labels")
                os.makedirs(lbl_dst, exist_ok=True)
                for fname in os.listdir(lbl_root):
                    fpath = os.path.join(lbl_root, fname)
                    if os.path.isfile(fpath) and fname.lower().endswith(".txt"):
                        try:
                            shutil.copy2(fpath, os.path.join(lbl_dst, fname))
                            success.append(f"标签: {fname}")
                        except Exception as e:
                            fail.append(f"标签 {fname}: {str(e)}")

        if found_classes:
            cls_dir = os.path.join(ds_path, "classes")
            os.makedirs(cls_dir, exist_ok=True)
            try:
                shutil.copy2(found_classes, os.path.join(cls_dir, "classes.txt"))
                success.append("类别文件: classes.txt")
            except Exception as e:
                fail.append(f"classes.txt: {str(e)}")

        if not success:
            shutil.rmtree(ds_path, ignore_errors=True)
            return "压缩包中未找到有效的图片或标签文件。", [], None, [], None, [["—", 0, 0, "—"]], "暂无数据集。"

        # 刷新
        ds_choices, ds_value = get_dataset_choices()
        yaml_choices, yaml_value = get_dataset_yaml_choices()
        rows, summary = list_datasets_info()

        parts = [f"📦 压缩包解压导入完成，数据集名称: **{final_name}**"]
        parts.append(f"✅ 成功导入 {len(success)} 个文件:\n" + "\n".join(success[:15]) + ("..." if len(success) > 15 else ""))
        if fail:
            parts.append(f"❌ 失败 {len(fail)} 个:\n" + "\n".join(fail[:10]))
        msg = "\n\n".join(parts)
        return msg, ds_choices, ds_value, yaml_choices, yaml_value, rows, summary

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def delete_dataset_func(dataset_path):
    """删除整个数据集文件夹。返回 (msg, ds_choices, ds_value, yaml_choices, yaml_value, rows, summary)"""
    if not dataset_path:
        return "请选择要删除的数据集。", [], None, [], None, [["—", 0, 0, "—"]], "暂无数据集。"
    if not dataset_path.startswith(DATASETS_ROOT):
        return "仅允许删除 my_dataset 下的数据集。", [], None, [], None, [["—", 0, 0, "—"]], "暂无数据集。"
    try:
        shutil.rmtree(dataset_path)
        ds_choices, ds_value = get_dataset_choices()
        yaml_choices, yaml_value = get_dataset_yaml_choices()
        rows, summary = list_datasets_info()
        return f"已删除数据集: {os.path.basename(dataset_path)}", ds_choices, ds_value, yaml_choices, yaml_value, rows, summary
    except Exception as e:
        return f"删除失败: {str(e)}", [], None, [], None, [["—", 0, 0, "—"]], "暂无数据集。"


def _train_in_thread(tid, data_path, epochs, task, model_name, device, include_test, test_size, val_size,
                     batch_size, imgsz, patience, optimizer, lr0, lrf, momentum, weight_decay,
                     warmup_epochs, warmup_momentum, warmup_bias_lr, box, cls, dfl, pose, kobj,
                     label_smoothing, nbs, overlap_mask, mask_ratio, dropout,
                     augment, mixup, copy_paste, mosaic, amp):
    """在子线程中执行训练，结束后更新全局状态和任务状态。"""
    global _training_active, _training_result
    result_msg = None
    try:
        result_msg = train_model(
            data_path, epochs, task, model_name, device, include_test, test_size, val_size,
            batch_size, imgsz, patience, optimizer, lr0, lrf, momentum, weight_decay,
            warmup_epochs, warmup_momentum, warmup_bias_lr, box, cls, dfl, pose, kobj,
            label_smoothing, nbs, overlap_mask, mask_ratio, dropout,
            augment, mixup, copy_paste, mosaic, amp
        )
        _training_result = result_msg
        stopped = _training_stop_flag.is_set()
        final_status = "stopped" if (stopped and result_msg and "停止" in result_msg) else (
            "failed" if result_msg and ("错误" in result_msg or "失败" in result_msg or "出错" in result_msg) else "completed"
        )
        _update_task(tid, status=final_status, end_at=time.time(), result_msg=result_msg)
    except Exception as e:
        result_msg = f"训练出错: {e}"
        _training_result = result_msg
        _update_task(tid, status="failed", end_at=time.time(), result_msg=result_msg)
    finally:
        with _training_lock:
            _training_active = False


def train_model_with_log(data_path, epochs, task, model_name, device, include_test, test_size, val_size,
                         batch_size, imgsz, patience, optimizer, lr0, lrf, momentum, weight_decay,
                         warmup_epochs, warmup_momentum, warmup_bias_lr, box, cls, dfl, pose, kobj,
                         label_smoothing, nbs, overlap_mask, mask_ratio, dropout,
                         augment, mixup, copy_paste, mosaic, amp, _schedule_ts=None):
    """启动训练线程，实时 yield 训练日志。_schedule_ts 为预约时间戳（None=立即训练）。"""
    global _training_active, _training_all_logs, _training_result

    # 预约模式：如果传入预约时间且未到，则创建任务并在预约时启动
    if _schedule_ts and _schedule_ts > time.time():
        tid = _add_task("scheduled", task, model_name, data_path, epochs, schedule_at=_schedule_ts)

        def _scheduled_launcher():
            global _training_active, _training_all_logs, _training_result
            # 到了预约时间再尝试启动训练
            wait_for = _schedule_ts - time.time()
            if wait_for > 0:
                time.sleep(wait_for)
            # 再次检查是否已有训练在运行
            with _training_lock:
                if _training_active:
                    _update_task(tid, status="failed", start_at=time.time(), end_at=time.time(),
                                 result_msg="预约启动时已有训练任务运行，已跳过。")
                    return
                _training_active = True
            # 清空旧日志和状态
            _training_stop_flag.clear()
            _training_all_logs = []
            _training_result = None
            while not _training_log_queue.empty():
                _training_log_queue.get()
            # 恢复为立即执行
            _update_task(tid, status="running", start_at=time.time())
            # 启动训练子线程
            t = threading.Thread(target=_train_in_thread, args=(
                tid, data_path, epochs, task, model_name, device, include_test, test_size, val_size,
                batch_size, imgsz, patience, optimizer, lr0, lrf, momentum, weight_decay,
                warmup_epochs, warmup_momentum, warmup_bias_lr, box, cls, dfl, pose, kobj,
                label_smoothing, nbs, overlap_mask, mask_ratio, dropout,
                augment, mixup, copy_paste, mosaic, amp
            ), daemon=True)
            t.start()

        timer = threading.Timer(0, _scheduled_launcher)
        timer.daemon = True
        with _training_lock:
            _schedule_timers[tid] = timer
        timer.start()

        dt = datetime.datetime.fromtimestamp(_schedule_ts).strftime("%Y-%m-%d %H:%M:%S")
        yield f"📅 已创建预约训练任务 #{tid}\n预约时间：{dt}\n参数：{task} · {model_name} · {epochs} epochs\n\n到时间后将自动启动（当前训练结束后会自动检测并执行；若已设置的任务不需要可在训练 Tab 取消预约）。"
        return

    # 防止重复启动训练
    with _training_lock:
        if _training_active:
            yield "⚠️ 已有训练正在运行，请先停止当前训练。"
            return
        _training_active = True

    # 创建任务记录
    tid = _add_task("running", task, model_name, data_path, epochs)

    _training_stop_flag.clear()
    _training_all_logs = []
    _training_result = None
    while not _training_log_queue.empty():
        _training_log_queue.get()

    yield "⏳ [1/4] 正在检查数据集..."
    yield "⏳ [2/4] 正在分割数据集（train/val/test）..."
    yield "⏳ [3/4] 正在加载模型..."
    yield f"🚀 [4/4] 开始训练（任务 #{tid}）... 可点击「⏹️ 停止训练」中断。"

    # 启动训练子线程
    thread = threading.Thread(target=_train_in_thread, args=(
        tid, data_path, epochs, task, model_name, device, include_test, test_size, val_size,
        batch_size, imgsz, patience, optimizer, lr0, lrf, momentum, weight_decay,
        warmup_epochs, warmup_momentum, warmup_bias_lr, box, cls, dfl, pose, kobj,
        label_smoothing, nbs, overlap_mask, mask_ratio, dropout,
        augment, mixup, copy_paste, mosaic, amp
    ), daemon=True)
    thread.start()

    # 轮询队列，实时 yield 日志，并更新任务 current_epoch
    while True:
        with _training_lock:
            still_active = _training_active
        if not still_active and _training_log_queue.empty():
            break
        try:
            line = _training_log_queue.get(timeout=1.0)
            _training_all_logs.append(line)
            # 解析 Epoch x/y 更新 current_epoch
            if line.startswith("Epoch "):
                try:
                    cur = int(line.split("Epoch ")[1].split("/")[0])
                    _update_task(tid, current_epoch=cur)
                except Exception:
                    pass
        except queue_module.Empty:
            pass
        yield "\n".join(_training_all_logs[-200:])

    # 输出最终结果
    if _training_result:
        yield "\n".join(_training_all_logs[-200:]) + "\n\n" + _training_result


def cancel_scheduled_task(task_id_str):
    """取消预约任务。task_id_str 为字符串的 #id。"""
    if not task_id_str:
        return "⚠️ 请选择要取消的预约任务。"
    try:
        tid = int(str(task_id_str).lstrip("#"))
    except Exception:
        return "⚠️ 无效的任务编号。"
    t = _update_task(tid)
    if not t:
        return "⚠️ 任务不存在。"
    if t["status"] != "scheduled":
        return f"⚠️ 当前任务状态为「{t['status']}」，不能取消。"
    # 关闭 timer（如果存在）
    with _training_lock:
        timer = _schedule_timers.pop(tid, None)
    if timer:
        try:
            timer.cancel()
        except Exception:
            pass
    _update_task(tid, status="stopped", start_at=time.time(), end_at=time.time(), result_msg="用户取消预约训练。")
    return f"✅ 已取消预约任务 #{tid}。"


def get_scheduled_choices():
    """返回当前可取消的预约任务列表（下拉框选项）。"""
    with _training_lock:
        choices = [f"#{t['id']}  ({t['task']} · {t['model']} · {_fmt_time(t['schedule_at'])})"
                   for t in _tasks if t["status"] == "scheduled"]
    value = choices[0] if choices else None
    return choices, value


def get_training_status():
    """获取当前训练状态和日志（供页面刷新恢复使用）。"""
    with _training_lock:
        active = _training_active
        logs = list(_training_all_logs[-100:])
        result = _training_result
    if active:
        header = "🔄 训练进行中...\n\n"
        body = "\n".join(logs) if logs else "（等待日志输出...）"
        return header + body
    elif result:
        body = "\n".join(logs) if logs else ""
        return body + ("\n\n" if body and body != result else "") + result
    else:
        return ""


def list_training_runs():
    """扫描 runs/ 目录，列出所有训练结果。返回 (choices, rows, summary)。"""
    runs_dir = "runs"
    if not os.path.exists(runs_dir):
        return [], [["—", "—", "—", "—"]], "暂无训练结果。"

    results = []
    for task_type in os.listdir(runs_dir):
        task_path = os.path.join(runs_dir, task_type)
        if not os.path.isdir(task_path):
            continue
        for run_name in os.listdir(task_path):
            run_path = os.path.join(task_path, run_name)
            if not os.path.isdir(run_path):
                continue
            # 统计文件
            file_count = 0
            total_size = 0
            has_weights = False
            for root, dirs, files in os.walk(run_path):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        total_size += os.path.getsize(fp)
                        file_count += 1
                        if f.endswith('.pt'):
                            has_weights = True
                    except Exception:
                        pass
            size_mb = round(total_size / (1024 * 1024), 2)
            mtime = os.path.getmtime(run_path)
            time_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            label = f"{task_type}/{run_name}"
            results.append((label, time_str, file_count, size_mb, has_weights, run_path))

    if not results:
        return [], [["—", "—", "—", "—"]], "暂无训练结果。"

    results.sort(key=lambda x: x[1], reverse=True)
    choices = [r[0] for r in results]
    rows = [[r[0], r[1], r[2], r[3]] for r in results]
    total = round(sum(r[3] for r in results), 2)
    summary = f"共 {len(results)} 次训练，总占用 {total} MB。"
    return choices, rows, summary


def get_run_details(run_label):
    """获取指定训练结果的文件列表和预览图。返回 (file_rows, gallery, info)。"""
    if not run_label:
        return [["—", "—"]], [], "请选择训练结果。"
    run_path = os.path.join("runs", run_label)
    if not os.path.exists(run_path):
        return [["—", "—"]], [], f"路径不存在: {run_path}"

    file_rows = []
    gallery = []
    for root, dirs, files in os.walk(run_path):
        for f in sorted(files):
            fp = os.path.join(root, f)
            rel_path = os.path.relpath(fp, run_path)
            try:
                size_kb = round(os.path.getsize(fp) / 1024, 1)
            except Exception:
                size_kb = 0
            file_rows.append([rel_path, f"{size_kb} KB"])
            # 图片文件加入 gallery 预览
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                gallery.append((fp, rel_path))

    info = f"路径: {run_path}\n文件数: {len(file_rows)}"
    if gallery:
        info += f"\n可预览图片: {len(gallery)} 张"
    return file_rows, gallery, info


def package_training_run(run_label):
    """将训练结果打包为 zip 并返回文件路径。"""
    if not run_label:
        return None, "请选择要打包的训练结果。"
    run_path = os.path.join("runs", run_label)
    if not os.path.exists(run_path):
        return None, f"路径不存在: {run_path}"

    zip_name = run_label.replace("/", "_") + ".zip"
    zip_path = os.path.join(tempfile.gettempdir(), zip_name)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(run_path):
            for f in files:
                fp = os.path.join(root, f)
                arcname = os.path.relpath(fp, run_path)
                zf.write(fp, arcname)
    return zip_path, f"✅ 已打包 {run_label} → {zip_name}，点击下方下载。"


def list_run_files_for_download(run_label):
    """返回训练结果中可单独下载的文件列表（下拉框选项）。"""
    if not run_label:
        return [], None
    run_path = os.path.join("runs", run_label)
    if not os.path.exists(run_path):
        return [], None
    choices = []
    for root, dirs, files in os.walk(run_path):
        for f in sorted(files):
            fp = os.path.join(root, f)
            rel_path = os.path.relpath(fp, run_path)
            choices.append(rel_path)
    return choices, (choices[0] if choices else None)


def predict_batch(images, task, model_name, device, conf, iou, max_det, classes, agnostic_nms, augment):
    """批量预测多张图片，返回 (gallery_results, json_results)。"""
    gallery_results = []
    json_results = []
    if not images:
        return [], {"信息": "未提供图像。"}

    for img_item in images:
        # gr.File 返回的可能是字符串路径或带 name 的对象
        if isinstance(img_item, dict):
            path = img_item.get('name', str(img_item))
        else:
            path = img_item

        try:
            image = cv2.imread(path)
            if image is None:
                json_results.append({os.path.basename(path): {"error": "无法读取图像"}})
                continue

            plot_result, json_result = predict_image(
                image, task, model_name, device, conf, iou, max_det,
                classes, agnostic_nms, augment
            )
            caption = os.path.basename(path)

            if plot_result is not None:
                gallery_results.append((plot_result, caption))
            else:
                orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if isinstance(json_result, dict) and "error" in json_result:
                    caption += f" - 错误: {json_result['error']}"
                else:
                    caption += " - 无可视化结果"
                gallery_results.append((orig_rgb, caption))

            json_results.append({caption: json_result})
        except Exception as e:
            json_results.append({os.path.basename(path): {"error": str(e)}})

    if not gallery_results:
        return [], json_results if json_results else {"信息": "未生成任何结果。"}
    return gallery_results, json_results


# ==================== UI 辅助函数 ====================

def build_stat_cards(info):
    """生成首页状态卡片 HTML。"""
    cuda_status = "✅ 可用" if info["cuda_available"] else "❌ 不可用"
    cards = f"""
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:16px;margin:16px 0;">
      <div class="stat-card">
        <div class="stat-value">{info['model_count']}</div>
        <div class="stat-label">已下载模型</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{info['dataset_count']}</div>
        <div class="stat-label">数据集数量</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{cuda_status}</div>
        <div class="stat-label">CUDA 加速</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{info['torch_version']}</div>
        <div class="stat-label">PyTorch 版本</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{info['gpu_name']}</div>
        <div class="stat-label">GPU 设备</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{info['python_version']}</div>
        <div class="stat-label">Python 版本</div>
      </div>
    </div>
    """
    return cards


def refresh_home():
    """刷新首页系统状态，返回 (cards_html, detail_md)。"""
    info = get_system_info()
    cards = build_stat_cards(info)
    cuda_status = "✅ 可用" if info["cuda_available"] else "❌ 不可用"
    detail = f"""### 系统详细信息

| 项目 | 状态 |
|------|------|
| CUDA 加速 | {cuda_status} |
| PyTorch 版本 | `{info['torch_version']}` |
| GPU 设备 | {info['gpu_name']} |
| Python 版本 | `{info['python_version']}` |
| 已下载模型 | {info['model_count']} 个 |
| 数据集数量 | {info['dataset_count']} 个 |
"""
    return cards, detail


def preview_dataset_ui(data_path, page):
    """数据集预览 UI 回调，返回 (gallery, info, page)。"""
    page = int(page) if page else 0
    if page < 0:
        page = 0
    gallery, info, corrected_page = preview_dataset(data_path, page, page_size=12)
    return gallery, info, corrected_page


def prev_dataset_page(data_path, page):
    page = int(page) if page else 0
    page = max(0, page - 1)
    gallery, info, corrected_page = preview_dataset(data_path, page, page_size=12)
    return gallery, info, corrected_page


def next_dataset_page(data_path, page):
    page = int(page) if page else 0
    page = page + 1
    gallery, info, corrected_page = preview_dataset(data_path, page, page_size=12)
    # corrected_page 已在 preview_dataset 内部修正为有效页码
    return gallery, info, corrected_page


def check_dataset_ui(data_path):
    """数据集检查 UI 回调，返回 Markdown 文本。"""
    try:
        if not data_path:
            return "⚠️ 请输入数据集路径。"
        if check_dataset(data_path):
            class_names = get_class_names(data_path)
            images_dir = os.path.join(data_path, 'images')
            labels_dir = os.path.join(data_path, 'labels')
            img_count = 0
            lbl_count = 0
            if os.path.exists(images_dir):
                img_count = len([f for f in os.listdir(images_dir)
                                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))
                                 and os.path.isfile(os.path.join(images_dir, f))])
            if os.path.exists(labels_dir):
                lbl_count = len([f for f in os.listdir(labels_dir)
                                 if f.endswith('.txt') and os.path.isfile(os.path.join(labels_dir, f))])
            md = "✅ **数据集检查通过**\n\n"
            md += f"- 图像数量: **{img_count}**\n"
            md += f"- 标签数量: **{lbl_count}**\n"
            md += f"- 类别数量: **{len(class_names)}**\n"
            if class_names:
                md += f"- 类别列表: {', '.join(class_names[:50])}"
                if len(class_names) > 50:
                    md += f"  ...（共 {len(class_names)} 类）"
            return md
        else:
            return "❌ **数据集检查未通过**，请查看控制台日志。确保目录下包含 `images`、`labels` 和 `classes/classes.txt`。"
    except Exception as e:
        return f"❌ 检查出错: {str(e)}"


def get_local_model_choices(task):
    """仅返回本地已下载模型列表（用于模型管理删除下拉框）。"""
    sub = TASK_FOLDER_MAP.get(task)
    if not sub:
        return gr.Dropdown(choices=[])
    sub_path = os.path.join("models", sub)
    if not os.path.exists(sub_path):
        return gr.Dropdown(choices=[])
    choices = sorted([f for f in os.listdir(sub_path)
                      if f.endswith('.pt') and os.path.isfile(os.path.join(sub_path, f))])
    return gr.Dropdown(choices=choices)


def refresh_model_list():
    """刷新模型管理列表，返回 (dataframe_rows, summary_md)。"""
    models_list = list_all_models()
    if not models_list:
        return [["—", "—", 0]], "当前没有已下载的本地模型。可在「训练/预测」等标签页中首次使用模型后自动下载。"
    rows = [[m[0], m[1], m[2]] for m in models_list]
    total_size = round(sum(m[2] for m in models_list), 2)
    return rows, f"共找到 **{len(models_list)}** 个已下载模型，总占用 **{total_size} MB**。"


def init_all_model_choices():
    """初始化各任务下拉框对应的模型下拉框选项（含模型管理删除/导出下拉框）。"""
    upd = update_model_choices("检测")
    mgmt_upd = get_local_model_choices("检测")
    return upd, upd, upd, mgmt_upd, mgmt_upd


# ==================== Gradio界面 ====================

custom_css = """
:root {
    --font: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", "Segoe UI", system-ui, -apple-system, sans-serif;
}
body, .gradio-container {
    font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans SC", "Segoe UI", system-ui, -apple-system, sans-serif !important;
}
.main-header { text-align: center; padding: 8px 0 4px 0; }
.main-header h1 { margin: 0 0 6px 0; font-size: 28px; }
.main-header p { margin: 0; color: var(--body-text-color-subdued); font-size: 14px; }
.header-row { align-items: center; }
.stat-card {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 18px 12px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    min-height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}
.stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.12);
}
.stat-value {
    font-size: 22px;
    font-weight: 700;
    color: var(--color-accent);
    margin-bottom: 4px;
    word-break: break-word;
    line-height: 1.3;
}
.stat-label {
    font-size: 13px;
    color: var(--body-text-color-subdued);
}
.tab-tip {
    color: var(--body-text-color-subdued);
    font-size: 13px;
    margin-bottom: 8px;
}
/* Gallery 可滚动 */
.ds-gallery { overflow-y: auto !important; }
.ds-gallery > div { overflow-y: auto !important; height: 100% !important; }
footer { visibility: hidden; }

/* 首页训练任务面板 */
.task-panel {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 16px;
    background: var(--background-fill-secondary);
    margin-top: 8px;
    max-height: 420px;
    overflow-y: auto;
}
.task-empty {
    color: var(--body-text-color-subdued);
    font-size: 14px;
    padding: 24px;
    text-align: center;
}
.task-row {
    border: 1px solid var(--border-color-primary);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 12px;
    background: var(--body-background-fill);
    transition: border-color 0.2s;
}
.task-row:last-child { margin-bottom: 0; }
.task-row-head {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    flex-wrap: wrap;
}
.task-id {
    font-weight: 700;
    color: var(--body-text-color-subdued);
    font-size: 12px;
}
.task-title {
    font-weight: 600;
    font-size: 15px;
}
.task-row-body {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    color: var(--body-text-color-subdued);
    font-size: 13px;
    margin-bottom: 4px;
}
.task-result {
    margin-top: 6px;
    padding: 8px 12px;
    background: var(--background-fill-secondary);
    border-radius: 6px;
    font-size: 13px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    line-height: 1.6;
}
.badge-scheduled { background: #fff7e0; color: #a57814; border: 1px solid #f4dc8c; }
.badge-running   { background: #e0f2ff; color: #1a6cb7; border: 1px solid #a8d6ff; }
.badge-completed { background: #e3f7e3; color: #2d8537; border: 1px solid #a8dfa8; }
.badge-stopped   { background: #fde7dc; color: #a1521e; border: 1px solid #f5bd9b; }
.badge-failed    { background: #ffe2e2; color: #a03232; border: 1px solid #f6a8a8; }

/* 登录页面样式 */
.login-wrap {
    max-width: 50%;
    margin: 60px auto;
    background: var(--background-fill-primary);
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    padding: 40px 36px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.08);
}
.login-wrap input {
    min-width: 100% !important;
}
.login-header {
    text-align: center;
    margin-bottom: 4px !important;
}
.login-header * {
    font-size: 26px !important;
    font-weight: 700 !important;
    line-height: 1.3 !important;
    margin: 0 !important;
}
.login-desc {
    text-align: center;
}
.login-desc * {
    font-size: 13px !important;
    color: var(--body-text-color-subdued) !important;
    line-height: 1.8 !important;
    margin: 2px 0 !important;
}
/* ===== 文件上传/下载组件紧凑化 ===== */
/* 上传按钮：去掉大图标、压缩高度、精简提示文字（Gradio6 默认 240~320px） */
/* 注意：Dataframe 表格外层也叫 div.upload-container，必须限定 button，否则会压塌所有表格 */
button.upload-container {
    min-height: 46px !important;
    height: 46px !important;
    max-height: 46px !important;
    padding: 4px 12px !important;
    gap: 6px !important;
}
button.upload-container .icon-wrap {
    display: none !important;
}
button.upload-container .or {
    display: none !important;
}
/* 默认长提示是裸文本，用 font-size:0 隐藏，::after 换成精简文案 */
/* 注意：Gradio Dataframe 单元格也含 .wrap，必须用 data-testid 精确锚定上传组件 */
button.upload-container .wrap[data-testid="upload-text"] {
    font-size: 0 !important;
    gap: 0 !important;
    white-space: nowrap !important;
}
button.upload-container .wrap[data-testid="upload-text"]::after {
    content: "📄 点击或拖拽文件到此处上传";
    font-size: 13px !important;
    font-weight: 500 !important;
}
/* 文件预览/下载列表紧凑化 */
.file-preview-holder {
    padding: 2px 0 !important;
    max-height: 150px !important;
    overflow-y: auto !important;
}
.file-preview {
    margin: 0 !important;
}
.file-preview td {
    padding: 3px 8px !important;
}
.file-preview .filename {
    font-size: 13px !important;
}
.file-preview .download-link {
    font-size: 12px !important;
    white-space: nowrap !important;
}
/* 带 file-upload 类的 block：不加滚动条，避免空上传框右侧出现 ▲●▼ 箭头 */
/* 已选文件列表的滚动由 .file-preview-holder 负责 */
.file-upload {
    overflow: visible !important;
}
/* 文件组件 label 与控件间距收紧 */
.block.file > label,
.block.file-upload > label {
    margin-bottom: 2px !important;
    padding-bottom: 0 !important;
}
/* ===== Image/Video 空状态紧凑化 ===== */
/* Gradio 把 height 写在组件 block 内联样式上：默认（空）收缩为一行，检测到 img/video 后撑高 */
.compact-media {
    height: 52px !important;
    min-height: 52px !important;
    transition: height 0.15s ease;
}
.compact-media:has(img) {
    height: 300px !important;
    min-height: 300px !important;
}
.compact-media:has(video) {
    height: 240px !important;
    min-height: 240px !important;
}
/* gr.File 选了文件后出现文件列表，自动撑高（空态仍保持 52px） */
.compact-media:has(.file-preview-holder) {
    height: auto !important;
    min-height: 52px !important;
}
.compact-media .upload-container {
    height: 46px !important;
    min-height: 46px !important;
}
/* 底部 source 图标行（上传/摄像头/剪贴板）与上方重复，直接隐藏 */
.source-selection {
    display: none !important;
}

/* 修复 Group 嵌套导致的灰色背景 */
.gr-group, .group {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.gradio-container .gr-group {
    background: transparent !important;
}
"""

# ==================== 配置加载 ====================
def load_config():
    """读取 config.json 配置文件。"""
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    default_config = {
        "username": "admin",
        "password": "admin123",
        "app_title": "YOLO 视觉训练平台",
        "favicon_path": "",
        "default_theme": "light",
        "session_timeout_minutes": 720
    }
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        default_config.update(cfg)
    except Exception:
        pass
    return default_config

APP_CONFIG = load_config()


def verify_login(username, password):
    """验证登录，返回 (是否成功, 消息)。"""
    if not username or not password:
        return False, "请输入用户名和密码。"
    if username.strip() == APP_CONFIG["username"] and password == APP_CONFIG["password"]:
        return True, "登录成功"
    return False, "用户名或密码错误，请重试。"


def restore_login_state(state_value):
    """从 localStorage 恢复的登录状态值决定页面可见性。"""
    if state_value == "1":
        return gr.update(visible=False), gr.update(visible=True)
    return gr.update(visible=True), gr.update(visible=False)


def login_success(username, password):
    """登录按钮回调：验证后切换可见性，并同步 login_state。"""
    ok, msg = verify_login(username, password)
    if ok:
        return gr.update(visible=False), gr.update(visible=True), "", "1"
    else:
        return gr.update(visible=True), gr.update(visible=False), msg, ""


def logout():
    """退出登录，清理状态。"""
    return gr.update(visible=True), gr.update(visible=False), "", "", ""

# 主题切换 JS：在 html / gradio-app / body 上切换 dark 类
THEME_JS_INIT = """() => {
    const saved = localStorage.getItem('yolo_theme') || 'light';
    const targets = [document.documentElement, document.querySelector('gradio-app'), document.body];
    targets.forEach(t => { if (t) {
        if (saved === 'dark') t.classList.add('dark');
        else t.classList.remove('dark');
    }});
}"""

# 会话超时 JS：1小时无活动自动退出登录
SESSION_TIMEOUT_JS = """() => {
    const TIMEOUT_MS = 60 * 60 * 1000; // 1小时
    const CHECK_INTERVAL = 60 * 1000;  // 每分钟检查一次
    const ACTIVITY_KEY = 'yolo_last_activity';
    const LOGIN_KEY = 'yolo_logged_in';

    function updateActivity() {
        localStorage.setItem(ACTIVITY_KEY, String(Date.now()));
    }

    function checkSession() {
        if (localStorage.getItem(LOGIN_KEY) !== '1') return;
        const last = parseInt(localStorage.getItem(ACTIVITY_KEY) || '0', 10);
        if (last > 0 && (Date.now() - last) > TIMEOUT_MS) {
            localStorage.removeItem(LOGIN_KEY);
            localStorage.removeItem(ACTIVITY_KEY);
            location.reload();
        }
    }

    // 页面加载时先检查一次（防止刷新后仍处于过期状态）
    checkSession();

    // 监听用户活动
    ['mousemove', 'keydown', 'click', 'scroll', 'touchstart'].forEach(evt => {
        document.addEventListener(evt, updateActivity, { passive: true });
    });

    // 定时检查
    setInterval(checkSession, CHECK_INTERVAL);
}"""

theme_toggle_js = """() => {
    const targets = [document.documentElement, document.querySelector('gradio-app'), document.body];
    let isDark = false;
    targets.forEach(t => { if (t && t.classList.contains('dark')) isDark = true; });
    targets.forEach(t => { if (t) t.classList.toggle('dark'); });
    localStorage.setItem('yolo_theme', isDark ? 'light' : 'dark');
}"""

with gr.Blocks(title=APP_CONFIG["app_title"]) as demo:

    # 登录状态同步（隐藏，用于与 localStorage 通信）
    login_state = gr.Textbox(value="", visible=False)

    # ---------- 登录页面 ----------
    with gr.Column(visible=True) as login_group:
        with gr.Column(elem_classes="login-wrap"):
            gr.Markdown(
                f"# {APP_CONFIG['app_title']}",
                elem_classes="login-header"
            )
            gr.Markdown(
                "基于 Ultralytics YOLO11 的视觉智能训练平台  \n"
                "支持模型训练、图片预测、视频推理、模型导出、ONNX→RKNN 转换与数据集管理",
                elem_classes="login-desc"
            )
            gr.HTML("<hr style='border:none;border-top:1px solid var(--border-color-primary);margin:16px 0;'/>")
            gr.Markdown("请登录以继续使用", elem_classes="login-desc")
            
            login_user = gr.Textbox(label="用户名", placeholder="请输入用户名", max_lines=1, scale=3)
                
            login_pass = gr.Textbox(label="密码", placeholder="请输入密码", type="password", max_lines=1)
            login_btn = gr.Button("🔑 登录", variant="primary", size="lg")
            login_error = gr.Markdown("", visible=True)

    # ---------- 主内容区（登录后显示） ----------
    with gr.Column(visible=False) as main_group:
        # ---------- 顶部标题与主题切换 ----------
        gr.Markdown(
            f"# {APP_CONFIG['app_title']}\n"
            "基于 Ultralytics YOLO 的视觉训练平台，支持模型训练、图片预测、视频推理、模型导出与 ONNX→RKNN 转换。",
            elem_classes="main-header"
        )
        with gr.Row(elem_classes="header-row"):
            with gr.Column(scale=20):
                pass
            logout_btn = gr.Button("🚪 退出登录", variant="secondary", size="sm", min_width=120)
            gr.Button("🌓 切换主题", variant="secondary", size="sm", min_width=120).click(fn=None, js=theme_toggle_js)

        # 全局设备选择
        device_options = ["cuda", "cpu"] if cuda_available else ["cpu"]
        with gr.Row():
            device_input = gr.Radio(
                device_options, label="设备选择",
                value="cuda" if cuda_available else "cpu",
                info="选择模型运行设备，CUDA 可获得更快的推理与训练速度。"
            )

        # ==================== 首页 ====================
        with gr.Tab("🏠 首页"):
            # --- 顶部：系统状态 + 刷新 ---
            with gr.Row():
                refresh_home_btn = gr.Button("🔄 刷新", variant="primary", size="sm", scale=1, min_width=80)
            home_cards = gr.HTML(value=build_stat_cards(get_system_info()))

            # --- 训练任务 + 快速开始 并排 ---
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### 🧪 训练任务状态")
                    with gr.Row():
                        refresh_tasks_btn = gr.Button("🔄 刷新任务", variant="secondary", size="sm", min_width=80)
                    home_tasks_html = gr.HTML(value=_list_tasks_html())
                with gr.Column(scale=2):
                    gr.Markdown("### 📌 快速开始")
                    gr.Markdown(
                        "**1. 🎯 训练**  \n"
                        "选择数据集与模型，配置参数后**立即训练**或**预约训练**。\n\n"
                        "**2. 🖼️ 预测**  \n"
                        "上传图片进行单张或批量推理。\n\n"
                        "**3. 📁 数据集管理**  \n"
                        "预览数据集图片、检查数据集结构是否正确。\n\n"
                        "**4. 🗂️ 模型管理**  \n"
                        "导入、导出、查看与删除本地已下载的模型权重。"
                    )

            # 自动刷新定时器（5秒刷新任务列表 + 训练日志）
            auto_timer = gr.Timer(value=5)

            def _refresh_all_home():
                cards, _ = refresh_home()
                return cards, _list_tasks_html()

            refresh_home_btn.click(_refresh_all_home, outputs=[home_cards, home_tasks_html])
            refresh_tasks_btn.click(_list_tasks_html, outputs=home_tasks_html)
            # 定时自动刷新任务列表
            auto_timer.tick(_list_tasks_html, outputs=home_tasks_html)

            # --- 训练结果查看与下载 ---
            with gr.Accordion("📂 训练结果", open=False):
                _run_choices, _run_rows, _run_summary = list_training_runs()
                gr.Markdown(f"_{_run_summary}_", elem_classes="tab-tip")
                with gr.Row():
                    run_select = gr.Dropdown(_run_choices, label="选择训练结果", scale=3, allow_custom_value=True)
                    run_refresh_btn = gr.Button("🔄 刷新列表", size="sm", variant="secondary", scale=1)
                    run_view_btn = gr.Button("🔍 查看详情", size="sm", variant="primary", scale=1)

                run_info = gr.Textbox(label="结果信息", lines=3, interactive=False)
                run_files_df = gr.Dataframe(
                    headers=["文件路径", "大小"], datatype=["str", "str"],
                    value=_run_rows if _run_rows else [["—", "—"]],
                    label="文件列表", interactive=False, wrap=True
                )
                run_gallery = gr.Gallery(label="图片预览（训练曲线、混淆矩阵等）", columns=3, height=300)

                with gr.Row():
                    run_package_btn = gr.Button("📦 打包下载全部（ZIP）", variant="primary", size="sm")
                    run_download = gr.File(label="下载文件", visible=False)

            # 事件绑定
            def _view_run(label):
                rows, gallery, info = get_run_details(label)
                return info, rows, gallery

            def _refresh_runs():
                choices, rows, summary = list_training_runs()
                return gr.update(choices=choices, value=choices[0] if choices else None), summary, rows, []

            def _package_run(label):
                path, msg = package_training_run(label)
                if path:
                    return msg, gr.update(value=path, visible=True)
                else:
                    return msg, gr.update(visible=False)

            run_view_btn.click(_view_run, inputs=[run_select], outputs=[run_info, run_files_df, run_gallery])
            run_refresh_btn.click(_refresh_runs, outputs=[run_select, run_info, run_files_df, run_gallery])
            run_package_btn.click(_package_run, inputs=[run_select], outputs=[run_info, run_download])

        # ==================== 训练 ====================
        with gr.Tab("🎯 训练"):
            gr.Markdown("配置训练任务与数据集，点击「开始训练」后将实时输出训练日志。", elem_classes="tab-tip")
            with gr.Row():
                with gr.Column(scale=1):
                    train_task_input = gr.Dropdown(["检测", "分割", "姿态", "分类", "OBB"], label="任务类型", value="检测")
                    train_model_input = gr.Dropdown([], label="选择模型", info="切换任务类型后自动刷新可选模型")
                    _init_ds_choices, _init_ds_value = get_dataset_choices()
                    data_input = gr.Dropdown(label="选择数据集", info="数据集位于 my_dataset/ 目录下",
                                            value=_init_ds_value, choices=_init_ds_choices)
                with gr.Column(scale=1):
                    epochs_input = gr.Number(label="训练轮数 (epochs)", value=100)
                    batch_size_input = gr.Number(label="批量大小 (batch)", value=16)
                    imgsz_input = gr.Number(label="图像尺寸 (imgsz)", value=640)

            with gr.Accordion("数据集分割设置", open=False):
                with gr.Row():
                    include_test = gr.Checkbox(label="包含测试集", value=True)
                    test_size = gr.Slider(minimum=0.1, maximum=0.5, value=0.2, step=0.05, label="测试集比例")
                    val_size = gr.Slider(minimum=0.1, maximum=0.5, value=0.2, step=0.05, label="验证集比例")

            with gr.Accordion("优化器与学习率", open=False):
                with gr.Row():
                    patience_input = gr.Number(label="早停耐心值 (patience)", value=50)
                    optimizer_input = gr.Dropdown(["SGD", "Adam", "AdamW", "RMSProp"], label="优化器", value="SGD")
                with gr.Row():
                    lr0_input = gr.Number(label="初始学习率 (lr0)", value=0.01)
                    lrf_input = gr.Number(label="最终学习率因子 (lrf)", value=0.01)
                with gr.Row():
                    momentum_input = gr.Number(label="动量 (momentum)", value=0.937)
                    weight_decay_input = gr.Number(label="权重衰减 (weight_decay)", value=0.0005)
                with gr.Row():
                    warmup_epochs_input = gr.Number(label="预热轮数 (warmup_epochs)", value=3.0)
                    warmup_momentum_input = gr.Number(label="预热动量 (warmup_momentum)", value=0.8)
                    warmup_bias_lr_input = gr.Number(label="预热偏置学习率 (warmup_bias_lr)", value=0.1)

            with gr.Accordion("损失权重", open=False):
                with gr.Row():
                    box_input = gr.Number(label="框损失增益 (box)", value=7.5)
                    cls_input = gr.Number(label="类别损失增益 (cls)", value=0.5)
                    dfl_input = gr.Number(label="DFL损失增益 (dfl)", value=1.5)
                with gr.Row():
                    pose_input = gr.Number(label="姿态损失增益 (pose)", value=12.0)
                    kobj_input = gr.Number(label="关键点目标损失增益 (kobj)", value=2.0)
                    label_smoothing_input = gr.Number(label="标签平滑 (label_smoothing)", value=0.0)

            with gr.Accordion("训练策略与数据增强", open=False):
                with gr.Row():
                    nbs_input = gr.Number(label="标称批量大小 (nbs)", value=64)
                    mask_ratio_input = gr.Number(label="遮罩比例 (mask_ratio)", value=4)
                    dropout_input = gr.Number(label="Dropout率", value=0.0)
                with gr.Row():
                    overlap_mask_input = gr.Checkbox(label="重叠遮罩 (overlap_mask)", value=True)
                    augment_input = gr.Checkbox(label="使用数据增强 (augment)", value=True)
                    amp_input = gr.Checkbox(label="自动混合精度 (AMP)", value=True)
                with gr.Row():
                    mixup_input = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="Mixup 强度")
                    copy_paste_input = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="Copy-Paste 强度")
                    mosaic_input = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.05, label="Mosaic 强度")

            with gr.Accordion("⏰ 训练调度（立即/预约）", open=True):
                train_mode = gr.Radio(["🚀 立即训练", "📅 预约训练"], label="启动方式", value="🚀 立即训练", interactive=True)
                now_default = (datetime.datetime.now() + datetime.timedelta(minutes=5)).replace(second=0, microsecond=0)
                schedule_date = gr.DateTime(
                    label="预约开始时间",
                    value=now_default,
                    type="datetime",
                    interactive=True
                )
                gr.Markdown("预约时间必须晚于当前时间。到点后若有其他训练在运行，会等待其结束后再自动启动。", elem_classes="tab-tip")

            with gr.Accordion("📅 取消预约任务", open=False):
                with gr.Row():
                    cancel_scheduled_dd = gr.Dropdown([], label="选择预约任务", scale=3, allow_custom_value=True)
                    cancel_scheduled_refresh = gr.Button("🔄 刷新列表", size="sm", variant="secondary")
                    cancel_scheduled_btn = gr.Button("❌ 取消预约", variant="stop", size="sm")
                cancel_scheduled_result = gr.Textbox(label="取消结果", lines=2)

            with gr.Row():
                train_button = gr.Button("🚀 开始训练 / 创建预约", variant="primary")
                stop_train_button = gr.Button("⏹️ 停止训练", variant="stop")
                refresh_log_button = gr.Button("🔄 刷新训练日志", variant="secondary")
            train_output = gr.Textbox(label="训练日志 / 结果", lines=12, max_lines=30, autoscroll=True)

            train_task_input.change(update_model_choices, inputs=[train_task_input], outputs=[train_model_input])

            def _start_training(mode, sched_date, *args):
                sched_ts = None
                if "预约" in str(mode):
                    try:
                        if isinstance(sched_date, datetime.datetime):
                            sched_ts = sched_date.timestamp()
                        elif isinstance(sched_date, str):
                            cleaned = sched_date.replace("Z", "+00:00")
                            sched_ts = datetime.datetime.fromisoformat(cleaned).astimezone().timestamp()
                        else:
                            sched_ts = float(sched_date)
                        if sched_ts <= time.time():
                            yield "⚠️ 预约时间必须晚于当前时间。请重新设置预约时间。"
                            return
                    except Exception as e:
                        yield f"⚠️ 预约时间解析失败: {e}"
                        return
                yield from train_model_with_log(*args, _schedule_ts=sched_ts)

            train_click_event = train_button.click(
                _start_training,
                inputs=[train_mode, schedule_date,
                        data_input, epochs_input, train_task_input, train_model_input,
                        device_input, include_test, test_size, val_size, batch_size_input, imgsz_input,
                        patience_input, optimizer_input, lr0_input, lrf_input, momentum_input,
                        weight_decay_input, warmup_epochs_input, warmup_momentum_input,
                        warmup_bias_lr_input, box_input, cls_input, dfl_input, pose_input,
                        kobj_input, label_smoothing_input, nbs_input, overlap_mask_input,
                        mask_ratio_input, dropout_input,
                        augment_input, mixup_input, copy_paste_input, mosaic_input, amp_input],
                outputs=train_output
            )

            def _stop_training():
                if _training_stop_flag.is_set():
                    return ""
                _training_stop_flag.set()
                return "⏹️ 已发送停止信号，训练将在当前 epoch 结束后停止..."

            stop_train_button.click(
                _stop_training,
                outputs=train_output,
                cancels=[train_click_event]
            )

            # 刷新训练日志（页面刷新后恢复训练状态）
            refresh_log_button.click(
                get_training_status,
                outputs=train_output
            )

            # 训练进行中时自动刷新日志（3秒间隔）
            train_log_timer = gr.Timer(value=3, active=False)

            def _auto_refresh_train_log():
                with _training_lock:
                    active = _training_active
                if active:
                    return get_training_status()
                else:
                    return gr.update()

            train_log_timer.tick(_auto_refresh_train_log, outputs=train_output)

            # 训练开始时激活定时器，停止时关闭
            def _on_train_start():
                return gr.Timer(active=True)

            def _on_train_stop():
                return gr.Timer(active=False)

            train_click_event.then(_on_train_start, outputs=train_log_timer)
            stop_train_button.click(_on_train_stop, outputs=train_log_timer)

            # 取消预约
            def _cancel_scheduled_and_refresh(task_id_str):
                msg = cancel_scheduled_task(task_id_str)
                new_choices, new_value = get_scheduled_choices()
                return msg, gr.update(choices=new_choices, value=new_value)

            cancel_scheduled_btn.click(
                _cancel_scheduled_and_refresh,
                inputs=[cancel_scheduled_dd],
                outputs=[cancel_scheduled_result, cancel_scheduled_dd]
            )

            # 刷新预约列表
            def _refresh_scheduled_list():
                new_choices, new_value = get_scheduled_choices()
                return gr.update(choices=new_choices, value=new_value)

            cancel_scheduled_refresh.click(
                _refresh_scheduled_list,
                outputs=cancel_scheduled_dd
            )

        # ==================== 图片预测 ====================
        with gr.Tab("🔮 图片预测"):
            gr.Markdown("支持单张图片预测与批量图片预测，结果自动翻译为中文标签。", elem_classes="tab-tip")
            with gr.Row():
                with gr.Column(scale=1):
                    pred_task_input = gr.Dropdown(["检测", "分割", "姿态", "分类", "OBB"], label="任务类型", value="检测")
                    pred_model_input = gr.Dropdown([], label="选择模型")

            with gr.Accordion("预测参数", open=False):
                with gr.Row():
                    pred_conf = gr.Slider(minimum=0.1, maximum=1.0, value=0.25, step=0.05, label="置信度阈值")
                    pred_iou = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.05, label="IOU阈值")
                    pred_max_det = gr.Number(label="最大检测数", value=300)
                with gr.Row():
                    pred_classes = gr.Textbox(label="类别过滤（逗号分隔的索引）", value="", scale=2)
                    pred_agnostic_nms = gr.Checkbox(label="类别无关NMS", value=False)
                    pred_augment = gr.Checkbox(label="使用TTA", value=False)

            gr.Markdown("#### 单张图片预测")
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(type="numpy", label="输入图像", height=200, sources=["upload"], elem_classes="compact-media")
                    predict_button = gr.Button("🔮 开始预测", variant="primary")
                with gr.Column(scale=1):
                    image_output = gr.Image(type="numpy", label="预测结果", height=300, sources=[], elem_classes="compact-media")
                    text_output = gr.JSON(label="预测详情")
                    error_output = gr.Textbox(label="错误信息", visible=False)

            gr.Markdown("#### 批量图片预测")
            with gr.Row():
                with gr.Column(scale=1):
                    batch_input = gr.File(label="上传多张图片", file_count="multiple", file_types=["image"], elem_classes="compact-media")
                    batch_button = gr.Button("📦 批量预测", variant="primary")
                with gr.Column(scale=1):
                    batch_gallery = gr.Gallery(label="批量预测结果", columns=4, height=300, object_fit="contain", elem_classes="ds-gallery")
                    batch_json = gr.JSON(label="批量预测详情")

            pred_task_input.change(update_model_choices, inputs=[pred_task_input], outputs=[pred_model_input])

            def process_prediction(image, task, model_name, device, conf, iou, max_det, classes, agnostic_nms, augment):
                plot_result, json_result = predict_image(image, task, model_name, device, conf, iou, max_det, classes, agnostic_nms, augment)
                if plot_result is None and isinstance(json_result, dict) and "error" in json_result:
                    return gr.update(visible=True, value=json_result["error"]), gr.update(visible=False), gr.update(visible=False)
                else:
                    return gr.update(visible=False), plot_result, json_result

            predict_button.click(
                process_prediction,
                inputs=[image_input, pred_task_input, pred_model_input, device_input,
                        pred_conf, pred_iou, pred_max_det, pred_classes, pred_agnostic_nms, pred_augment],
                outputs=[error_output, image_output, text_output]
            )

            batch_button.click(
                predict_batch,
                inputs=[batch_input, pred_task_input, pred_model_input, device_input,
                        pred_conf, pred_iou, pred_max_det, pred_classes, pred_agnostic_nms, pred_augment],
                outputs=[batch_gallery, batch_json]
            )

        # ==================== 视频推理 ====================
        with gr.Tab("🎬 视频推理"):
            gr.Markdown("上传视频进行逐帧推理，处理完成后输出带标注的 MP4 视频。", elem_classes="tab-tip")
            with gr.Row():
                with gr.Column(scale=1):
                    video_task_input = gr.Dropdown(["检测", "分割", "姿态", "OBB"], label="任务类型", value="检测")
                    video_model_input = gr.Dropdown([], label="选择模型")
                    video_input = gr.Video(label="输入视频", height=240, sources=["upload"], elem_classes="compact-media")

                    with gr.Accordion("高级选项", open=False):
                        video_conf = gr.Slider(minimum=0.1, maximum=1.0, value=0.25, step=0.05, label="置信度阈值")
                        video_iou = gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.05, label="IOU阈值")
                        video_max_det = gr.Number(label="最大检测数", value=300)
                        video_classes = gr.Textbox(label="类别过滤（逗号分隔的索引）", value="")
                        video_agnostic_nms = gr.Checkbox(label="类别无关NMS", value=False)
                        video_augment = gr.Checkbox(label="使用TTA", value=False)

                    video_button = gr.Button("🎬 开始视频推理", variant="primary")
                with gr.Column(scale=1):
                    video_output = gr.Video(label="推理结果", height=240, sources=[], elem_classes="compact-media")
                    video_error_output = gr.Textbox(label="错误信息", lines=3)

            video_task_input.change(update_model_choices, inputs=[video_task_input], outputs=[video_model_input])
            video_button.click(
                on_video_button_click,
                inputs=[video_input, video_task_input, video_model_input, device_input,
                        video_conf, video_iou, video_max_det, video_classes, video_agnostic_nms, video_augment],
                outputs=[video_output, video_error_output]
            )

        # ==================== 数据集管理 ====================
        with gr.Tab("📁 数据集管理"):
            gr.Markdown("管理、导入、预览和检查数据集。每个数据集位于 `my_dataset/` 下的独立文件夹中。", elem_classes="tab-tip")

            # --- 数据集列表 ---
            with gr.Accordion("📋 已有数据集", open=True):
                ds_refresh_btn = gr.Button("🔄 刷新数据集列表", variant="primary", size="sm")
                _init_ds_rows, _init_ds_summary = list_datasets_info()
                ds_df = gr.Dataframe(
                    headers=["数据集名称", "图片数", "标签数", "类别"],
                    datatype=["str", "number", "number", "str"],
                    value=_init_ds_rows,
                    interactive=False,
                    wrap=True
                )
                ds_summary = gr.Markdown(value=_init_ds_summary)

            # --- 导入数据集 ---
            with gr.Accordion("📦 导入数据集", open=False):
                with gr.Accordion("方式一：上传压缩包（推荐，适合大数据集）", open=True):
                    gr.Markdown(
                        "压缩包结构（顶层文件夹名即数据集名称）：\n"
                        "• **扁平结构**：顶层 → `images/` + `labels/` + `classes.txt`\n"
                        "• **划分结构**：顶层 → `images/train、images/val` + `labels/train、labels/val` + `classes.txt`\n"
                        "支持 `.zip` / `.tar.gz` / `.tgz` / `.tar` 格式，后台自动解压导入。",
                        elem_classes="tab-tip"
                    )
                    ds_zip_upload = gr.File(
                        label="上传压缩包",
                        file_count="single",
                        file_types=[".zip", ".tar.gz", ".tgz", ".tar"],
                        elem_classes="file-upload"
                    )
                    ds_zip_btn = gr.Button("📦 解压并导入", variant="primary")
                    ds_zip_result = gr.Textbox(label="导入结果", lines=4)

                with gr.Accordion("方式二：逐个上传文件", open=False):
                    gr.Markdown("输入数据集名称，上传图片文件、标签文件和 classes.txt，系统将自动创建目录结构。", elem_classes="tab-tip")
                    ds_name_input = gr.Textbox(label="数据集名称", placeholder="例如: car_detection", scale=2)
                    with gr.Row():
                        ds_img_upload = gr.File(label="上传图片（可多选）", file_count="multiple",
                                                file_types=["image"], scale=2, elem_classes="file-upload")
                        ds_lbl_upload = gr.File(label="上传标签 .txt（可多选）", file_count="multiple",
                                                file_types=[".txt"], scale=2, elem_classes="file-upload")
                        ds_cls_upload = gr.File(label="上传 classes.txt（单个）", file_count="single",
                                                file_types=[".txt"], scale=1, elem_classes="file-upload")
                    ds_import_btn = gr.Button("📥 导入数据集", variant="primary")
                    ds_import_result = gr.Textbox(label="导入结果", lines=4)

            # --- 删除数据集 ---
            with gr.Accordion("🗑️ 删除数据集", open=False):
                _init_del_choices, _init_del_value = get_dataset_choices()
                ds_delete_dd = gr.Dropdown(
                    label="选择要删除的数据集",
                    choices=_init_del_choices,
                    value=None, scale=3
                )
                ds_delete_btn = gr.Button("🗑️ 删除数据集", variant="stop", scale=1)
                ds_delete_result = gr.Textbox(label="删除结果", lines=2)

            # --- 预览与检查 ---
            with gr.Accordion("🔍 预览与检查", open=False):
                _init_prev_choices, _init_prev_value = get_dataset_choices()
                dataset_path_input = gr.Dropdown(
                    label="选择数据集",
                    choices=_init_prev_choices,
                    value=_init_prev_value,
                    info="选择 my_dataset 下的数据集进行预览或检查"
                )
                with gr.Row():
                    preview_btn = gr.Button("🖼️ 预览数据集", variant="primary")
                    check_btn = gr.Button("✅ 检查数据集", variant="secondary")
                    prev_page_btn = gr.Button("⬅️ 上一页", size="sm")
                    next_page_btn = gr.Button("➡️ 下一页", size="sm")

                with gr.Row():
                    dataset_page_input = gr.Number(label="页码（从 0 开始，0=第1页）", value=0, precision=0, minimum=0, scale=1)

                dataset_gallery = gr.Gallery(label="数据集预览", columns=4, height=560, object_fit="contain", preview=True, elem_classes="ds-gallery")
                dataset_info = gr.Markdown(label="数据集信息")
                dataset_check_output = gr.Markdown(label="检查结果")

            # --- 事件绑定 ---
            def ds_import_and_refresh(name, imgs, lbls, cls):
                msg, ds_choices, ds_value, yaml_choices, yaml_value, rows, summary = import_dataset_func(name, imgs, lbls, cls)
                return msg, rows, summary, \
                    gr.update(choices=ds_choices, value=ds_value), \
                    gr.update(choices=ds_choices, value=ds_value), \
                    gr.update(choices=ds_choices, value=ds_value)

            ds_import_btn.click(
                ds_import_and_refresh,
                inputs=[ds_name_input, ds_img_upload, ds_lbl_upload, ds_cls_upload],
                outputs=[
                    ds_import_result, ds_df, ds_summary,
                    ds_delete_dd,
                    dataset_path_input,
                    data_input
                ]
            )

            def ds_delete_and_refresh(path):
                msg, ds_choices, ds_value, yaml_choices, yaml_value, rows, summary = delete_dataset_func(path)
                return msg, rows, summary, \
                    gr.update(choices=ds_choices, value=ds_value), \
                    gr.update(choices=ds_choices, value=ds_value), \
                    gr.update(choices=ds_choices, value=ds_value)

            ds_delete_btn.click(
                ds_delete_and_refresh,
                inputs=[ds_delete_dd],
                outputs=[
                    ds_delete_result, ds_df, ds_summary,
                    ds_delete_dd,
                    dataset_path_input,
                    data_input
                ]
            )

            def ds_refresh_all():
                ds_choices, ds_value = get_dataset_choices()
                yaml_choices, yaml_value = get_dataset_yaml_choices()
                rows, summary = list_datasets_info()
                return rows, summary, \
                    gr.update(choices=ds_choices, value=ds_value), \
                    gr.update(choices=ds_choices, value=ds_value), \
                    gr.update(choices=ds_choices, value=ds_value)

            ds_refresh_btn.click(
                ds_refresh_all,
                outputs=[
                    ds_df, ds_summary,
                    ds_delete_dd,
                    dataset_path_input,
                    data_input
                ]
            )

            def ds_zip_and_refresh(zip_file):
                msg, ds_choices, ds_value, yaml_choices, yaml_value, rows, summary = import_dataset_zip(zip_file)
                return msg, rows, summary, \
                    gr.update(choices=ds_choices, value=ds_value), \
                    gr.update(choices=ds_choices, value=ds_value), \
                    gr.update(choices=ds_choices, value=ds_value)

            ds_zip_btn.click(
                ds_zip_and_refresh,
                inputs=[ds_zip_upload],
                outputs=[
                    ds_zip_result, ds_df, ds_summary,
                    ds_delete_dd,
                    dataset_path_input,
                    data_input
                ]
            )

            preview_btn.click(
                preview_dataset_ui,
                inputs=[dataset_path_input, dataset_page_input],
                outputs=[dataset_gallery, dataset_info, dataset_page_input]
            )
            prev_page_btn.click(
                prev_dataset_page,
                inputs=[dataset_path_input, dataset_page_input],
                outputs=[dataset_gallery, dataset_info, dataset_page_input]
            )
            next_page_btn.click(
                next_dataset_page,
                inputs=[dataset_path_input, dataset_page_input],
                outputs=[dataset_gallery, dataset_info, dataset_page_input]
            )
            check_btn.click(check_dataset_ui, inputs=[dataset_path_input], outputs=[dataset_check_output])

        # ==================== 模型管理 ====================
        with gr.Tab("🗂️ 模型管理"):
            gr.Markdown("查看、导入、导出和管理本地已下载的模型权重文件。", elem_classes="tab-tip")
            with gr.Row():
                refresh_models_btn = gr.Button("🔄 刷新全部列表", variant="primary")

            # --- 模型列表 ---
            with gr.Accordion("📋 已下载模型", open=True):
                model_df = gr.Dataframe(
                    headers=["任务类型", "模型名称", "大小 (MB)"],
                    datatype=["str", "str", "number"],
                    value=refresh_model_list()[0],
                    interactive=False,
                    wrap=True
                )
                model_summary = gr.Markdown(value=refresh_model_list()[1])

            # --- 导出模型 ---
            with gr.Accordion("📦 导出模型", open=False):
                gr.Markdown("选择模型并导出为部署格式，导出后自动下载。选择 **pt** 格式可下载原始模型文件。", elem_classes="tab-tip")
                with gr.Row():
                    mgmt_export_task = gr.Dropdown(["检测", "分割", "姿态", "分类", "OBB"], label="任务类型", value="检测", scale=1)
                    mgmt_export_model = gr.Dropdown([], label="选择模型", info="仅显示本地已下载模型", scale=2)
                    mgmt_export_format = gr.Dropdown(
                        ["pt", "onnx", "openvino", "engine", "coreml", "saved_model", "pb", "tflite", "edgetpu", "tfjs"],
                        label="导出格式", value="onnx", scale=1,
                        info="pt = 下载原始模型"
                    )
                with gr.Accordion("高级选项", open=False):
                    with gr.Row():
                        mgmt_export_imgsz = gr.Number(label="图像尺寸 (imgsz)", value=640)
                        mgmt_export_batch = gr.Number(label="批量大小 (batch)", value=1)
                        mgmt_export_opset = gr.Number(label="ONNX操作集 (opset)", value=12)
                    with gr.Row():
                        mgmt_export_half = gr.Checkbox(label="使用半精度 (half)", value=False)
                        mgmt_export_int8 = gr.Checkbox(label="INT8量化", value=False)
                        mgmt_export_dynamic = gr.Checkbox(label="动态轴 (dynamic)", value=False)
                        mgmt_export_simplify = gr.Checkbox(label="简化模型 (simplify)", value=True)
                mgmt_export_btn = gr.Button("📦 导出并下载", variant="primary")
                mgmt_export_result = gr.Textbox(label="导出结果", lines=3)
                mgmt_export_download = gr.File(label="下载导出文件", visible=False, elem_id="mgmt_export_download")

            # --- 导出历史 ---
            with gr.Accordion("📂 导出历史文件", open=False):
                gr.Markdown("所有已导出的文件列表，可随时下载。文件保存在 `models/` 目录下，刷新不会丢失。", elem_classes="tab-tip")
                _init_exp_rows, _init_exp_summary, _init_exp_dict = list_exported_files()
                export_history_df = gr.Dataframe(
                    headers=["任务类型", "文件名", "格式", "大小 (MB)"],
                    datatype=["str", "str", "str", "number"],
                    value=_init_exp_rows,
                    interactive=False,
                    wrap=True
                )
                export_history_summary = gr.Markdown(value=_init_exp_summary)
                with gr.Row():
                    export_history_select = gr.Dropdown(
                        label="选择文件下载",
                        choices=list(_init_exp_dict.keys()),
                        scale=3
                    )
                    export_history_download_btn = gr.Button("⬇️ 下载选中文件", variant="secondary", scale=1)
                    export_history_delete_btn = gr.Button("🗑️ 删除选中文件", variant="stop", scale=1)
                export_history_file = gr.File(label="下载", visible=False)
                export_history_msg = gr.Textbox(label="操作结果", lines=2, visible=False)

            # --- 导入模型 ---
            with gr.Accordion("📥 导入模型", open=False):
                gr.Markdown("上传本地训练的 `.pt` 权重文件，导入后可在训练/图片预测等页面直接使用。支持批量上传。", elem_classes="tab-tip")
                with gr.Row():
                    import_task_input = gr.Dropdown(["检测", "分割", "姿态", "分类", "OBB"], label="任务类型（决定模型存放目录）", value="检测", scale=1)
                    import_name_input = gr.Textbox(label="自定义文件名（仅单文件时生效，可不填）", placeholder="例如 my_best_model，可不加 .pt 后缀", scale=2)
                with gr.Row():
                    import_file_input = gr.File(label="上传 .pt 模型文件", file_count="multiple", file_types=[".pt"], scale=3, elem_classes="file-upload")
                    import_btn = gr.Button("📥 导入模型", variant="primary", size="lg", scale=1)
                import_result = gr.Textbox(label="导入结果", lines=4)

            # --- 删除模型 ---
            with gr.Accordion("🗑️ 删除模型", open=False):
                with gr.Row():
                    mgmt_task_input = gr.Dropdown(["检测", "分割", "姿态", "分类", "OBB"], label="任务类型", value="检测")
                    mgmt_model_input = gr.Dropdown([], label="选择模型", info="仅显示本地已下载模型")
                    delete_model_btn = gr.Button("🗑️ 删除选中模型", variant="stop")
                delete_result = gr.Textbox(label="删除结果", lines=2)

            # --- 事件绑定 ---
            mgmt_task_input.change(get_local_model_choices, inputs=[mgmt_task_input], outputs=[mgmt_model_input])
            mgmt_export_task.change(get_local_model_choices, inputs=[mgmt_export_task], outputs=[mgmt_export_model])

            def refresh_all_lists():
                m_rows, m_summary = refresh_model_list()
                e_rows, e_summary, e_dict = list_exported_files()
                return m_rows, m_summary, e_rows, e_summary, gr.update(choices=list(e_dict.keys()))

            refresh_models_btn.click(
                refresh_all_lists,
                outputs=[model_df, model_summary, export_history_df, export_history_summary, export_history_select]
            )

            def delete_and_refresh(task_type, model_name):
                msg = delete_model_func(task_type, model_name)
                rows, summary = refresh_model_list()
                choices = get_local_model_choices(task_type)
                return msg, rows, summary, choices

            delete_model_btn.click(
                delete_and_refresh,
                inputs=[mgmt_task_input, mgmt_model_input],
                outputs=[delete_result, model_df, model_summary, mgmt_model_input]
            )

            def import_and_refresh(task_type, uploaded_files, custom_name):
                msg = import_model_func(task_type, uploaded_files, custom_name)
                rows, summary = refresh_model_list()
                mgmt_choices = get_local_model_choices(task_type)
                mgmt_export_choices = get_local_model_choices(task_type)
                common_choices = update_model_choices(task_type)
                return msg, rows, summary, mgmt_choices, mgmt_export_choices, common_choices, common_choices, common_choices

            import_btn.click(
                import_and_refresh,
                inputs=[import_task_input, import_file_input, import_name_input],
                outputs=[
                    import_result, model_df, model_summary, mgmt_model_input, mgmt_export_model,
                    train_model_input, pred_model_input, video_model_input
                ]
            )

            def mgmt_export_and_refresh(format, task, model_name, device, imgsz, batch, half, int8, dynamic, simplify, opset):
                try:
                    msg, export_path = export_model(format, task, model_name, device, imgsz, batch, half, int8, dynamic, simplify, opset)
                except Exception as e:
                    msg, export_path = f"❌ 导出失败: {str(e)}", None
                # 刷新导出历史
                e_rows, e_summary, e_dict = list_exported_files()
                if export_path and os.path.exists(export_path):
                    return msg, gr.update(value=export_path, visible=True), e_rows, e_summary, gr.update(choices=list(e_dict.keys()))
                else:
                    return msg, gr.update(value=None, visible=False), e_rows, e_summary, gr.update(choices=list(e_dict.keys()))

            mgmt_export_btn.click(
                mgmt_export_and_refresh,
                inputs=[mgmt_export_format, mgmt_export_task, mgmt_export_model, device_input,
                        mgmt_export_imgsz, mgmt_export_batch, mgmt_export_half, mgmt_export_int8,
                        mgmt_export_dynamic, mgmt_export_simplify, mgmt_export_opset],
                outputs=[mgmt_export_result, mgmt_export_download, export_history_df, export_history_summary, export_history_select]
            ).then(
                fn=None,
                js="""() => {
                    setTimeout(() => {
                        const fileComponent = document.querySelector('#mgmt_export_download');
                        if (!fileComponent) return;
                        const downloadBtn = fileComponent.querySelector('a[download], button[title*="download" i], a[href*="file="]');
                        if (downloadBtn) {
                            downloadBtn.click();
                        } else {
                            const links = fileComponent.querySelectorAll('a');
                            for (const link of links) {
                                if (link.href && (link.href.includes('file=') || link.hasAttribute('download'))) {
                                    link.click();
                                    break;
                                }
                            }
                        }
                    }, 800);
                }"""
            )

            def download_exported_file(selected_name):
                if not selected_name:
                    return gr.update(value=None, visible=False)
                _, _, e_dict = list_exported_files()
                fpath = e_dict.get(selected_name)
                if fpath and os.path.exists(fpath):
                    return gr.update(value=fpath, visible=True)
                return gr.update(value=None, visible=False)

            export_history_download_btn.click(
                download_exported_file,
                inputs=[export_history_select],
                outputs=[export_history_file]
            )

            def delete_exported_file(selected_name):
                if not selected_name:
                    return "请先选择要删除的文件。", gr.update(), gr.update(), gr.update()
                _, _, e_dict = list_exported_files()
                fpath = e_dict.get(selected_name)
                if not fpath or not os.path.exists(fpath):
                    return "文件不存在，可能已被删除。", gr.update(), gr.update(), gr.update()
                try:
                    os.remove(fpath)
                    e_rows, e_summary, new_dict = list_exported_files()
                    return f"✅ 已删除: {selected_name}", e_rows, e_summary, gr.update(choices=list(new_dict.keys()))
                except Exception as e:
                    return f"❌ 删除失败: {str(e)}", gr.update(), gr.update(), gr.update()

            export_history_delete_btn.click(
                delete_exported_file,
                inputs=[export_history_select],
                outputs=[export_history_msg, export_history_df, export_history_summary, export_history_select]
            )

        # ==================== 模型转换 (ONNX → RKNN) ====================
        with gr.Tab("🔄 模型转换"):
            gr.Markdown("上传 **ONNX** 模型，一键转换为 RK3588 等 Rockchip NPU 使用的 **RKNN** 模型（转换在服务端完成，文件保存在 `models/rknn/`）。", elem_classes="tab-tip")
            with gr.Row():
                with gr.Column(scale=1):
                    rknn_onnx_input = gr.File(label="① 上传 ONNX 模型", file_types=[".onnx"], elem_classes="file-upload")
                    rknn_target_input = gr.Dropdown(
                        ["rk3588", "rk3576", "rk3568", "rk3566", "rk3562"],
                        label="② 目标平台（Rockchip 芯片型号）", value="rk3588"
                    )
                    rknn_quantize_input = gr.Checkbox(
                        label="③ 启用量化 INT8（模型更小、NPU 推理更快，需上传校准图片；不勾选则为浮点、精度最高）",
                        value=False
                    )
                    rknn_calib_input = gr.File(
                        label="校准图片（量化时必填，建议 100-500 张与实际场景相近的 JPG）",
                        file_count="multiple", file_types=["image"], visible=False
                    )
                    rknn_convert_btn = gr.Button("🔄 开始转换", variant="primary", size="lg")
                with gr.Column(scale=1):
                    rknn_log_output = gr.Textbox(label="转换日志", lines=14, max_lines=30)
                    rknn_result_file = gr.File(label="下载转换后的 RKNN 模型", visible=False)

            rknn_quantize_input.change(
                fn=lambda q: gr.update(visible=bool(q)),
                inputs=rknn_quantize_input,
                outputs=rknn_calib_input
            )

            with gr.Accordion("📂 已转换的 RKNN 文件", open=False):
                _rknn_rows, _rknn_summary, _rknn_dict = list_rknn_files()
                rknn_df = gr.Dataframe(
                    headers=["文件名", "格式", "大小 (MB)"],
                    datatype=["str", "str", "number"],
                    value=_rknn_rows, interactive=False, wrap=True
                )
                rknn_summary_md = gr.Markdown(value=_rknn_summary)
                with gr.Row():
                    rknn_select = gr.Dropdown(label="选择文件", choices=list(_rknn_dict.keys()), scale=3)
                    rknn_dl_btn = gr.Button("⬇️ 下载", variant="secondary", scale=1)
                    rknn_del_btn = gr.Button("🗑️ 删除", variant="stop", scale=1)
                    rknn_refresh_btn = gr.Button("🔄 刷新", variant="primary", scale=1)
                rknn_dl_file = gr.File(label="下载", visible=False)
                rknn_op_msg = gr.Textbox(label="操作结果", lines=2, visible=False)

            def _rknn_list_updates():
                rows, summary, fdict = list_rknn_files()
                return rows, summary, gr.update(choices=list(fdict.keys()))

            def rknn_convert_and_refresh(onnx_file, target, quantize, calib):
                msg, rknn_path = convert_onnx_to_rknn(onnx_file, target, quantize, calib)
                rows, summary, fdict = list_rknn_files()
                if rknn_path and os.path.exists(rknn_path):
                    file_upd = gr.update(value=rknn_path, visible=True)
                else:
                    file_upd = gr.update(value=None, visible=False)
                return msg, file_upd, rows, summary, gr.update(choices=list(fdict.keys()))

            rknn_convert_btn.click(
                rknn_convert_and_refresh,
                inputs=[rknn_onnx_input, rknn_target_input, rknn_quantize_input, rknn_calib_input],
                outputs=[rknn_log_output, rknn_result_file, rknn_df, rknn_summary_md, rknn_select]
            )

            rknn_refresh_btn.click(
                _rknn_list_updates,
                outputs=[rknn_df, rknn_summary_md, rknn_select]
            )

            def rknn_download_selected(name):
                if not name:
                    return gr.update(value=None, visible=False)
                _, _, fdict = list_rknn_files()
                fpath = fdict.get(name)
                if fpath and os.path.exists(fpath):
                    return gr.update(value=fpath, visible=True)
                return gr.update(value=None, visible=False)

            rknn_dl_btn.click(rknn_download_selected, inputs=rknn_select, outputs=rknn_dl_file)

            def rknn_delete_selected(name):
                if not name:
                    rows, summary, fdict = list_rknn_files()
                    return gr.update(value="请先选择文件。", visible=True), rows, summary, gr.update(choices=list(fdict.keys()))
                _, _, fdict = list_rknn_files()
                fpath = fdict.get(name)
                try:
                    if fpath and os.path.exists(fpath):
                        os.remove(fpath)
                    msg = f"✅ 已删除: {name}"
                except Exception as e:
                    msg = f"❌ 删除失败: {e}"
                rows, summary, new_dict = list_rknn_files()
                return gr.update(value=msg, visible=True), rows, summary, gr.update(choices=list(new_dict.keys()))

            rknn_del_btn.click(
                rknn_delete_selected,
                inputs=rknn_select,
                outputs=[rknn_op_msg, rknn_df, rknn_summary_md, rknn_select]
            )

        # ==================== 帮助文档 ====================
        with gr.Tab("📖 帮助文档"):
            gr.Markdown("## YOLO11 视觉训练平台 - 完整使用手册", elem_classes="main-header")

            with gr.Accordion("🚀 快速入门", open=True):
                gr.Markdown("""
    #### 5 步上手

    1. **导入数据集** → 进入「📁 数据集管理」，上传压缩包（推荐）或手动上传图片/标签/classes.txt
    2. **选择模型** → 进入「🎯 训练」，选择任务类型和数据集，选择预训练模型（如 yolo11n.pt）
    3. **开始训练** → 设置 epochs/batch 等参数，选择「立即训练」或「预约训练」，点击开始
    4. **查看结果** → 训练完成后，在「🏠 首页」的「📂 训练结果」中查看曲线图、混淆矩阵，可打包下载
    5. **预测/导出/转换** → 进入「🔮 图片预测」测试图片，「🗂️ 模型管理」导出 ONNX（opset=12），「🔄 模型转换」生成 RKNN 部署到边缘盒子

    #### 目录结构
    ```
    YOLOV11/
    ├── models/              # 预训练模型 & 训练结果 & 导入的模型
    │   ├── detection/       # 检测任务模型 (.pt)
    │   ├── segmentation/    # 分割任务模型
    │   ├── pose/            # 姿态任务模型
    │   ├── classification/  # 分类任务模型
    │   └── obb/             # 旋转目标检测模型
    ├── my_dataset/          # 数据集根目录
    │   └── your_dataset/    # 每个数据集一个文件夹
    │       ├── images/      # 图片 (支持 train/val/test 子目录)
    │       ├── labels/      # YOLO 格式标签 (.txt)
    │       └── classes.txt  # 类别定义
    ├── runs/                # 训练完整输出目录 (曲线图、权重等)
    ├── assets/              # 网页图标
    ├── config.json          # 配置文件 (账号密码、主题等)
    ├── Dockerfile.cpu       # CPU 版 Docker 构建
    ├── Dockerfile.gpu       # GPU 版 Docker 构建
    └── docker-compose.yml   # Docker Compose 编排
    ```
    """)

            with gr.Accordion("📁 数据集管理", open=False):
                gr.Markdown("""
    #### 方式一：上传压缩包（推荐，适合大数据集）

    支持 `.zip` / `.tar.gz` / `.tgz` / `.tar` 格式。压缩包结构：

    **扁平结构：**
    ```
    dataset.zip
    └── dataset/
        ├── images/          # 图片 (支持 train/val 子目录)
        │   ├── train/
        │   │   ├── 001.jpg
        │   │   └── 002.jpg
        │   └── val/
        │       └── 003.jpg
        ├── labels/          # 标签 (与 images 子目录对应)
        │   ├── train/
        │   │   ├── 001.txt
        │   │   └── 002.txt
        │   └── val/
        │       └── 003.txt
        └── classes.txt      # 类别名称
    ```

    **划分结构（自动识别 train/val/test 子目录）：**
    ```
    dataset.zip
    └── my_data/
        ├── images/
        │   ├── train/       # 训练集图片
        │   ├── val/         # 验证集图片
        │   └── test/        # 测试集图片
        ├── labels/
        │   ├── train/       # 训练集标签
        │   ├── val/
        │   └── test/
        └── classes.txt
    ```

    #### 方式二：手动上传

    1. 输入数据集名称
    2. 上传图片文件到 `images/`
    3. 上传标签文件到 `labels/`
    4. 上传 `classes.txt` 或手动输入类别名

    #### YOLO 标签格式
    每张图片对应一个同名的 `.txt` 文件，每行格式：
    ```
    类别索引 中心x 中心y 宽 高
    ```
    坐标值归一化到 0~1 范围（除以图片宽高）。

    **示例**（图片 640x480，物体在左上角 100x100 区域）：
    ```
    0 0.078 0.104 0.156 0.208
    ```

    #### classes.txt 格式
    每行一个类别名称，行号即为类别索引：
    ```
    car
    person
    bicycle
    ```
    对应：0=car, 1=person, 2=bicycle

    #### 任务类型与标签格式对应
    | 任务 | 标签格式 | 每行内容 |
    |------|----------|----------|
    | 检测 | `cls cx cy w h` | 类别 + 边界框 |
    | 分割 | `cls x1 y1 x2 y2 ...` | 类别 + 多边形点 |
    | 姿态 | `cls cx cy w h px1 py1 v1 ...` | 类别 + 框 + 关键点 |
    | 分类 | 无标签文件 | 仅需 images/ 和 classes.txt |
    | OBB | `cls x1 y1 x2 y2 x3 y3 x4 y4` | 类别 + 旋转框四角 |
    """)

            with gr.Accordion("🎯 训练功能", open=False):
                gr.Markdown("""
    #### 立即训练 vs 预约训练

    | 模式 | 说明 |
    |------|------|
    | 🚀 立即训练 | 点击后立即开始训练，实时输出日志 |
    | 📅 预约训练 | 设置未来某个时间点自动开始训练，到点后自动启动 |

    预约训练适合在空闲时段（如夜间）自动执行。预约后可在「📅 取消预约任务」中取消。

    #### 停止训练
    训练过程中可随时点击「⏹️ 停止训练」，训练会在当前 epoch 完成后停止。已完成的 epoch 权重会保存。

    #### 训练结果
    - **模型自动复制**：训练完成后，`best.pt` 自动复制到 `models/` 目录（如 `yolo11n_trained.pt`），可直接在图片预测中使用
    - **完整结果**：在「🏠 首页」→「📂 训练结果」中查看，包含训练曲线、混淆矩阵、验证批次图等
    - **打包下载**：点击「📦 打包下载全部」可将整个训练结果目录打包为 ZIP 下载

    #### 基本参数
    | 参数 | 默认值 | 说明 |
    |------|--------|------|
    | 任务类型 | 检测 | 选择检测/分割/姿态/分类/OBB |
    | 选择模型 | yolo11n.pt | 预训练模型，n/s/m/l/x 依次增大 |
    | 选择数据集 | — | 从 my_dataset/ 下选择已导入的数据集 |
    | epochs | 100 | 训练总轮数，数据量小时可设 50~200 |
    | batch | 16 | 每次迭代的图片数，显存不足时调小（4/8） |
    | imgsz | 640 | 输入图像尺寸，必须为 32 的倍数 |

    #### 数据集分割
    | 参数 | 默认值 | 说明 |
    |------|--------|------|
    | 包含测试集 | 是 | 是否分割出独立测试集 |
    | 测试集比例 | 0.2 | 测试集占总数据的比例 |
    | 验证集比例 | 0.2 | 验证集占总数据的比例 |

    #### 优化器与学习率
    | 参数 | 默认值 | 说明 |
    |------|--------|------|
    | patience | 50 | 早停耐心值，连续 N 轮无提升则停止 |
    | optimizer | SGD | 优化器：SGD(稳定)/Adam(快速)/AdamW/RMSProp |
    | lr0 | 0.01 | 初始学习率，SGD 推荐 0.01，Adam 推荐 0.001 |
    | lrf | 0.01 | 最终学习率 = lr0 × lrf，即衰减到 1% |
    | momentum | 0.937 | SGD 动量，通常 0.9~0.98 |
    | weight_decay | 0.0005 | L2 正则化系数，防止过拟合 |
    | warmup_epochs | 3.0 | 预热轮数，学习率从 0 线性升到 lr0 |
    | warmup_momentum | 0.8 | 预热阶段动量 |
    | warmup_bias_lr | 0.1 | 预热阶段偏置参数学习率 |

    #### 损失权重
    | 参数 | 默认值 | 说明 |
    |------|--------|------|
    | box | 7.5 | 边界框回归损失权重 |
    | cls | 0.5 | 分类损失权重 |
    | dfl | 1.5 | Distribution Focal Loss 权重 |
    | pose | 12.0 | 姿态关键点损失权重（仅姿态任务） |
    | kobj | 2.0 | 关键点目标损失权重（仅姿态任务） |
    | label_smoothing | 0.0 | 标签平滑，0~0.1，防止过拟合 |

    #### 训练策略与数据增强
    | 参数 | 默认值 | 说明 |
    |------|--------|------|
    | nbs | 64 | 标称批量大小，自动调整学习率 |
    | mask_ratio | 4 | 分割掩码下采样比例 |
    | dropout | 0.0 | Dropout 比率，0~0.5 |
    | overlap_mask | 是 | 分割掩码是否重叠 |
    | augment | 是 | 是否启用马赛克等数据增强 |
    | amp | 是 | 自动混合精度训练，加速+省显存 |
    | mixup | 0.1 | Mixup 增强强度，0=关闭 |
    | copy_paste | 0.1 | Copy-Paste 增强强度，0=关闭 |
    | mosaic | 1.0 | Mosaic 增强强度，0=关闭 |
    """)

            with gr.Accordion("🔄 模型转换（ONNX → RKNN）详解", open=False):
                gr.Markdown("""
    在「🔄 模型转换」页面可将 ONNX 模型转换为 Rockchip NPU 使用的 RKNN 模型，部署到 RK3588 等边缘盒子。

    | 参数 | 默认值 | 说明 |
    |------|--------|------|
    | 上传 ONNX | — | 需先用「🗂️ 模型管理」导出 ONNX（推荐 opset=12） |
    | 目标平台 | rk3588 | Rockchip 芯片型号：rk3588/rk3576/rk3568/rk3566/rk3562 |
    | 量化 INT8 | 否 | 勾选后模型更小、NPU 推理更快，但需上传 100-500 张校准图片 |
    | 校准图片 | — | 仅量化时必填，应与实际使用场景相近的 JPG 图片 |

    **推荐流程**：训练 → 模型管理导出 ONNX（opset=12）→ 模型转换生成 RKNN → 上传到边缘盒子部署。

    > 未量化（浮点）模型体积较大但精度最高，适合先验证链路；量化模型适合正式量产部署。
    > 转换后的文件保存在 `models/rknn/` 目录，可在页面下方列表下载或删除。
    """)

            with gr.Accordion("🔮 预测参数详解", open=False):
                gr.Markdown("""
    #### 图片预测
    支持单张或批量上传图片（jpg/png/bmp），返回标注后的图片和 JSON 格式的检测结果。

    #### 视频推理
    上传视频文件，逐帧进行目标检测，输出标注后的视频。

    | 参数 | 默认值 | 说明 |
    |------|--------|------|
    | conf | 0.25 | 置信度阈值，值越高检测越严格 |
    | iou | 0.7 | IoU 阈值，值越低重叠框过滤越激进 |
    | max_det | 300 | 每张图最大检测数 |
    | classes | 空 | 类别过滤，如 `0,2` 只检测第 0 和第 2 类 |
    | agnostic_nms | 否 | 类别无关 NMS，不同类别间也做重叠抑制 |
    | augment | 否 | 测试时增强 (TTA)，多尺度推理提升精度但变慢 |
    """)

            with gr.Accordion("🗂️ 模型管理", open=False):
                gr.Markdown("""
    #### 功能概览
    | 功能 | 说明 |
    |------|------|
    | 📋 已下载模型 | 查看本地 models/ 目录下的所有模型文件 |
    | 📦 导出模型 | 将 .pt 模型导出为 ONNX 等部署格式，导出后自动下载 |
    | 📂 导出历史 | 查看所有已导出的文件，支持单独下载或删除 |
    | 📥 导入模型 | 上传 .pt 权重文件到 models/ 对应目录 |
    | 🗑️ 删除模型 | 删除本地不需要的模型文件 |

    #### 导出格式说明
    | 格式 | 说明 | 适用场景 |
    |------|------|----------|
    | pt | PyTorch 原始格式 | 直接返回模型文件，不转换 |
    | onnx | ONNX 通用格式 | 跨平台推理（推荐） |
    | openvino | Intel 推理引擎 | Intel CPU/GPU 加速 |
    | engine | TensorRT | NVIDIA GPU 极速推理 |
    | coreml | Apple CoreML | iOS/macOS 部署 |
    | saved_model | TensorFlow SavedModel | TF Serving |
    | tflite | TensorFlow Lite | 移动端部署 |
    | tfjs | TensorFlow.js | 浏览器端推理 |

    #### 导出参数
    | 参数 | 默认值 | 说明 |
    |------|--------|------|
    | imgsz | 640 | 导出模型的输入尺寸 |
    | batch | 1 | 导出模型的批量大小 |
    | opset | 12 | ONNX 操作集版本（仅 ONNX 格式，推荐 12，兼容 RKNN 转换） |
    | half | 否 | FP16 半精度，减小模型体积 |
    | int8 | 否 | INT8 量化，进一步压缩 |
    | dynamic | 否 | 动态输入尺寸，支持可变 batch/imgsz |
    | simplify | 是 | 简化 ONNX 模型图 |
    """)

            with gr.Accordion("🏠 首页功能", open=False):
                gr.Markdown("""
    #### 系统状态卡片
    显示当前系统环境信息：设备类型、CPU/GPU、PyTorch 版本、CUDA 版本等。

    #### 训练任务状态
    实时展示所有训练任务（每 5 秒自动刷新）：
    | 状态 | 说明 |
    |------|------|
    | 📅 已预约 | 预约任务等待到点启动 |
    | 🔄 运行中 | 当前正在训练，显示 Epoch 进度 |
    | ✅ 已完成 | 正常完成 |
    | ⏹️ 已停止 | 用户取消预约或中途停止 |
    | ❌ 失败 | 训练异常或预约冲突 |

    #### 训练结果
    - 选择某次训练运行 → 查看文件列表和图片预览（训练曲线、混淆矩阵等）
    - 点击「📦 打包下载全部」可将结果打包为 ZIP 下载
    """)

            with gr.Accordion("📏 模型命名规则", open=False):
                gr.Markdown("""
    #### YOLO11 模型规格
    | 模型 | 参数量 | mAP | 速度 | 适用场景 |
    |------|--------|-----|------|----------|
    | yolo11n.pt | 2.6M | 39.5 | 最快 | 边缘设备/实时 |
    | yolo11s.pt | 9.4M | 47.0 | 快 | 通用场景 |
    | yolo11m.pt | 20.1M | 51.5 | 中等 | 精度优先 |
    | yolo11l.pt | 25.3M | 53.4 | 较慢 | 高精度需求 |
    | yolo11x.pt | 56.9M | 54.7 | 最慢 | 最高精度 |

    > n=nano, s=small, m=medium, l=large, x=extra large
    >
    > 训练后的模型自动命名为 `{原模型名}_trained.pt`（如 `yolo11n_trained.pt`），重复训练自动追加序号 `_1`、`_2`
    """)

            with gr.Accordion("❓ 常见问题", open=False):
                gr.Markdown("""
    #### Q: CUDA 不可用怎么办？
    A: 检查 NVIDIA 驱动是否安装，PyTorch 版本是否匹配 CUDA 版本。CPU 环境下训练会较慢，建议减少 epochs 或使用 yolo11n 小模型。

    #### Q: 训练时显存不足 (OOM)？
    A: 调小 `batch`（如 4 或 2），或减小 `imgsz`（如 320），或关闭 `amp`。

    #### Q: 数据集图片不显示？
    A: 确认图片在 `images/` 目录下（支持 `images/train/` 子目录），且格式为 jpg/png/bmp。

    #### Q: 压缩包导入失败？
    A: 确保压缩包顶层有文件夹（如 `dataset/`），里面包含 `images/`、`labels/` 和 `classes.txt`。支持 `.zip` / `.tar.gz` / `.tgz` / `.tar` 格式。

    #### Q: ONNX 转 RKNN 失败？
    A: ① 确认镜像已安装 `rknn-toolkit2`；② ONNX 请用 opset=12 导出（模型管理 → 高级选项）；③ 量化模式必须上传校准图片；④ 目标平台需与实际芯片一致。

    #### Q: 导出 ONNX 失败？
    A: 确保已安装 `onnx`、`onnxruntime`、`onnxslim` 包：`pip install onnx onnxruntime onnxslim`

    #### Q: 如何使用自己训练的模型进行预测？
    A: 训练完成后，最佳模型 `best.pt` 会自动复制到 `models/` 目录（文件名如 `yolo11n_trained.pt`），可直接在图片预测/模型管理中选择。完整训练结果（曲线图、混淆矩阵等）可在「🏠 首页」→「📂 训练结果」中查看和打包下载。

    #### Q: 训练中断了怎么办？
    A: 点击「⏹️ 停止训练」后，已完成的 epoch 结果会保存。`best.pt` 会自动复制到 `models/` 目录。完整训练结果在 `runs/` 目录中。

    #### Q: 预约训练到了时间没启动？
    A: 检查是否有其他训练任务正在运行。预约启动时会检测，若已有训练在运行则自动跳过并标记为失败。

    #### Q: 页面刷新后训练日志丢失？
    A: 不会丢失。点击「🔄 刷新训练日志」按钮可恢复当前训练日志。训练进行中时日志也会每 3 秒自动刷新。

    #### Q: 忘记登录密码？
    A: 查看 `config.json` 文件中的 `password` 字段。
    """)

        # 初始化各任务下拉框对应的模型选项 + 导出历史列表 + 训练状态恢复 + 取消预约列表
        def init_page_lists():
            choices = init_all_model_choices()
            m_rows, m_summary = refresh_model_list()
            e_rows, e_summary, e_dict = list_exported_files()
            train_status = get_training_status()
            sch_choices, sch_value = get_scheduled_choices()
            run_choices, run_rows, run_summary = list_training_runs()
            return (*choices, m_rows, m_summary, e_rows, e_summary,
                    gr.update(choices=list(e_dict.keys())), train_status,
                    gr.update(choices=sch_choices, value=sch_value),
                    gr.update(choices=run_choices, value=run_choices[0] if run_choices else None),
                    run_summary, run_rows)

        demo.load(
            init_page_lists,
            outputs=[train_model_input, pred_model_input, video_model_input,
                     mgmt_model_input, mgmt_export_model,
                     model_df, model_summary, export_history_df, export_history_summary, export_history_select,
                     train_output,
                     cancel_scheduled_dd,
                     run_select, run_info, run_files_df]
        )

    # ---------- 登录/退出事件 ----------
    login_btn.click(
        login_success,
        inputs=[login_user, login_pass],
        outputs=[login_group, main_group, login_error, login_state]
    )
    login_pass.submit(
        login_success,
        inputs=[login_user, login_pass],
        outputs=[login_group, main_group, login_error, login_state]
    )
    logout_btn.click(
        logout,
        outputs=[login_group, main_group, login_user, login_pass, login_state]
    )

    # 登录状态变化时：Python 端切页面
    login_state.change(
        restore_login_state,
        inputs=[login_state],
        outputs=[login_group, main_group]
    )
    # 登录状态变化时：前端写/清 localStorage（两种情况一起）
    login_state.change(
        fn=None,
        inputs=[login_state],
        outputs=None,
        js="""(state) => {
            if (state === '1') {
                localStorage.setItem('yolo_logged_in', '1');
                localStorage.setItem('yolo_last_activity', String(Date.now()));
            } else {
                localStorage.removeItem('yolo_logged_in');
                localStorage.removeItem('yolo_last_activity');
            }
        }"""
    )

    # 页面加载：从 localStorage 读取，写入 login_state 触发上面的 change 恢复登录
    demo.load(
        fn=None,
        inputs=None,
        outputs=[login_state],
        js="""() => { return [localStorage.getItem('yolo_logged_in') || '']; }"""
    )

if __name__ == "__main__":
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _favicon_rel = APP_CONFIG.get("favicon_path", "")
    _favicon_abs = os.path.join(_base_dir, _favicon_rel) if _favicon_rel else ""
    _favicon_kwargs = {"favicon_path": _favicon_abs} if _favicon_abs and os.path.exists(_favicon_abs) else {}
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=False,
        share=False,
        theme=gr.themes.Soft(),
        css=custom_css,
        head=f"<script>({THEME_JS_INIT})()</script><script>({SESSION_TIMEOUT_JS})()</script>",
        **_favicon_kwargs
    )
