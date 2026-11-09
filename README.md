# YOLO 视觉训练平台

基于 **Gradio + Ultralytics YOLO11** 的一站式计算机视觉 WebUI：数据集管理、模型训练、图片/视频推理、模型导出与 RKNN 转换，全部在浏览器中完成，并提供 CPU / GPU 双版本 Docker 部署方案。

## 功能模块

| 模块 | 说明 |
| --- | --- |
| 🏠 首页 | 查看训练任务状态，快速进入各功能模块 |
| 🎯 训练 | 选择数据集与预训练模型，配置轮次、图像尺寸等超参数，实时查看训练进度与结果曲线 |
| 🔮 图片预测 | 支持单张图片预测与批量图片预测，可调置信度、IoU、最大检测数等参数 |
| 🎬 视频推理 | 对视频逐帧推理，输出带标注的结果视频 |
| 📁 数据集管理 | 上传压缩包（`.zip` / `.tar.gz` / `.tgz` / `.tar`）自动解压导入，查看图片数、标签数、类别统计，支持预览检查 |
| 🗂️ 模型管理 | 已下载模型列表、导入 `.pt` 模型、导出模型、删除模型、查看导出历史 |
| 🔄 模型转换 | ONNX → RKNN（瑞芯微 RK3588 等 NPU），可选 INT8 量化 |
| 📖 帮助文档 | 内置操作说明、数据集格式说明与常见问题 |

任务类型覆盖 YOLO11 的检测（detect）、分割（segment）、姿态（pose）、旋转框（OBB）与分类（classify）。

## 技术栈

- Python 3.10、Gradio 6、Ultralytics 8.3.x（YOLO11）
- PyTorch 2.0.1 + CUDA 11.7（GPU 版）
- ONNX 1.15 / onnxruntime / onnxslim / rknn-toolkit2 2.3.2
- Docker + Docker Compose（`cpu` / `gpu` 双 profile）

## 快速开始

### 方式一：Docker 部署（推荐）

```bash
# 1. 准备配置文件（首次部署必须执行）
cp config.example.json config.json

# 2. 启动 CPU 版本
docker compose --profile cpu up -d --build

# 或启动 GPU 版本（宿主机需安装 NVIDIA 驱动与 NVIDIA Container Toolkit）
docker compose --profile gpu up -d --build
```

浏览器访问 `http://<服务器IP>:7860`，默认账号 `admin` / `admin123`，**部署后请立刻在 `config.json` 中修改密码**。

完整的构建说明、GPU 环境准备、数据持久化与故障排查见 [DEPLOYMENT.md](DEPLOYMENT.md)。

### 方式二：本地直接运行

```bash
pip install -r requirements.txt
cp config.example.json config.json
python app.py
```

## 目录结构

```
yolo-train/
├── app.py                  # 主程序：Gradio 界面 + 训练/推理/转换逻辑
├── config.example.json     # 配置模板（复制为 config.json 后修改）
├── convert.py              # ONNX → RKNN 转换脚本
├── pt_to_onnx.py           # .pt → ONNX 导出脚本
├── requirements.txt        # Python 依赖
├── Dockerfile.cpu          # CPU 版镜像
├── Dockerfile.gpu          # GPU 版镜像
├── docker-compose.yml      # 容器编排（cpu / gpu 两个 profile）
├── DEPLOYMENT.md           # 详细部署文档
├── assets/                 # 网页图标等静态资源
├── models/                 # 模型权重目录（不入库，启动时自动创建）
├── my_dataset/             # 数据集目录（不入库）
└── runs/                   # 训练输出目录（不入库）
```

> `models/`、`my_dataset/`、`runs/` 体积大且随训练变化，已加入 `.gitignore`；克隆仓库后目录会由程序启动时自动创建。预训练权重首次使用时由 Ultralytics 自动下载，或手动放入 `models/` 对应任务子目录。

## 数据集格式

每个数据集在 `my_dataset/` 下占一个文件夹（文件夹名即数据集名称）：

```
my_dataset/
└── dataset_name/
    ├── images/
    │   ├── train/          # 训练图片（也可直接平铺在 images/ 下）
    │   └── val/            # 验证图片
    ├── labels/
    │   ├── train/          # YOLO 格式 .txt 标签
    │   └── val/
    └── classes.txt         # 类别名称，每行一个
```

也支持在「数据集管理」页面上传压缩包，后台自动解压并按上述结构识别。

## 配置说明

`config.json`（从 `config.example.json` 复制）字段：

| 字段 | 说明 | 默认值 |
| --- | --- | --- |
| `username` | 登录用户名 | `admin` |
| `password` | 登录密码（明文存储，便于找回） | `admin123` |
| `app_title` | 浏览器标签页标题 | `YOLO 视觉训练平台` |
| `favicon_path` | 网页图标路径 | `assets/fangte_dark.ico` |
| `default_theme` | 默认主题，`light` / `dark` | `light` |
| `session_timeout_minutes` | 无操作自动退出时间（分钟） | `720` |

`config.json` 含明文密码，已加入 `.gitignore`，请勿提交到仓库。

## 许可证

本项目基于 [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)，遵循 **AGPL-3.0** 许可证，详见 [LICENSE](LICENSE)。如需将本平台用于商业闭源产品，需向 Ultralytics 申请企业许可证。
