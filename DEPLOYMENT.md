# YOLO + RF-DETR 视觉训练平台 - 部署指南

本文档详细介绍如何在 CPU / GPU 服务器上通过 Docker 部署视觉训练平台（支持 YOLO11 与 RF-DETR 双框架）。

---

## 一、环境要求

### 1.1 通用要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（推荐 Ubuntu 20.04 / 22.04） |
| Docker | 20.10+ |
| Docker Compose | v2+（`docker compose` 命令，非旧版 `docker-compose`） |
| 内存 | ≥ 4GB（训练建议 ≥ 8GB） |
| 磁盘 | ≥ 25GB 可用空间（镜像约 6GB + 训练数据 + 训练产物） |
| 端口 | 7860（WebUI 默认端口） |

### 1.2 GPU 版本额外要求

| 项目 | 要求 |
|------|------|
| NVIDIA GPU | 支持 CUDA 11.8 的 NVIDIA 显卡（RTX 3060 12GB 可跑 RF-DETR Nano/Small/Medium） |
| NVIDIA 驱动 | ≥ 525.60（宿主机安装） |
| NVIDIA Container Toolkit | 最新版（宿主机安装，用于容器内调用 GPU） |

**安装 NVIDIA Container Toolkit**（Ubuntu 示例）：
```bash
# 添加 NVIDIA 仓库
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 安装
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 配置 Docker 运行时
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证
sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

---

## 二、项目文件结构

```
yolo-train/
├── app.py                    # 主程序（Gradio WebUI，YOLO + RF-DETR 双框架）
├── rfdetr_support.py         # RF-DETR 训练/推理适配层
├── config.example.json       # 配置模板（首次部署复制为 config.json）
├── config.json               # 配置文件（账号、主题、超时等，不入库）
├── docker-compose.yml        # Docker Compose 编排文件
├── Dockerfile.cpu            # CPU 版构建文件
├── Dockerfile.gpu            # GPU 版构建文件
├── .dockerignore             # Docker 构建忽略清单
├── requirements.txt          # Python 依赖清单（YOLO 等基础依赖）
├── requirements-rfdetr.txt   # RF-DETR 依赖清单（可选）
├── convert.py                # ONNX → RKNN 转换脚本
├── pt_to_onnx.py             # .pt → ONNX 导出脚本
├── DEPLOYMENT.md             # 部署文档
├── assets/                   # 网页图标等静态资源
├── ultralytics/              # Ultralytics 源码（含本地修改，不入库）
├── models/                   # 模型文件目录（.pt / .pth，不入库）
│   └── rfdetr/               # RF-DETR 权重（.pth + .json 侧车元数据）
├── my_dataset/               # 数据集目录（不入库）
├── runs/                     # 训练结果输出目录（不入库）
└── wheels_gpu/               # GPU 版 PyTorch 本地 wheel（可选，加速构建）
    ├── torch-2.6.0+cu118-cp310-cp310-linux_x86_64.whl
    └── torchvision-0.21.0+cu118-cp310-cp310-linux_x86_64.whl
```

---

## 三、配置文件说明

`config.json` 用于配置登录账号、界面主题等：

```json
{
    "username": "admin",
    "password": "admin123",
    "app_title": "YOLO 视觉训练平台",
    "favicon_path": "assets/fangte_dark.ico",
    "default_theme": "light",
    "session_timeout_minutes": 720
}
```

| 字段 | 说明 | 默认值 |
|------|------|--------|
| `username` | 登录用户名 | `admin` |
| `password` | 登录密码（明文，方便忘记时查看） | `admin123` |
| `app_title` | 浏览器标签页标题 | `YOLO 视觉训练平台` |
| `favicon_path` | 网页图标路径 | `assets/fangte_dark.ico` |
| `default_theme` | 默认主题（`light` / `dark`） | `light` |
| `session_timeout_minutes` | 会话超时时间（分钟），无活动后自动退出 | `720` |

> **安全提示**：生产环境部署后请立即修改默认密码。

---

## 四、CPU 版本部署

### 4.1 快速部署

```bash
# 1. 进入项目目录
cd /path/to/yolo-train

# 2. 首次部署：复制配置模板并修改账号密码
cp config.example.json config.json

# 3. 构建并启动（首次会下载基础镜像，约 5-10 分钟）
docker compose --profile cpu up -d --build

# 4. 查看启动日志
docker compose --profile cpu logs -f
```

### 4.2 启用 RF-DETR（可选）

CPU 版默认不装 RF-DETR。如需启用（仅建议做推理/链路验证，CPU 训练极慢）：

```bash
docker compose --profile cpu build --build-arg INSTALL_RFDETR=true
docker compose --profile cpu up -d
```

### 4.3 验证部署

浏览器访问 `http://<服务器IP>:7860`，使用 `config.json` 中的账号密码登录。

---

## 五、GPU 版本部署

### 5.1 准备 PyTorch Wheel 文件（推荐，避免构建超时）

GPU 版构建默认会从 PyTorch 官方源下载约 2GB 的 wheel，国内服务器可能超时。建议预先下载到 `wheels_gpu/` 目录：

```bash
# 在项目根目录创建 wheels_gpu 文件夹
mkdir -p wheels_gpu
cd wheels_gpu

# 下载 PyTorch 2.6.0+cu118（RF-DETR 依赖 transformers>=5.1，要求 torch>=2.6）
# 方式一：PyTorch 官方源
wget https://download.pytorch.org/whl/cu118/torch-2.6.0%2Bcu118-cp310-cp310-linux_x86_64.whl
wget https://download.pytorch.org/whl/cu118/torchvision-0.21.0%2Bcu118-cp310-cp310-linux_x86_64.whl

# 方式二：阿里云镜像（国内推荐）
wget https://mirrors.aliyun.com/pytorch-wheels/cu118/torch-2.6.0%2Bcu118-cp310-cp310-linux_x86_64.whl
wget https://mirrors.aliyun.com/pytorch-wheels/cu118/torchvision-0.21.0%2Bcu118-cp310-cp310-linux_x86_64.whl

cd ..

# 验证文件存在
ls -lh wheels_gpu/
# 应显示两个 .whl 文件，文件名包含 cp310 和 cu118
```

> **重要**：wheel 文件名必须包含 `cp310-cp310-linux_x86_64` 和 `cu118`。Dockerfile 会按 `TORCH_VERSION` / `TORCH_CUDA` 构建参数 glob 匹配文件名，不匹配则回退在线安装。
>
> **为什么是 2.6.0**：RF-DETR 1.12 依赖 transformers>=5.1，而 transformers 5.x 要求 torch>=2.6。cu118 最高只提供 torch 2.6.0，2.7 起官方不再提供 cu118 wheel。

### 5.2 构建并启动

```bash
# 构建并启动（使用本地 wheel，构建约 5-8 分钟）
docker compose --profile gpu up -d --build

# 如需启用 RF-DETR 训练/推理：
docker compose --profile gpu build --build-arg INSTALL_RFDETR=true
docker compose --profile gpu up -d

# 查看启动日志
docker compose --profile gpu logs -f
```

### 5.3 验证 GPU 可用性

```bash
# 验证 CUDA 是否可用
docker exec yolov11-gpu python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

预期输出：
```
PyTorch: 2.6.0+cu118
CUDA available: True
GPU: NVIDIA GeForce RTX xxxx
```

### 5.4 验证 RF-DETR 可用性（如已启用）

```bash
docker exec yolov11-gpu python -c "import rfdetr; print('RF-DETR:', rfdetr.__version__)"
```

---

## 六、RF-DETR 使用说明

### 6.1 支持的变体

| 变体 | 默认分辨率 | 参数量 | 适用场景 |
|------|-----------|--------|---------|
| RF-DETR Nano | 384 | ~30M | 边缘设备、快速验证 |
| RF-DETR Small | 512 | ~50M | 平衡精度与速度 |
| RF-DETR Medium | 576 | ~80M | 高精度需求 |
| RF-DETR Large | 704 | ~130M | 服务器级 GPU（12GB 显存不够） |

### 6.2 RTX 3060 12GB 推荐配置

| 变体 | 分辨率 | batch_size | 梯度累积 | 有效 batch | 显存占用 |
|------|--------|-----------|---------|-----------|---------|
| Nano | 384 | 8 | 2 | 16 | ~5-6 GB |
| Small | 512 | 4 | 4 | 16 | ~8-10 GB |
| Medium | 576 | 2 | 8 | 16 | ~11-12 GB |

> - 分辨率必须是 32 的倍数
> - 有效 batch = batch_size × grad_accum_steps，建议保持约 16
> - 首次训练会自动下载对应变体的 COCO 预训练权重

### 6.3 数据集格式

RF-DETR 直接复用 YOLO 数据集目录结构，训练时自动补齐 `data.yaml`：

```
my_dataset/
└── my_data/
    ├── images/
    │   ├── train/    # 训练集图片
    │   └── val/      # 验证集图片
    ├── labels/
    │   ├── train/    # YOLO 格式 .txt 标签
    │   └── val/
    ├── classes.txt   # 类别定义（每行一个类别名）
    └── data.yaml     # 训练时自动生成（无需手动创建）
```

### 6.4 训练产物

训练完成后，最佳权重自动拷贝到 `models/rfdetr/`：

```
models/rfdetr/
├── rfdetr_nano_20261112_022423.pth       # 最佳权重（EMA）
└── rfdetr_nano_20261112_022423.json      # 侧车元数据（类别名、变体、分辨率等）
```

`.pth` 文件会出现在预测页面的模型下拉框中，选中即可推理。原始训练输出（含日志、中间 checkpoint）保存在 `runs/rfdetr_<时间戳>/`。

### 6.5 RF-DETR 与 YOLO 的区别

| 特性 | YOLO | RF-DETR |
|------|------|---------|
| 权重后缀 | `.pt` | `.pth` |
| 预约训练 | 支持 | 不支持 |
| 训练框架 | Ultralytics | PyTorch Lightning |
| 数据格式 | YOLO 格式 | YOLO 格式（自动转 COCO） |
| 混合精度 | AMP | bf16-mixed |

---

## 七、常用操作

### 7.1 启停服务

```bash
# 启动（CPU / GPU 二选一）
docker compose --profile cpu up -d
docker compose --profile gpu up -d

# 停止
docker compose --profile cpu down
docker compose --profile gpu down

# 重启（修改 app.py / config.json / rfdetr_support.py 后生效）
docker compose --profile gpu restart
```

> **注意**：CPU 和 GPU 容器共享端口 7860，**不能同时运行**。

### 7.2 查看日志

```bash
# 实时日志
docker compose --profile gpu logs -f

# 最近 100 行
docker compose --profile gpu logs --tail 100

# 查看训练进度（训练日志会输出到容器日志）
docker compose --profile gpu logs -f | grep -E "epoch|train|val|mAP"
```

### 7.3 更新代码

项目通过 volume 挂载了 `app.py`、`rfdetr_support.py`、`config.json`、`assets/`、`ultralytics/`，修改这些文件**无需重新构建镜像**：

```bash
# 1. 上传新代码到服务器（覆盖原文件）
scp app.py rfdetr_support.py ubuntu@<服务器IP>:/path/to/yolo-train/

# 2. 重启容器即可生效
docker compose --profile gpu restart
```

**仅当修改 Dockerfile 或依赖列表时**才需要重新构建：

```bash
docker compose --profile gpu up -d --build
```

### 7.4 容器内执行命令

```bash
# 进入容器交互式终端
docker exec -it yolov11-gpu bash

# 检查 GPU 状态
docker exec yolov11-gpu nvidia-smi

# 检查 Python 包版本
docker exec yolov11-gpu python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"

# 安装临时依赖（容器重启后失效，永久需写进 Dockerfile）
docker exec yolov11-gpu pip install <包名> -i https://mirrors.aliyun.com/pypi/simple/

# 临时安装 gcc（RF-DETR 训练报 "Failed to find C compiler" 时）
docker exec yolov11-gpu apt-get update && apt-get install -y gcc g++
```

---

## 八、数据持久化

以下目录通过 volume 挂载到宿主机，容器重建后数据不丢失：

| 容器路径 | 宿主机路径 | 用途 |
|----------|-----------|------|
| `/app/app.py` | `./app.py` | 主程序（热更新） |
| `/app/rfdetr_support.py` | `./rfdetr_support.py` | RF-DETR 适配层（热更新） |
| `/app/config.json` | `./config.json` | 配置文件（热更新） |
| `/app/assets` | `./assets` | 图标等资源（热更新） |
| `/app/ultralytics` | `./ultralytics` | Ultralytics 源码（热更新） |
| `/app/models` | `./models` | 模型文件（.pt / .pth） |
| `/app/my_dataset` | `./my_dataset` | 数据集 |
| `/app/runs` | `./runs` | 训练结果（曲线图、权重等） |

> **备份**：定期备份 `models/`、`my_dataset/`、`runs/` 三个目录即可保留所有训练成果。

> **注意（克隆仓库后首次部署）**：仓库不包含 `config.json` 与 `ultralytics/` 目录。
> 1. `config.json`：先执行 `cp config.example.json config.json`，否则 Docker 会把该挂载点创建成目录。
> 2. `ultralytics/`：容器内已通过 `requirements.txt` 安装该包。如果不打算修改其源码，建议从 `docker-compose.yml` 中删除 `./ultralytics:/app/ultralytics` 这一行；若保留且本地目录为空，空目录会被挂载进容器并遮蔽已安装的包，导致 `import ultralytics` 失败。

---

## 九、数据集目录结构

数据集放在 `my_dataset/` 下，每个数据集一个子文件夹：

```
my_dataset/
└── dataset/                    # 数据集名（与文件夹名一致）
    ├── images/                 # 图片目录
    │   ├── train/              # 训练集图片
    │   │   ├── 001.jpg
    │   │   └── ...
    │   └── val/                # 验证集图片
    │       ├── 001.jpg
    │       └── ...
    ├── labels/                 # 标签目录（YOLO 格式 .txt）
    │   ├── train/
    │   │   ├── 001.txt
    │   │   └── ...
    │   └── val/
    │       ├── 001.txt
    │       └── ...
    └── classes.txt             # 类别定义（每行一个类别名）
```

YOLO 与 RF-DETR 共用同一套数据集，无需额外转换。RF-DETR 训练时会自动生成 `data.yaml`。

也支持上传压缩包自动解压（`.zip` / `.tar.gz` / `.tgz` / `.tar`），压缩包内需包含上述结构。

---

## 十、共享内存配置

PyTorch DataLoader 多进程加载需要共享内存。Docker 默认 `/dev/shm` 仅 64MB，会导致训练时报错：

```
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
```

`docker-compose.yml` 已配置 `shm_size: '8g'`，无需额外处理。如服务器内存不足，可调整为 `'4g'`。

---

## 十一、常见问题排查

### 11.1 构建失败：pip 下载 403 错误

**现象**：`HTTP error 403 while getting https://pypi.tuna.tsinghua.edu.cn/...`

**原因**：清华 PyPI 镜像对部分包限流。

**解决**：Dockerfile 已改用阿里云源（`mirrors.aliyun.com/pypi/simple/`），如仍失败可尝试更换为其他镜像：
```dockerfile
# 中科大源
-i https://pypi.mirrors.ustc.edu.cn/simple/
# 腾讯云源
-i https://mirrors.cloud.tencent.com/pypi/simple/
```

### 11.2 构建失败：wheel not supported on this platform

**现象**：`torch-xxx.whl is not a supported wheel on this platform`

**原因**：wheel 文件的 Python 版本（如 cp311）与容器内 Python 版本（如 cp310）不匹配。

**解决**：确认 `wheels_gpu/` 中的 wheel 文件名包含 `cp310-cp310-linux_x86_64`（容器使用 Python 3.10）。

### 11.3 构建超时：PyTorch 下载卡住

**现象**：构建过程中 `Downloading torch-xxx.whl` 长时间不动。

**原因**：国内访问 PyTorch 官方源慢。

**解决**：使用 `wheels_gpu/` 目录预下载 wheel 文件（见 [第五章](#五gpu-版本部署)）。

### 11.4 RF-DETR 训练报错：Failed to find C compiler

**现象**：`Failed to find C compiler. Please specify via CC environment variable.`

**原因**：RF-DETR 训练时需实时编译 C 扩展，运行容器里缺少 gcc/g++。

**解决**：当前 Dockerfile 已在运行阶段安装 gcc/g++。如果是旧镜像，临时补装：
```bash
docker exec yolov11-gpu apt-get update && apt-get install -y gcc g++
```
永久修复需用更新后的 Dockerfile 重建镜像。

### 11.5 RF-DETR 导入失败：BackboneConfigMixin

**现象**：`ImportError: cannot import name 'BackboneConfigMixin'`

**原因**：torch 版本低于 2.6 时，transformers 5.x 会禁用 PyTorch 后端。

**解决**：确保 torch>=2.6。Dockerfile.gpu 默认已设为 2.6.0+cu118，并用 constraints 文件防止 pip 在安装 rfdetr 时自动升级 torch。

### 11.6 训练报错：No space left on device

**现象**：训练过程中 `[Errno 28] No space left on device`

**原因**：服务器磁盘空间不足。

**解决**：
```bash
# 清理 Docker 构建缓存
docker builder prune -a -f

# 清理无用镜像
docker image prune -a -f

# 查看磁盘占用
df -h
docker system df
```

### 11.7 训练报错：insufficient shared memory

**现象**：`Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).`

**原因**：Docker 默认 `/dev/shm` 太小（64MB）。

**解决**：确认 `docker-compose.yml` 中已配置 `shm_size: '8g'`，然后重启容器：
```bash
docker compose --profile gpu down
docker compose --profile gpu up -d
```

### 11.8 训练报错：NumPy 2.x 不兼容

**现象**：`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

**原因**：torch 用 NumPy 1.x 编译，但环境装了 NumPy 2.x。

**解决**：Dockerfile 已固定 `numpy<2`，如仍出现，在容器内手动降级：
```bash
docker exec yolov11-gpu pip install "numpy<2" -i https://mirrors.aliyun.com/pypi/simple/
```

### 11.9 wandb 登录报错

**现象**：训练启动时报 `No API key configured (wandb login)`

**原因**：rfdetr[loggers] 带入了 wandb，ultralytics 默认自动启用。

**解决**：`app.py` 启动时已通过 `ultralytics.settings.update(wandb=False, clearml=False, mlflow=False)` 全局关闭所有第三方日志集成，无需额外操作。

### 11.10 GPU 训练时显示 CPU

**现象**：日志显示 `CPU` 而非 GPU 设备名。

**解决**：
1. 确认宿主机已安装 NVIDIA 驱动：`nvidia-smi`
2. 确认已安装 NVIDIA Container Toolkit（见 [1.2 节](#12-gpu-版本额外要求)）
3. 确认 `docker-compose.yml` 包含 GPU 配置：
   ```yaml
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
   ```

### 11.11 忘记登录密码

查看 `config.json` 文件中的 `password` 字段：
```bash
cat config.json
```

### 11.12 RF-DETR 推理 FutureWarning

**现象**：推理时出现 `FutureWarning: Use a size-specific class...` 或 `Model is not optimized for inference`

**原因**：rfdetr 库内部的弃用提示和性能建议，不影响功能。

**解决**：无需处理，属于正常日志。

---

## 十二、完整部署流程示例（GPU 版 + RF-DETR）

以下是从零开始的完整部署流程：

```bash
# ============ 1. 服务器环境准备 ============
# 安装 Docker（如未安装）
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# 安装 NVIDIA Container Toolkit（GPU 版需要）
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证 GPU 可用
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi

# ============ 2. 上传项目代码 ============
# 在本地 Windows 执行（用 scp 上传整个目录）
scp -r yolo-train ubuntu@<服务器IP>:/home/ubuntu/yolo-train

# ============ 3. 准备 PyTorch wheel ============
cd /home/ubuntu/yolo-train
mkdir -p wheels_gpu
cd wheels_gpu
wget https://mirrors.aliyun.com/pytorch-wheels/cu118/torch-2.6.0%2Bcu118-cp310-cp310-linux_x86_64.whl
wget https://mirrors.aliyun.com/pytorch-wheels/cu118/torchvision-0.21.0%2Bcu118-cp310-cp310-linux_x86_64.whl
cd ..

# ============ 4. 修改配置 ============
# 从模板创建配置并修改默认密码
cp config.example.json config.json
nano config.json
# 将 "password" 改为强密码

# ============ 5. 构建并启动（含 RF-DETR） ============
docker compose --profile gpu build --build-arg INSTALL_RFDETR=true
docker compose --profile gpu up -d

# ============ 6. 验证 ============
# 查看日志
docker compose --profile gpu logs -f

# 验证 GPU
docker exec yolov11-gpu python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 验证 RF-DETR
docker exec yolov11-gpu python -c "import rfdetr; print('RF-DETR OK')"

# 浏览器访问
# http://<服务器IP>:7860
```

---

## 十三、参考信息

| 项目 | 版本 / 值 |
|------|-----------|
| Python | 3.10 |
| PyTorch（GPU） | 2.6.0+cu118 |
| PyTorch（CPU） | 2.2.2+cpu |
| CUDA | 11.8（GPU 版，通过 wheel + nvidia-* 子包） |
| Ultralytics | 8.3.x |
| RF-DETR | ≥1.12.0（可选，`INSTALL_RFDETR=true` 启用） |
| Gradio | 6.x |
| WebUI 端口 | 7860 |
| 镜像大小（GPU，含 RF-DETR） | ~6GB |
| 镜像大小（CPU） | ~3GB |
