# YOLO11 视觉训练平台 - 部署指南

本文档详细介绍如何在 CPU / GPU 服务器上通过 Docker 部署 YOLO11 视觉训练平台。

---

## 一、环境要求

### 1.1 通用要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（推荐 Ubuntu 20.04 / 22.04） |
| Docker | 20.10+ |
| Docker Compose | v2+（`docker compose` 命令，非旧版 `docker-compose`） |
| 内存 | ≥ 4GB（训练建议 ≥ 8GB） |
| 磁盘 | ≥ 20GB 可用空间（镜像约 5GB + 训练数据 + 训练产物） |
| 端口 | 7860（WebUI 默认端口） |

### 1.2 GPU 版本额外要求

| 项目 | 要求 |
|------|------|
| NVIDIA GPU | 支持 CUDA 11.7 的 NVIDIA 显卡 |
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
sudo docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu20.04 nvidia-smi
```

---

## 二、项目文件结构

```
YOLOV11/
├── app.py                  # 主程序（Gradio WebUI）
├── config.json             # 配置文件（账号、主题、超时等）
├── docker-compose.yml      # Docker Compose 编排文件
├── Dockerfile.cpu          # CPU 版构建文件
├── Dockerfile.gpu          # GPU 版构建文件
├── .dockerignore           # Docker 构建忽略清单
├── requirements.txt        # Python 依赖清单
├── assets/                 # 网页图标等静态资源
├── ultralytics/            # Ultralytics 源码（含本地修改）
├── models/                 # 模型文件目录（.pt）
├── my_dataset/             # 数据集目录
├── runs/                   # 训练结果输出目录
└── wheels_gpu/             # GPU 版 PyTorch 本地 wheel（可选，加速构建）
    ├── torch-2.0.1+cu117-cp310-cp310-linux_x86_64.whl
    └── torchvision-0.15.2+cu117-cp310-cp310-linux_x86_64.whl
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
cd /path/to/YOLOV11

# 2. 构建并启动（首次会下载基础镜像，约 5-10 分钟）
docker compose --profile cpu up -d --build

# 3. 查看启动日志
docker compose --profile cpu logs -f
```

### 4.2 验证部署

浏览器访问 `http://<服务器IP>:7860`，使用 `config.json` 中的账号密码登录。

---

## 五、GPU 版本部署

### 5.1 准备 PyTorch Wheel 文件（推荐，避免构建超时）

GPU 版构建默认会从 PyTorch 官方源下载约 1.8GB 的 wheel，国内服务器可能超时。建议预先下载到 `wheels_gpu/` 目录：

```bash
# 在项目根目录创建 wheels_gpu 文件夹
mkdir -p wheels_gpu
cd wheels_gpu

# 下载 PyTorch 2.0.1+cu117（自带 CUDA 11.7 运行库，无需额外下载 nvidia-* 依赖）
# 方式一：PyTorch 官方源
wget https://download.pytorch.org/whl/cu117/torch-2.0.1%2Bcu117-cp310-cp310-linux_x86_64.whl
wget https://download.pytorch.org/whl/cu117/torchvision-0.15.2%2Bcu117-cp310-cp310-linux_x86_64.whl

# 方式二：阿里云镜像（国内推荐）
wget https://mirrors.aliyun.com/pytorch-wheels/cu117/torch-2.0.1%2Bcu117-cp310-cp310-linux_x86_64.whl
wget https://mirrors.aliyun.com/pytorch-wheels/cu117/torchvision-0.15.2%2Bcu117-cp310-cp310-linux_x86_64.whl

cd ..

# 验证文件存在
ls -lh wheels_gpu/
# 应显示两个 .whl 文件，文件名包含 cp310 和 cu117
```

> **重要**：wheel 文件名必须包含 `cp310-cp310-linux_x86_64` 和 `cu117`，否则安装会失败。

### 5.2 构建并启动

```bash
# 构建并启动（使用本地 wheel，构建约 3-5 分钟）
docker compose --profile gpu up -d --build

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
PyTorch: 2.0.1+cu117
CUDA available: True
GPU: NVIDIA GeForce RTX xxxx
```

---

## 六、常用操作

### 6.1 启停服务

```bash
# 启动（CPU / GPU 二选一）
docker compose --profile cpu up -d
docker compose --profile gpu up -d

# 停止
docker compose --profile cpu down
docker compose --profile gpu down

# 重启（修改 app.py / config.json 后生效）
docker compose --profile gpu restart
```

> **注意**：CPU 和 GPU 容器共享端口 7860，**不能同时运行**。

### 6.2 查看日志

```bash
# 实时日志
docker compose --profile gpu logs -f

# 最近 100 行
docker compose --profile gpu logs --tail 100

# 查看训练进度（训练日志会输出到容器日志）
docker compose --profile gpu logs -f | grep -E "epoch|train|val"
```

### 6.3 更新代码

项目通过 volume 挂载了 `app.py`、`config.json`、`assets/`、`ultralytics/`，修改这些文件**无需重新构建镜像**：

```bash
# 1. 上传新代码到服务器（覆盖原文件）
scp app.py ubuntu@<服务器IP>:/path/to/YOLOV11/app.py

# 2. 重启容器即可生效
docker compose --profile gpu restart
```

**仅当修改 Dockerfile 或依赖列表时**才需要重新构建：

```bash
docker compose --profile gpu up -d --build
```

### 6.4 容器内执行命令

```bash
# 进入容器交互式终端
docker exec -it yolov11-gpu bash

# 检查 GPU 状态
docker exec yolov11-gpu nvidia-smi

# 检查 Python 包版本
docker exec yolov11-gpu python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"

# 安装临时依赖（容器重启后失效，永久需写进 Dockerfile）
docker exec yolov11-gpu pip install <包名> -i https://mirrors.aliyun.com/pypi/simple/
```

---

## 七、数据持久化

以下目录通过 volume 挂载到宿主机，容器重建后数据不丢失：

| 容器路径 | 宿主机路径 | 用途 |
|----------|-----------|------|
| `/app/app.py` | `./app.py` | 主程序（热更新） |
| `/app/config.json` | `./config.json` | 配置文件（热更新） |
| `/app/assets` | `./assets` | 图标等资源（热更新） |
| `/app/ultralytics` | `./ultralytics` | Ultralytics 源码（热更新） |
| `/app/models` | `./models` | 模型文件（.pt） |
| `/app/my_dataset` | `./my_dataset` | 数据集 |
| `/app/runs` | `./runs` | 训练结果（曲线图、权重等） |

> **备份**：定期备份 `models/`、`my_dataset/`、`runs/` 三个目录即可保留所有训练成果。

---

## 八、数据集目录结构

数据集放在 `my_dataset/` 下，每个数据集一个子文件夹：

```
my_dataset/
└── dataset/                    # 数据集名（与文件夹名一致）
    ├── images/                 # 图片目录
    │   ├── train/              # 训练集图片（可选，也可直接放 images/ 下）
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

也支持上传压缩包自动解压（`.zip` / `.tar.gz` / `.tgz` / `.tar`），压缩包内需包含上述结构。

---

## 九、共享内存配置

PyTorch DataLoader 多进程加载需要共享内存。Docker 默认 `/dev/shm` 仅 64MB，会导致训练时报错：

```
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
```

`docker-compose.yml` 已配置 `shm_size: '8g'`，无需额外处理。如服务器内存不足，可调整为 `'4g'`。

---

## 十、常见问题排查

### 10.1 构建失败：pip 下载 403 错误

**现象**：`HTTP error 403 while getting https://pypi.tuna.tsinghua.edu.cn/...`

**原因**：清华 PyPI 镜像对部分包限流。

**解决**：Dockerfile 已改用阿里云源（`mirrors.aliyun.com/pypi/simple/`），如仍失败可尝试更换为其他镜像：
```dockerfile
# 中科大源
-i https://pypi.mirrors.ustc.edu.cn/simple/
# 腾讯云源
-i https://mirrors.cloud.tencent.com/pypi/simple/
```

### 10.2 构建失败：wheel not supported on this platform

**现象**：`torch-xxx.whl is not a supported wheel on this platform`

**原因**：wheel 文件的 Python 版本（如 cp311）与容器内 Python 版本（如 cp310）不匹配。

**解决**：确认 `wheels_gpu/` 中的 wheel 文件名包含 `cp310-cp310-linux_x86_64`（容器使用 Python 3.10）。

### 10.3 构建超时：PyTorch 下载卡住

**现象**：构建过程中 `Downloading torch-xxx.whl (780.4 MB)` 长时间不动。

**原因**：国内访问 PyTorch 官方源慢。

**解决**：使用 `wheels_gpu/` 目录预下载 wheel 文件（见 [第五章](#五gpu-版本部署)）。

### 10.4 训练报错：No space left on device

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

### 10.5 训练报错：insufficient shared memory

**现象**：`Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).`

**原因**：Docker 默认 `/dev/shm` 太小（64MB）。

**解决**：确认 `docker-compose.yml` 中已配置 `shm_size: '8g'`，然后重启容器：
```bash
docker compose --profile gpu down
docker compose --profile gpu up -d
```

### 10.6 训练报错：NumPy 2.x 不兼容

**现象**：`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`

**原因**：torch 2.0.1 用 NumPy 1.x 编译，但环境装了 NumPy 2.x。

**解决**：Dockerfile 已固定 `numpy<2`，如仍出现，在容器内手动降级：
```bash
docker exec yolov11-gpu pip install "numpy<2" -i https://mirrors.aliyun.com/pypi/simple/
```

### 10.7 训练时缺少 seaborn

**现象**：`No module named 'seaborn'`（不影响训练，但训练曲线图无法生成）

**解决**：Dockerfile 已添加 seaborn 依赖。临时修复：
```bash
docker exec yolov11-gpu pip install seaborn -i https://mirrors.aliyun.com/pypi/simple/
```

### 10.8 容器内 app.py 编码错误

**现象**：`SyntaxError: Non-UTF-8 code starting with '\xb6'`

**原因**：镜像构建时复制了编码损坏的 app.py。

**解决**：`docker-compose.yml` 已通过 volume 挂载宿主机的 app.py，确保使用最新文件：
```bash
docker compose --profile gpu down
docker compose --profile gpu up -d
```

### 10.9 GPU 训练时显示 CPU

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

### 10.10 忘记登录密码

查看 `config.json` 文件中的 `password` 字段：
```bash
cat config.json
```

---

## 十一、完整部署流程示例（GPU 版）

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
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu20.04 nvidia-smi

# ============ 2. 上传项目代码 ============
# 在本地 Windows 执行（用 scp 上传整个目录）
scp -r YOLOV11 ubuntu@<服务器IP>:/home/ubuntu/yolo-train

# ============ 3. 准备 PyTorch wheel ============
cd /home/ubuntu/yolo-train
mkdir -p wheels_gpu
cd wheels_gpu
wget https://mirrors.aliyun.com/pytorch-wheels/cu117/torch-2.0.1%2Bcu117-cp310-cp310-linux_x86_64.whl
wget https://mirrors.aliyun.com/pytorch-wheels/cu117/torchvision-0.15.2%2Bcu117-cp310-cp310-linux_x86_64.whl
cd ..

# ============ 4. 修改配置 ============
# 修改默认密码
nano config.json
# 将 "password" 改为强密码

# ============ 5. 构建并启动 ============
docker compose --profile gpu up -d --build

# ============ 6. 验证 ============
# 查看日志
docker compose --profile gpu logs -f

# 验证 GPU
docker exec yolov11-gpu python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 浏览器访问
# http://<服务器IP>:7860
```

---

## 十二、参考信息

| 项目 | 版本 / 值 |
|------|-----------|
| Python | 3.10 |
| PyTorch（GPU） | 2.0.1+cu117 |
| PyTorch（CPU） | 最新 CPU 版 |
| CUDA | 11.7（GPU 版，通过 wheel 自带） |
| Ultralytics | 8.3.x |
| Gradio | 6.x |
| WebUI 端口 | 7860 |
| 镜像大小（GPU） | ~5.2GB |
| 镜像大小（CPU） | ~2.5GB |
