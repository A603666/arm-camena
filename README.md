# 集成测试 Jetson ARM64 一体化运行包（四点位流程版）

本目录是可直接拷贝部署的单文件夹运行包。将整个 `集成测试` 复制到目标 Jetson 后，即可运行：

- 动态抓取控制（`dynamic_grasp`）
- 视觉服务与相机发布（`camera_runtime`）
- NERO 机械臂运行与调试 CLI（`robot_runtime`）
- 官方 `pyAgxArm`、Orbbec ARM64 SDK、本地模型文件

本文档是详细操作说明，覆盖部署、配置、启动、联调、安全约束与排障。

---

## 1. 当前业务流程（四点位）

系统当前业务流程固定为：

`到预备 -> 预备开夹爪 -> 到夹取 -> 关闭夹爪 -> 回预备 -> 到搬运 -> 到倾倒 -> 开夹爪 -> 回搬运 -> 回预备`

对应点位命名：

- `ready`：预备
- `pick`：夹取
- `transport`：搬运
- `dump`：倾倒

关键安全约束：

- 关节 7 必须满足 **`J7 > 10°`**（严格大于，`=10°` 视为不通过）。
- 该约束在“点位加载时”和“运行时关键运动后”都会检查。

---

## 2. 目录结构

- `run_dynamic_grasp.py`：动态抓取主入口
- `dynamic_grasp/`：视觉轮询、手眼变换、抓取状态机、机械臂桥接
- `robot_runtime/`：机械臂 CLI、自动使能 daemon、默认配置
- `camera_runtime/`：视觉服务、ARM64 相机发布器、`yolo26n.pt`
- `vendor/`：`pyAgxArm` 与 ARM64 Orbbec SDK 最小运行子集
- `scripts/`：启动和部署脚本
- `模型文件/`：手眼外参、URDF、CAD 与手册
- `tests/`：动态抓取/桥接/配置等单测

---

## 3. 一次性环境准备

### 3.1 系统依赖

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libopencv-dev libzmq3-dev python3-pip python3-venv
```

### 3.2 视觉 Python 依赖

```bash
cd 集成测试/camera_runtime/vision_service
python3 -m pip install -r requirements.txt
```

说明：

- Jetson 侧请先准备与 CUDA 匹配的 `torch/torchvision`，再安装 `ultralytics`。
- 若跑视觉全量测试提示缺少 `scikit-learn`，按环境补装即可。

---

## 4. 一次性设备准备

### 4.1 安装相机 udev 规则

```bash
cd 集成测试/vendor/OrbbecSDK/misc/scripts
sudo chmod +x ./install_udev_rules.sh
sudo ./install_udev_rules.sh
sudo udevadm control --reload
sudo udevadm trigger
```

### 4.2 USB-CAN 端口绑定

默认 USB-CAN 端口在 `robot_runtime/config/auto_enable.yaml` 中配置为固定 `usb_bus_info`。

如果现场接口变化，请先改该配置再启动自动使能。

---

## 5. 配置说明

### 5.1 动态抓取配置 `dynamic_grasp_config.yaml`

重点字段：

- `vision`：视觉服务地址、轮询频率、健康检查参数
- `handeye`：手眼参数路径
- `scan/grasp`：扫描收敛、下探安全高度、抓取验证区间
- `route`：流程点位来源
  - `ready_from: threepoint.ready`
  - `transport_from: threepoint.transport`
  - `dump_from: threepoint.dump`

兼容性：

- `transport_from` 是新字段
- 若旧配置仍使用 `dump_pre_from`，程序会自动作为 `transport_from` 读取

### 5.2 机械臂默认配置 `robot_runtime/config/default.yaml`

`threepoint` 节点已升级为四点位（字段名保持兼容）：

- `strict_down_enabled: false`
  - 默认不锁死 `pick` 姿态，降低奇异解/无解风险
- `min_joint7_deg: 10.0`
  - 严格执行 `J7 > 10°`
- 四个示教点：`ready/pick/transport/dump`
  - 均包含 `joints_deg` 与 `pose_mm_deg`
  - 已写入你最新示教数据

---

## 6. 启动顺序

### 6.1 检查视觉环境

```bash
cd 集成测试
./scripts/start_vision_arm64.sh --check
```

### 6.2 启动自动使能 daemon

前台运行：

```bash
cd 集成测试
python3 robot_runtime/nero_auto_enable_daemon.py --config robot_runtime/config/auto_enable.yaml
```

或安装 systemd 服务：

```bash
cd 集成测试
./scripts/install_auto_enable_service.sh
```

### 6.3 启动视觉服务

```bash
cd 集成测试
./scripts/start_vision_arm64.sh
```

默认面板地址：

```text
http://127.0.0.1:18000/
```

### 6.4 启动动态抓取主程序

```bash
cd 集成测试
./scripts/start_dynamic_grasp.sh
```

常用参数：

```bash
./scripts/start_dynamic_grasp.sh --mode auto
./scripts/start_dynamic_grasp.sh --once
./scripts/start_dynamic_grasp.sh --vision-url http://127.0.0.1:18080
```

---

## 7. 手动联调（NERO CLI）

进入 CLI：

```bash
cd 集成测试
python3 robot_runtime/nero_test_cli.py --config robot_runtime/config/default.yaml
```

常用命令：

- `status`：查看关节与法兰反馈
- `precheck`：运行前检查（包含点位完整性与 `J7 > 10°` 校验）
- `show points`：打印四点位参数
- `run threepoint step`：四点位分步执行
- `run threepoint auto`：四点位自动执行
- `open` / `close`：测试夹爪
- `estop`：急停

`run threepoint` 实际顺序：

`ready -> open -> pick -> close -> save_closed_state -> ready -> transport -> dump -> open -> transport -> ready`

---

## 8. 安全与失败恢复行为

系统内置安全机制：

- 每次运动前检查会话健康（使能状态、反馈活性）
- 关键运动后检查 `J7 > 10°`
- 动态抓取失败时自动执行回退到 `ready`
- 键盘中断时执行 `estop + recover`（如驱动支持）

仍需注意：

- 以上逻辑不能替代现场物理安全措施
- 首次联调务必空载、低速、旁站
- 出现异常先 `estop`，再定位原因

---

## 9. 校验与测试

### 9.1 集成单测

```bash
cd 集成测试
python3 -m unittest discover -s tests -v
```

### 9.2 视觉配置测试

```bash
cd 集成测试
PYTHONPATH=./camera_runtime python3 -m pytest camera_runtime/vision_service/tests/test_config.py -q
```

### 9.3 手眼链路验证

```bash
cd 集成测试
python3 validate_nero_handeye_setup.py
```

---

## 10. 常见问题与排障

模型找不到：

- 确认 `camera_runtime/yolo26n.pt` 存在
- 或设置 `export DABAI_YOLO_MODEL=/your/model.pt`

相机动态库找不到：

- 确认 `vendor/OrbbecSDK/lib/arm64` 存在
- 通过 `./scripts/start_vision_arm64.sh` 启动，让脚本自动设置 `LD_LIBRARY_PATH`

USB-CAN 不匹配：

- 检查 `robot_runtime/config/auto_enable.yaml` 的 `usb_bus_info`
- 确认实际 USB 端口未变化

自动使能未恢复：

- 先前台执行一次：
  `python3 robot_runtime/nero_auto_enable_daemon.py --once`
- 再看服务状态：
  `systemctl status nero-auto-enable.service --no-pager`

`J7` 校验失败：

- 程序日志会标出当前 `J7` 实测角度
- 若小于等于 10°，需重新示教该点位，确保 `J7 > 10°`

---

## 11. 推荐联调流程

1. 先跑 `precheck`，确认硬件反馈与点位校验通过。  
2. 使用 `run threepoint step` 单步确认轨迹与姿态。  
3. 确认无碰撞风险后，再切 `run threepoint auto`。  
4. 最后启动 `start_dynamic_grasp.sh --mode auto` 做闭环抓取。  
