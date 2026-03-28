# NERO 手眼相机与夹爪坐标系说明

## 坐标系树

当前名义模型文件：

- `./模型文件/nero_description/urdf/nero_with_handeye_camera.xacro`

新增后的末端坐标系链为：

- `link7 -> end_effector -> flange_frame`
- `flange_frame -> camera_mount_frame`
- `flange_frame -> handeye_camera_link -> handeye_camera_optical_frame`
- `flange_frame -> gripper_mount_frame -> gripper_base_frame -> gripper_tcp_nominal`

语义约定：

- `flange_frame`
  - 作为 NERO 末端法兰语义坐标系。
  - 与当前模型里的 `end_effector` 共点共轴，用于承接相机外参和后续 TCP 偏移。
- `camera_mount_frame`
  - 基于相机支架法兰侧安装孔系得到。
  - 当前与 `flange_frame` 同轴，仅包含名义平移偏移。
- `handeye_camera_link`
  - 基于支架 STEP 中相机侧安装孔系和倾角得到。
  - 这是相机本体参考坐标系，不直接作为 optical frame 使用。
- `handeye_camera_optical_frame`
  - 由 `handeye_camera_link` 派生的 REP-103 光学坐标系。
  - 约定 `+Z` 朝前、`+X` 向右、`+Y` 向下。
- `gripper_mount_frame`
  - 当前名义实现与 `flange_frame` 重合。
  - 用来明确“夹爪挂载基准”和“相机挂载基准”是两条独立链。
- `gripper_base_frame`
  - 当前名义实现使用法兰适配件的 CAD 厚度作为刚体参考偏移。
- `gripper_tcp_nominal`
  - 固定几何 TCP。
  - 表示两指夹爪的名义抓取中心，不随开口宽度动态变化。

## 名义位姿来源

名义外参配置文件：

- `./模型文件/nero_description/config/handeye_extrinsics.yaml`

计算脚本：

- `./模型文件/nero_description/urdf/compute_nominal_extrinsics.py`

脚本当前使用的资料：

- 相机支架 STEP：
  - `./模型文件/奥比中光大白DC1标准相机支架 (1)/Piper夹爪相机支架.step`
- 法兰适配件 STEP：
  - `./模型文件/NERO+两指夹爪-法兰.STEP/NERO+两指夹爪-法兰.STEP`
- 相机手册：
  - `./模型文件/相机手册.pdf`

当前相机名义值来自以下 STEP 特征：

- `flange_frame`
  - 来自支架法兰侧参考圆。
- `camera_mount_frame`
  - 来自支架法兰侧 4 个安装孔的质心。
- `handeye_camera_link`
  - 来自相机侧 4 个安装孔的质心和安装面法向。

当前脚本输出的核心名义值为：

- `flange_frame -> camera_mount_frame`
  - `xyz = [0.0360, 0.0, -0.02625] m`
  - `rpy = [0.0, 0.0, 0.0] rad`
- `flange_frame -> handeye_camera_link`
  - `xyz = [0.0598795391, 0.0, -0.0288839165] m`
  - `rpy = [3.1409771866, 0.3490653300, 3.1397931476] rad`

这组值对应相机相对法兰存在约 `20 deg` 的名义下俯。

## Calibrated 覆盖规则

`handeye_extrinsics.yaml` 中保留了两层相机外参：

- `nominal_camera`
  - 由 xacro 模型直接挂到 `flange_frame` 上。
- `calibrated_camera`
  - 用于后续真实手眼标定结果。
  - 当前默认 `enabled: false`。

当前展示 launch：

- `./模型文件/nero_description/launch/display_handeye_camera.launch.py`

行为规则：

- 当 `publish_calibrated_camera=false` 时，只使用 `handeye_camera_link` 这条名义链。
- 当 `publish_calibrated_camera=true` 时，launch 会额外发布：
  - `flange_frame -> handeye_camera_calibrated`
- 推荐的运行习惯：
  - 调试支架/CAD 时看 `handeye_camera_link`
  - 接入真实标定后，算法优先消费 `handeye_camera_calibrated`

## TCP 说明

`gripper_tcp_nominal` 当前是固定名义 TCP，不随夹爪开闭实时变化。

当前实现假设：

- `gripper_mount_frame` 与 `flange_frame` 重合。
- `gripper_base_frame` 沿当前名义工具前向偏移法兰适配件厚度。
- `gripper_tcp_nominal` 再沿同一前向固定偏移 `0.1 m`。

这意味着：

- 它适合做可视化、流程接口和后续真实 TCP 标定的占位基准。
- 它不是“已做实物测量的最终抓取中心”。

如果后续拿到夹爪实测尺寸或更完整装配 CAD，优先更新：

- `gripper_nominal.xyz_m`
- `gripper_nominal.tcp_xyz_m`

## 验证方法

重新生成外参配置：

```bash
python ./模型文件/nero_description/urdf/compute_nominal_extrinsics.py
```

展开 xacro：

```bash
xacro ./模型文件/nero_description/urdf/nero_with_handeye_camera.xacro \
  extrinsics_file:=./模型文件/nero_description/config/handeye_extrinsics.yaml
```

检查 URDF：

```bash
check_urdf /tmp/nero_with_handeye_camera.urdf
```

当前验证结论：

- `xacro` 可成功展开。
- `check_urdf` 可成功解析。
- 新增 frame 已出现在末端链路中。
- 关键 joint 的 `xyz/rpy` 与 `handeye_extrinsics.yaml` 完全一致。
- `display_handeye_camera.launch.py` 可正常导入并生成 launch 描述。

一键本地验证脚本：

- `./validate_nero_handeye_setup.py`

运行方式：

```bash
python ./validate_nero_handeye_setup.py
```
