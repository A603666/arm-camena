# 视觉 v2 配置迁移说明

## 适用范围

当你的 `pipeline_config.yaml` 仍使用 `vision_runtime` 根级平铺键（flat keys）时，需要执行本迁移。

当前版本仅支持分组结构：

- `stream`
- `service`
- `publisher`
- `build`
- `detector`
- `geometry`
- `tracking`
- `segmentation`
- `stability`
- `control`
- `rollout`

## 一次性迁移

```bash
cd 集成测试
python3 ./scripts/migrate_pipeline_vision_runtime_v2.py --config ./pipeline_config.yaml --in-place
```

脚本行为：

- 将 flat keys 迁移到对应分组路径
- 已存在分组值时，不覆盖，直接移除对应 flat key
- 默认生成备份：`pipeline_config.yaml.bak`

## 迁移后验证

```bash
cd 集成测试
python3 scripts/unified_config_loader.py --config ./pipeline_config.yaml --mode vision-env
./scripts/start_vision_arm64.sh --check --config ./pipeline_config.yaml
```

## 常见报错

如果看到以下错误，说明配置里仍有 flat keys 未迁移：

```text
vision_runtime contains deprecated flat keys
```

请重新执行迁移脚本，或手动改为分组结构后重试。
