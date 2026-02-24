## 基本信息
项目是从github上的同名项目移植而来

## 环境配置
```bash
uv sync
```

## 效果演示
```python
python game_render.py
```
## 运行脚本
python game_render.py --agent_type AGENT_TYPE --model_name MODEL_NAME
AGENT_TYPE: 选择使用的智能体类型，如ppo、ddqn、dddqn
MODEL_NAME: 选择使用的模型权重文件名
运行我们的最佳模型：
```bash
python game_render.py --agent_type DDQN --model_name test_1
```


## 训练脚本

### ppo
```bash
python train_ppo_vec.py
或 python train_ppo_vec_mlx.py (mac上训练使用)
```

### ddqn
python TRAIN_SCRIPT --episodes NUMBER --name MODEL_NAME
TRAIN_SCRIPT: 选择训练脚本，如train_cnn_dddqn.py、train_acnn_dddqn.py、train_dddqn_agent.py
NUMBER: 训练的总回合数
MODEL_NAME: 训练完成后保存的模型权重文件名

## 模型权重文件
在 agents/agent_zoo下