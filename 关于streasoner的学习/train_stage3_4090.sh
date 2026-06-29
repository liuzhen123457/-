#!/bin/bash

set -x

# 修改为你的阶段二训练输出路径（Qwen3-4B 经过 Stage1+2 的模型）
MODEL_PATH=./output/Qwen3-4B-stage2   # 请根据实际路径调整

python3 -m src.EasyR1.verl.trainer.main \
    config=./src/EasyR1/examples/config.yaml \
    data.train_files=./data/ST-Bench/ST-RL/etiological_rl.jsonl,./data/ST-Bench/ST-RL/forecasting_rl.jsonl,./data/ST-Bench/ST-RL/correlation_rl.jsonl,./data/ST-Bench/ST-RL/entity_rl.jsonl \
    data.val_files=./data/ST-Bench/ST-Test/etiological_test.jsonl,./data/ST-Bench/ST-Test/forecasting_test.jsonl,./data/ST-Bench/ST-Test/correlation_test.jsonl,./data/ST-Bench/ST-Test/entity_test.jsonl \
    data.prompt_key=input \
    data.ts_key=timeseries \
    data.answer_key=output \
    data.format_prompt=./src/EasyR1/examples/format_prompt/str.jinja \
    data.rollout_batch_size=16 \
    data.val_batch_size=16 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.model.trust_remote_code=true \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.optim.lr_warmup_ratio=0.2 \
    worker.reward.reward_function=./src/EasyR1/examples/reward_function/str.py:compute_score \
    worker.rollout.n=2 \
    trainer.experiment_name=qwen3_4b_grpo_stage3 \
    trainer.n_gpus_per_node=1 \
    trainer.find_last_checkpoint=false \
    trainer.total_epochs=1 \
    trainer.save_freq=100 \
    trainer.save_limit=1 \
    algorithm.enable_spatial_reward=true \
    algorithm.spatial_reward_weight=0.1 \
    data.enable_spatial_reward=true