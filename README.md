# Behavior Cloning for Robot Manipulation
本项目实现：

- BC（MLP）
- BC-RNN（GRU）
- BC-Transformer
- ACT

## DataSet(Robomimic)
本项目使用 [robomimic](https://robomimic.github.io/) 提供的数据集，目前主要使用以下两个任务：

- `Lift`：机械臂抓取并抬起方块
- `Can`：机械臂将罐子放入指定容器
运行命令：
```bash
python robomimic/scripts/download_datasets.py --tasks lift can --dataset_types ph --hdf5_types low_dim --download_dir ".\robomimic"
```
## train/evaluate
- BC:
  train:
  ```bash
    python -m scripts.train_bc --dataset ".\datasets\robomimic\lift\ph\low_dim_v15.hdf5" --output "checkpoints\best_bc_mlp.pth" --history-output "results\bc_mlp_loss_history.json"
  ```
  eval:
  ```bash
    python -m scripts.evaluate_bc --checkpoint "checkpoints\best_bc_mlp.pth" --dataset ".\datasets\robomimic\lift\ph\low_dim_v15.hdf5" --episodes 10 --horizon 400
   ```
- ACT:
  train:
  ```bash
    python -m scripts.train_act --dataset ".\datasets\robomimic\lift\ph\image_v15.hdf5" --output "checkpoints\best_deterministic_act.pth" --history-output "results\act_loss_history.json" --chunk-size 20 --kl-weight 10
  ```
  eval:
  ```bash
    python -m scripts.evaluate_act --checkpoint checkpoints/can_ph_cvae_act.pth --dataset data/can/ph/image_v15.hdf5 --episodes 10 --horizon 400 --execution-horizon 1 --temporal-ensemble --temporal-ensemble-coeff 0.01
  ```

