# Open-loop inference evaluation

`run_open_loop.py` replays a recorded LeRobot episode against a running OpenPI RLinf HTTP server. It compares returned absolute action chunks with recorded ground-truth actions and generates per-dimension metrics and plots.

The reader uses `LeRobotDataset`, including its per-episode and per-camera video metadata. This supports LeRobot v2.1 per-episode files and v3 packed parquet/video shards without deriving file names from episode numbers.

This is a teacher-forced diagnostic, not closed-loop robot evaluation: each inference request uses a recorded observation, so the result does not measure recovery from the model's own actions.

## Built-in tasks

Start the matching inference server, then run from the repository root:

```bash
source .venv/bin/activate

# Consume one returned 50-step chunk for each 50 recorded frames.
python toolkits/inference/open_loop/run_open_loop.py --tasks pour --episode 24
python toolkits/inference/open_loop/run_open_loop.py --tasks ring --episode 30

# First-action teacher-forced baseline.
python toolkits/inference/open_loop/run_open_loop.py \
  --tasks cook --execution-mode single_step
```

Built-in presets cover Dobot cook/pour/tidy/towel, XRobot ring, and Cobot cube. `--data-root`, `--server-host`, `--episode`, `--chunk-size`, and `--max-frames` can override runtime settings.

## Custom robot or task

Pass a YAML file with one or more task mappings:

```yaml
tasks:
  custom_pick:
    robot: custom_robot
    prompt: pick up the object
    port: 8030
    dataset: /path/to/lerobot_dataset
    cameras:
      high: observation.images.head
      left_wrist: observation.images.left_wrist
      right_wrist: observation.images.right_wrist
    # Optional for datasets whose raw vectors contain extra dimensions.
    state_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    action_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    action_dimensions:
      - left_0
      - left_1
      - left_2
      - left_3
      - left_4
      - left_5
      - left_gripper
      - right_0
      - right_1
      - right_2
      - right_3
      - right_4
      - right_5
      - right_gripper
    # Optional training/inference parity audit.
    infer_config: custom_inference.yaml
    train_config: custom_sft.yaml
```

```bash
python toolkits/inference/open_loop/run_open_loop.py \
  --task-config /path/to/tasks.yaml --tasks custom_pick
```

The dataset must expose `observation.state`, `action`, and the three configured video features. The HTTP server must accept the existing state plus three-camera payload and return an `actions` array.

## Alignment semantics

- `chunk` (default): at offsets `0, chunk_size, 2*chunk_size, ...`, read one metadata-aligned observation and its future GT action chunk, request one predicted chunk, and compare the consumed actions. LeRobot clamps the final GT horizon to the episode boundary; the evaluator consumes only the remaining frames.
- `single_step`: request inference for every recorded observation and compare the first returned action.

## Outputs

The default output directory is `toolkits/inference/open_loop/results/`, which is ignored by Git. Each run writes:

- `{task}_episode_{episode}_{mode}.npz`: prediction, GT, errors, chunk offsets and latencies.
- `{task}_episode_{episode}_{mode}_metrics.json`: aggregate/per-dimension error, server/config audit, dataset version/FPS/camera keys and episode boundaries.
- `{task}_episode_{episode}_{mode}.png`: stacked per-dimension GT and prediction curves.
- `open_loop_all_tasks_{mode}.png` and `summary_{mode}.json`: run-level summaries.

A scalar MAE mixes dimensions with different units. Inspect `per_dim_mae` and the individual curves before attributing errors to the model or dataset.
