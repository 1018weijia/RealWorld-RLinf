# Open-loop inference evaluation

`run_open_loop.py` replays recorded states and synchronized camera frames to a running OpenPI RLinf HTTP inference server. It compares the returned **absolute 14-D action chunk** with recorded absolute ground-truth actions and writes metrics and plots.

This is a teacher-forced diagnostic, not a closed-loop robot or environment evaluation: recorded observations are used at each request boundary, and it does not measure recovery from its own actions.

## Run

Start the matching inference servers, then run from the repository root:

```bash
source .venv/bin/activate
python toolkits/inference/open_loop/run_open_loop.py --tasks cook pour tidy towel
```

Useful options:

```bash
# Teacher-forced first-action baseline.
python toolkits/inference/open_loop/run_open_loop.py --tasks cook --execution-mode single_step

# Consume shorter action chunks and evaluate another recorded episode.
python toolkits/inference/open_loop/run_open_loop.py --tasks pour --episode 3 --chunk-size 25

# Use datasets or servers outside the defaults.
python toolkits/inference/open_loop/run_open_loop.py \
  --tasks cook \
  --data-root /path/to/datasets \
  --server-host 10.0.0.8
```

## Execution modes

- `chunk` (default): request an action chunk, compare its actions with the following recorded frames, then request the next chunk. This approximates a chunk-consuming client.
- `single_step`: request inference at every recorded observation and compare only the first action. Use it to diagnose horizon drift separately from chunk execution.

## Outputs

The default output directory is `toolkits/inference/open_loop/results/`, which is intentionally ignored by Git. Each run writes:

- `{task}_episode_{episode}_{mode}.npz`: predictions, ground truth, errors, chunk metadata, latencies, and returned chunks.
- `{task}_episode_{episode}_{mode}_metrics.json`: MAE/RMSE, per-dimension MAE, server metadata, and training/inference configuration audit.
- `{task}_episode_{episode}_{mode}.png`: per-dimension ground-truth and prediction plot.
- `open_loop_all_tasks_{mode}.png` and `summary_{mode}.json`: run-level summaries.

For Dobot tasks, the evaluator fails when the SFT and inference YAMLs disagree on action dimensions, horizon, OpenPI config, image/state settings, precision, prompt, or norm-stats path. The audit cannot detect command-line overrides used when the server was launched.

A scalar MAE mixes joints and grippers; inspect `per_dim_mae` and the per-dimension plots before attributing errors to a model or dataset.
