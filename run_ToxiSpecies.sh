#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

mkdir -p logs/ToxiSpecies
mkdir -p Models
mkdir -p "Results/Experiment setting/loss curve"
mkdir -p "Results/Experiment setting/seeds"

settings=(1 2 3 4)
gpus=(1 2 3 4)
lrs=(0.0001 0.0005 0.001)
episodes=60
# Evaluation seeds: reduced from default 20 to 10 to cut evaluation time
# while maintaining statistical stability. Variance reduction from 20 samples
# has diminishing returns; 10 is sufficient for stable mean and std.
eval_seeds=10

run_pipeline() {
  local setting="$1"
  local gpu="$2"
  local lr="$3"
  local lr_tag="lr${lr}"
  local setting_dir="logs/ToxiSpecies/${lr_tag}/${setting}"
  local pipeline_log="${setting_dir}/pipeline.log"
  local status_file="${setting_dir}/status.txt"
  local fa_log="${setting_dir}/FA.log"
  local la_log="${setting_dir}/LA.log"
  local da_log="${setting_dir}/DA.log"

  mkdir -p "${setting_dir}"

  nohup env \
    ROOT_DIR="${ROOT_DIR}" \
    SETTING="${setting}" \
    GPU="${gpu}" \
    LR="${lr}" \
    EPISODES="${episodes}" \
    EVAL_SEEDS="${eval_seeds}" \
    FA_LOG="${fa_log}" \
    LA_LOG="${la_log}" \
    DA_LOG="${da_log}" \
    STATUS_FILE="${status_file}" \
    bash -c '
      set -euo pipefail
      cd "$ROOT_DIR"

      printf "[%s] setting %s -> START\n" "$(date "+%F %T")" "$SETTING"
      printf "START\n" > "$STATUS_FILE"

      printf "[%s] setting %s -> Main_FA running on GPU %s\n" "$(date "+%F %T")" "$SETTING" "$GPU"
      printf "Main_FA running\n" > "$STATUS_FILE"
      CUDA_VISIBLE_DEVICES="$GPU" python -u Main_FA.py --setting "$SETTING" --base_lr "$LR" --meta_lr "$LR" --episodes "$EPISODES" --eval_seeds "$EVAL_SEEDS" > "$FA_LOG" 2>&1 &
      fa_pid=$!

      printf "[%s] setting %s -> Main_LA running on GPU %s\n" "$(date "+%F %T")" "$SETTING" "$GPU"
      printf "Main_FA running; Main_LA running\n" > "$STATUS_FILE"
      CUDA_VISIBLE_DEVICES="$GPU" python -u Main_LA.py --setting "$SETTING" --base_lr "$LR" --meta_lr "$LR" --episodes "$EPISODES" --eval_seeds "$EVAL_SEEDS" > "$LA_LOG" 2>&1 &
      la_pid=$!

      wait $fa_pid
      printf "[%s] setting %s -> Main_FA finished\n" "$(date "+%F %T")" "$SETTING"
      printf "Main_FA finished; Main_LA running\n" > "$STATUS_FILE"

      wait $la_pid
      printf "[%s] setting %s -> Main_LA finished\n" "$(date "+%F %T")" "$SETTING"
      printf "Main_FA finished; Main_LA finished\n" > "$STATUS_FILE"

      printf "[%s] setting %s -> Ensemble_DA running on GPU %s\n" "$(date "+%F %T")" "$SETTING" "$GPU"
      printf "Ensemble_DA running\n" > "$STATUS_FILE"
      CUDA_VISIBLE_DEVICES="$GPU" python -u Ensemble_DA.py --setting "$SETTING" --base_lr "$LR" --meta_lr "$LR" --episodes "$EPISODES" --eval_seeds "$EVAL_SEEDS" > "$DA_LOG" 2>&1

      printf "[%s] setting %s -> Ensemble_DA finished\n" "$(date "+%F %T")" "$SETTING"
      printf "DONE\n" > "$STATUS_FILE"

      printf "[%s] setting %s -> PIPELINE FINISHED\n" "$(date "+%F %T")" "$SETTING"
    ' > "${pipeline_log}" 2>&1 &
}

for lr in "${lrs[@]}"; do
  for i in "${!settings[@]}"; do
    run_pipeline "${settings[$i]}" "${gpus[$i]}" "$lr"
  done
done

wait

echo "All pipelines finished."

# pkill -f 'run_ToxiSpecies.sh|Main_FA.py|Main_LA.py|Ensemble_DA.py'
# nohup bash run_ToxiSpecies.sh > logs/run_ToxiSpecies.log 2>&1 &

