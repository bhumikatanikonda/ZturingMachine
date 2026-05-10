#!/bin/bash -l
#SBATCH --job-name=qubit_infer
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --constraint=EPYC_7763
#SBATCH --time=00:15:00

# ── Logging ────────────────────────────────────────────────────────────────
echo "Inference started: $(date)"
echo "Running on node: $(hostname)"

# ── Umgebung laden ─────────────────────────────────────────────────────────
# Basierend auf deinem Training-Skript laden wir die exakt gleiche Umgebung
module load stack/2024-06 python/3.11.6
source $HOME/qubit_env/bin/activate

# ── Ins Projektverzeichnis wechseln ────────────────────────────────────────
cd $HOME/QHackathon

# ── Inferenz starten ───────────────────────────────────────────────────────
# Wir testen hier beispielhaft einige Seeds, die nicht im 10k-Training waren
SEEDS=(10001 10002 10003 12345)

echo "Starting inference for multiple seeds..."
echo "Model directory: models_10k_v2"
echo "----------------------------------------------------------------------"

for SEED in "${SEEDS[@]}"
do
    echo "Processing SEED: $SEED"
    python infer_qubit.py \
        --model-dir models_10k_v2 \
        --seed $SEED
    echo "----------------------------------------------------------------------"
done

echo "Inference finished: $(date)"