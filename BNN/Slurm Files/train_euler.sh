#!/bin/bash -l
#SBATCH --job-name=qubit_train
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --constraint=EPYC_7763
#SBATCH --time=02:00:00

# ── Logging ────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"

# ── Umgebung laden ─────────────────────────────────────────────────────────
module load stack/2024-06 python/3.11.6
source $HOME/qubit_env/bin/activate

# ── Ins Projektverzeichnis wechseln ────────────────────────────────────────
cd $HOME/QHackathon

# ── Training starten ───────────────────────────────────────────────────────
echo "Starting ensemble training..."
python train_ensemble.py \
    --dataset dataset_10k_v2.npz \
    --out-dir models_10k_v2 \
    --n-members 5 \
    --n-epochs 200 \
    --batch-size 128 \
    --lr 1e-3 \
    --patience 40

echo "Training finished: $(date)"

# ── Zusammenfassung ────────────────────────────────────────────────────────
if [ -d models_10k ]; then
    echo "SUCCESS: models_10k_2/ created"
    ls -lh models_10_v2/
else
    echo "ERROR: models_10k_v2/ not found — check ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
fi