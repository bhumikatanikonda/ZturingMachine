#!/bin/bash -l
#SBATCH --job-name=qubit_dataset
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=56
#SBATCH --ntasks-per-node=56
#SBATCH --cpus-per-task=1
#SBATCH --constraint=EPYC_7763
#SBATCH --time=01:00:00

# ── Logging ────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "Tasks allocated: $SLURM_NTASKS"

# ── Umgebung laden ─────────────────────────────────────────────────────────
module load stack/2024-06 python/3.11.6
source $HOME/qubit_env/bin/activate

# ── Ins Projektverzeichnis wechseln ────────────────────────────────────────
cd $HOME/QHackathon

# ── Dataset generieren ─────────────────────────────────────────────────────
echo "Starting dataset generation..."
python generate_dataset.py \
    --n-qubits 10000 \
    --workers 56 \
    --out dataset_10k_v2.npz


echo "Dataset generation finished: $(date)"

# ── Kurze Zusammenfassung ──────────────────────────────────────────────────
if [ -f dataset_10k.npz ]; then
    echo "SUCCESS: dataset_10k_v2.npz created"
    ls -lh dataset_10k_v2.npz
else
    echo "ERROR: dataset_10k.npz not found — check ${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
fi