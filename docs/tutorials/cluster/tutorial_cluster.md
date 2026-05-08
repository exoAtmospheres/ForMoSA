# Tutorial 6 — Cluster / MPI Deployment

This tutorial shows how to run ForMoSA on an HPC cluster using MPI parallelism
with PyMultiNest. Two patterns are covered:

- **Pattern A — nohup** (single node, no job scheduler)
- **Pattern B — SLURM** (multi-node, job scheduler)

Both use the same Python run script.

---

## When do you need this?

The `nestle` backend runs in a single thread. It is fine for notebooks and
quick checks (≤ 3 free parameters, ≤ 100 live points). For production runs:

| Situation | Recommendation |
|-----------|----------------|
| ≤ 3 free parameters | `nestle`, single core |
| 4–6 free parameters, local machine | `nestle`, 200–500 live points |
| 4+ free parameters, cluster available | `pymultinest` + MPI |
| > 6 free parameters | `pymultinest` + MPI, ≥ 200 live points |

---

## Prerequisites

### Software

```bash
# Install PyMultiNest (see Installation page for full instructions)
pip install mpi4py
# MultiNest must be compiled separately — see docs/installation.rst

# Verify
python -c "import pymultinest; print('PyMultiNest OK')"
python -c "from mpi4py import MPI; print(f'mpi4py OK — {MPI.Get_library_version()}')"
```

### On the cluster: load required modules

Exact module names vary by cluster. Typical pattern:

```bash
module load openmpi/4.1.5     # or whatever version is available
module load gcc/12.2.0
module load anaconda/2023.09  # or miniconda
conda activate env_formosa
```

Check available modules with `module avail openmpi` and `module avail gcc`.

---

## The Python run script

Save this as `run_formosa.py` in your analysis directory. It auto-detects MPI
and runs serially if MPI is not available.

```python
"""
ForMoSA v2.0 — MPI-aware run script.

Usage:
  Single-core:
    python run_formosa.py

  Parallel (PyMultiNest):
    mpirun -np 12 python run_formosa.py
"""
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# ── MPI detection ──────────────────────────────────────────────────────────
# Imports must happen BEFORE any ForMoSA import to avoid fork-safety issues.
# MPI initialises the process model; importing multiprocessing-based libraries
# before MPI init can cause deadlocks on some systems.
try:
    from mpi4py import MPI
    COMM  = MPI.COMM_WORLD
    RANK  = COMM.Get_rank()
    SIZE  = COMM.Get_size()
    HAS_MPI = SIZE > 1
except ImportError:
    RANK, SIZE, HAS_MPI = 0, 1, False
    COMM = None

IS_ROOT = (RANK == 0)

# ═══════════════════════════════════════════════════════════════════
#  USER CONFIGURATION — edit this block
# ═══════════════════════════════════════════════════════════════════

WORK_PATH   = Path("/path/to/your/analysis/directory/")
CONFIG_FILE = "config.ini"        # relative to WORK_PATH
FILTER_PATH = Path("~/filters").expanduser()  # SVO filter cache

ADAPT   = False          # True: run adaptation (rank 0 only)
NPOINTS = 500            # live points — override config.ini value
NS_ALGO = "pymultinest"

# ═══════════════════════════════════════════════════════════════════


def log(msg: str) -> None:
    """Print only from rank 0 to avoid garbled output."""
    if IS_ROOT:
        prefix = f"[rank 0/{SIZE}]" if HAS_MPI else ""
        print(f"{prefix} {msg}", flush=True)


def main() -> None:
    t0 = time.time()

    from ForMoSA import Analysis
    from ForMoSA.config.global_config import ConfigLoader, Config_NS
    from ForMoSA.core.config import set_filter_path

    set_filter_path(FILTER_PATH)

    cfg = ConfigLoader(str(WORK_PATH / CONFIG_FILE))
    sections = cfg.load()

    cfg.config["config_inversion"].ns_algo  = NS_ALGO
    cfg.config["config_inversion"].npoints  = NPOINTS

    config_ns = Config_NS(
        nestle=cfg.config["config_nestle"],
        pymultinest=cfg.config["config_pymultinest"],
        ultranest=cfg.config["config_ultranest"],
    )

    # ── Step 1: Adaptation (rank 0 only) ──────────────────────────────────
    # Adaptation already uses an internal ThreadPool — MPI ranks do not help here.
    if IS_ROOT and ADAPT:
        log("STEP 1 — Grid adaptation")
        t1 = time.time()
        analysis = Analysis(cfg.config["config_path"], adapted=False, fitted=False)
        analysis.adapt(cfg.config["config_adapt"], cfg.config["config_inversion"])
        log(f"Adaptation done in {time.time()-t1:.1f}s")

    # Wait for rank 0 to finish adaptation before all ranks enter NS.
    if HAS_MPI:
        COMM.Barrier()

    # ── Step 2: Nested sampling (all ranks participate) ────────────────────
    # PyMultiNest handles inter-rank communication internally via MPI.
    # Every rank creates its own Analysis object pointing to the adapted grid.
    log("STEP 2 — Nested sampling")
    t2 = time.time()

    analysis = Analysis(cfg.config["config_path"], adapted=True, fitted=False)
    analysis.nested_sampling(
        cfg.config["config_parameters"],
        cfg.config["config_adapt"],
        cfg.config["config_inversion"],
        config_NS=config_ns,
    )

    if IS_ROOT:
        log(f"Nested sampling done in {time.time()-t2:.1f}s")

    if HAS_MPI:
        COMM.Barrier()

    # ── Step 3: Plotting (rank 0 only) ─────────────────────────────────────
    if IS_ROOT:
        log("STEP 3 — Plotting")
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for batch runs

        analysis = Analysis(cfg.config["config_path"], adapted=True, fitted=True)
        analysis.plot(analysis.ns.results, plot_native_model=False)
        log(f"Total wall time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
```

---

## Pattern A — nohup (single node)

Use `nohup` to run in the background and keep the process alive after you
disconnect from the node.

```bash
# 1. Load modules (adjust to your cluster)
module load openmpi/4.1.5 gcc/12.2.0
conda activate env_formosa

# 2. Navigate to your analysis directory
cd /path/to/your/analysis/

# 3. Run with MPI — safe ceiling is ~85% of available cores
#    (e.g., 12 processes on a 14-core node)
nohup mpirun -np 12 python run_formosa.py > run.log 2>&1 &

# 4. Save the process ID so you can kill it if needed
echo "PID: $!"

# 5. Monitor progress in real time
tail -f run.log

# 6. To stop the run
kill <PID>
```

```{note}
`nohup` keeps the process alive even if your SSH session drops. Pair it with
`screen` or `tmux` if you want an interactive terminal that survives reconnects.
```

---

## Pattern B — SLURM (multi-node)

Use SLURM when you need more than one node or when the cluster requires job scheduling.

Save this as `job.sh` in your analysis directory:

```bash
#!/bin/bash
#SBATCH --job-name=formosa
#SBATCH --nodes=2                 # number of nodes
#SBATCH --ntasks-per-node=16      # MPI ranks per node
#SBATCH --time=04:00:00           # wall-clock limit (HH:MM:SS)
#SBATCH --partition=compute       # partition / queue name
#SBATCH --output=formosa_%j.log   # %j = SLURM job ID

# ── Load environment ────────────────────────────────────────────────────────
module load openmpi/4.1.5 gcc/12.2.0
conda activate env_formosa

# ── Run ─────────────────────────────────────────────────────────────────────
# $SLURM_NTASKS = nodes × ntasks-per-node = total MPI rank count
cd /path/to/your/analysis/
mpirun -np $SLURM_NTASKS python run_formosa.py
```

Submit and monitor:

```bash
# Submit
sbatch job.sh

# Check status
squeue -u $USER

# Monitor output
tail -f formosa_<JOBID>.log

# Cancel
scancel <JOBID>
```

---

## Expected speedup

PyMultiNest scales well up to ~32 MPI ranks for typical ForMoSA problems
(4–6 free parameters, 300–500 live points). Beyond that, communication overhead
between ranks starts to dominate and speedup flattens.

| Ranks | Approx. speedup vs. single-core |
|-------|----------------------------------|
| 4     | ~3×                               |
| 8     | ~5×                               |
| 16    | ~8×                               |
| 32    | ~12×                              |

Rule of thumb for wall time: `t_parallel ≈ t_serial / (0.4 × n_ranks)`.
