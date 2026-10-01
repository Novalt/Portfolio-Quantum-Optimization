# Portfolio Quantum Optimization

Quantum portfolio selection using QAOA on real IBM Quantum hardware.

## Overview

This project implements a QAOA (Quantum Approximate Optimization Algorithm) to solve a combinatorial portfolio optimization problem, executed on real IBM Quantum hardware, not only on a simulator.

The goal is to select the optimal subset of assets (for example 3 out of 6) that minimizes covariance risk while maximizing expected return. This is a classic NP-hard combinatorial problem.

## Why quantum?

Classical optimizers struggle with combinatorial asset selection as the number of assets grows. QAOA maps the problem to a quantum circuit, explores the solution space in superposition and uses interference to raise the probability of good solutions.

## Versions

| File | Description |
|------|-------------|
| `src/mainX13IBM.py` | Fast baseline version: quick circuit execution |
| `src/mainX14IBM.py` | Advanced version with classical benchmarking, quality ranking and performance analysis |

## Results

| Metric | Value |
|--------|-------|
| Penalty factor | 35.0 |
| QAOA parameters | [0.7, 0.3, 0.5, 0.5] |
| Valid solutions | about 26% of shots |
| Optimal probability | 1.27% |

The classical optimal solution was found and validated by the quantum run.

## Installation

```bash
git clone https://github.com/Novalt/Portfolio-Quantum-Optimization.git
cd Portfolio-Quantum-Optimization
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### IBM Quantum access

You need your own IBM Quantum account and API key. Save your credentials locally by following IBM's current instructions for `QiskitRuntimeService.save_account`, or use the helper script `src/env-IBM-Cloud-pyAuthentication.py`.

Never commit API keys or credential files to the repository.

## Usage

```bash
# Recommended: full analysis with benchmarking
python src/mainX14IBM.py

# Fast version
python src/mainX13IBM.py
```

## Configuration

```python
CONFIG = {
    "NUM_ATIVOS": 6,              # Total assets
    "NUM_SELECIONAR": 3,          # Assets to select
    "PENALIDADE_FACTOR": 35.0,    # Constraint penalty
    "PARAMETROS_FIXOS": [0.7, 0.3, 0.5, 0.5],  # QAOA angles
    "NUM_SHOTS": 2048             # Quantum circuit executions
}
```

## Repository structure

```
Portfolio-Quantum-Optimization/
├── src/            # Main versions, IBM authentication helper and connection tests
├── Codes/          # QAOA notebook (QAOA.ipynb)
├── prototipos/     # Earlier prototype scripts (main.py to mainX12IBM.py)
├── requirements.txt
└── README.md
```

## Tech stack

Python, Qiskit, IBM Quantum Runtime, NumPy.
