# Portfolio Quantum Optimization 🔬

> Quantum portfolio selection using QAOA on real IBM Quantum hardware.

## Overview

This project implements a **QAOA (Quantum Approximate Optimization Algorithm)** to solve a combinatorial portfolio optimization problem executed on **real IBM Quantum hardware** — not a simulator.

The goal: select the optimal subset of assets (e.g. 3 out of 6) that **minimizes covariance risk** while **maximizing expected return** — a classic NP-hard combinatorial problem that quantum computing is uniquely positioned to address.

## Why Quantum?

Classical portfolio optimizers struggle with combinatorial asset selection at scale. QAOA maps the optimization problem to a quantum circuit, exploring the solution space in superposition and leveraging quantum interference to amplify the probability of optimal solutions.

## Versions

| File | Description |
|------|-------------|
| `mainX13IBM.py` | Fast baseline version — quick circuit execution |
| `mainX14IBM.py` | Advanced version with classical benchmarking, quality ranking, and performance analysis |

## Results

```
Optimal Configuration Found
Penalty Factor:     35.0
QAOA Parameters:    [0.7, 0.3, 0.5, 0.5]
Valid Solutions:    ~26% of shots
Optimal Probability: 1.27%
Performance Rating: ✅ GOOD
```

Classical optimal solution was found and validated by the quantum run.

## Installation

```bash
git clone https://github.com/Novalt/Portfolio-Quantum-Optimization.git
cd Portfolio-Quantum-Optimization
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## IBM Quantum Setup

```bash
python env-IBM-Cloud-pyAuthentication.py
```

Or manually:
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
```

## Usage

```bash
# Recommended — full analysis with benchmarking
python src/mainX14IBM.py

# Fast version
python src/mainX13IBM.py
```

## Configuration

```python
CONFIG = {
    "NUM_ATIVOS": 6,           # Total assets
    "NUM_SELECIONAR": 3,       # Assets to select
    "PENALIDADE_FACTOR": 35.0, # Constraint penalty
    "PARAMETROS_FIXOS": [0.7, 0.3, 0.5, 0.5],  # QAOA angles
    "NUM_SHOTS": 2048          # Quantum circuit executions
}
```

## Tech Stack

- **Python** — Core implementation
- **Qiskit** — Quantum circuit construction
- **IBM Quantum Runtime** — Real quantum hardware execution
- **NumPy** — Matrix operations and covariance computation

## Structure

```
Portfolio-Quantum-Optimization/
├── src/
│   ├── mainX13IBM.py          # Baseline version
│   ├── mainX14IBM.py          # Advanced version
│   └── utils.py               # Helper functions
├── env-IBM-Cloud-pyAuthentication.py
├── requirements.txt
└── README.md
```

---

*Executed on IBM Quantum hardware — not a simulation.*
