# Scientific Machine Learning for Parametric Elasticity: FEM, PINN, FNO and PINO

This project studies a plane-stress rectangular plate with a circular hole under unit horizontal traction. The parameterized dataset varies only the hole radius, R in [0.20, 0.35], with fixed L=4, H=2, E=1, nu=0.3, hole center (2,1), and thickness 1.

## Results

Abaqus validation at R=0.25: max U1 = 4.335395 and max S11 = 3.046240.

Python FEM at R=0.25: max U1 = 4.510764 and U1 error vs Abaqus = 4.05%.

| Model | Mean U1 Error | Mean U2 Error | Internal PDE Residual |
|---|---:|---:|---:|
| FNO | 0.275% | 0.528% | 3.2821 |
| PINO | 0.662% | 1.189% | 0.0648 |

FNO worst U1 error was 0.409% and worst U2 error was 0.624%. PINO mean U1 error was 0.662%, mean U2 error was 1.189%, and internal equilibrium residual was 0.064823.

PINO reduced the internal equilibrium residual by approximately 50.6x compared with FNO, while FNO achieved higher displacement prediction accuracy.

Under the tested training configurations, the PINN baselines showed optimization difficulties in resolving the localized stress concentration.

The PINO result is an internal residual experiment and does not claim complete satisfaction of the circular-hole traction boundary.

The fem directory contains the Python FEM solver and dataset generator. The fno and pino directories contain trained-model scripts and CSV metrics. The data directory contains the parameterized displacement dataset and summary. The results directory contains representative FNO plots.
