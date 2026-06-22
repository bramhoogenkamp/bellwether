# Self-improving loop ledger

Train instances: 43, validation instances: 29.
Negative vs_average means the design beats the plain average. gap_to_oracle closer to zero means closer to the fully informed upper bound.

| design | train vs_avg | val vs_avg | val gap_to_oracle | decision |
|---|---|---|---|---|
| log-opinion pool (equilibrium) | -0.0006 | +0.0041 | +0.2959 | reject: won on train but not validation (likely overfit) |
| log pool extremized x1.73 | +0.0364 | +0.0465 | +0.3383 | reject: did not beat the average on train |
| sim market capped (exp1 settings) | +0.0015 | +0.0035 | +0.2953 | reject: did not beat the average on train |
| sim market uncapped (exp2 settings) | -0.0025 | -0.0094 | +0.2824 | accept: beats average on train and validation |
