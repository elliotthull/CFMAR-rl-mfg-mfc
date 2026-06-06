# Reinforcement Learning for Mean Field Games and Mean Field Control

A model-free reinforcement learning approach to **Mean Field Games (MFG)** and **Mean Field Control (MFC)**, built on a **two-timescale Q-learning** algorithm. The same core method recovers the competitive solution (MFG / Nash equilibrium), the cooperative solution (MFC / social optimum), and a mixed formulation that combines the two — selected simply by the speed at which the value function and the population distribution are learned.

## Overview

Mean field models describe large populations of interacting agents, where each agent's optimal behavior depends on the *distribution* of the whole population. These problems are classically solved through coupled forward–backward PDE systems that require full knowledge of the dynamics. This project instead **learns the optimal controls directly from simulation**, with no PDE solver.

The engine is a finite-horizon, **two-timescale Q-learning** scheme that maintains two coupled iterates — the action-value function `Q` and the mean-field distribution `μ` — and updates them at different rates `ρ_Q ∝ 1/n^ω_Q` and `ρ_μ ∝ 1/n^ω_μ`:

- **μ on the slow timescale** (`ω_μ > ω_Q`) → the **MFG** (Nash equilibrium) regime.
- **μ on the fast timescale** (`ω_μ < ω_Q`) → the **MFC** (social optimum) regime.
- **local and global mean fields on separate timescales** → a **mixed MFG/MFC** model (a Mean Field Control Game).

I would like to share 3 results:

1. The **Mean Field Game** solution of an optimal-execution / price-impact problem (the main result).
2. A **mixed MFG/MFC** example (Mean Field Control Game).
3. A **basic Mean Field Control** result on a 2×2 toy model.

---

## 1. Mean Field Game — optimal execution under price impact

The headline result formulates an optimal execution / price-impact problem as a **Mean Field Game**: a large population of traders each liquidate a position over a finite horizon, and every trader's own activity moves the price, so each agent's best trading rate depends on the aggregate behavior of the crowd. The equilibrium is found two independent ways and cross-checked against the closed-form optimum:

- Online RL Q-learning — `notebooks/05_price_impact_mfg_rl.ipynb` — learns the equilibrium controls from simulation over up to 1,000,000 episodes, with no model dynamics supplied to the learner.
- Idealized MFG fixed-point — `notebooks/06_price_impact_mfg_idealized.ipynb` — a model-based trajectory-relaxation solver used as a reference, with convergence diagnostics.

`src/control_overlay_plots.py` overlays both solutions against the analytical optimum to produce the headline figure:

![Learned MFG controls vs. theoretical optimum](results/controls_overlay.png)

The RL-learned controls track both the idealized MFG solution and the theoretical optimal controls across the state space at `t = 0`, `7/16`, and `15/16`, confirming that the model-free agent converges to the correct mean-field equilibrium.

![Execution: learned vs optimal](results/execution_learned_vs_optimal.png)

## 2. Mixed MFG/MFC — a Mean Field Control Game

`notebooks/04_mixed_mfg_mfc.ipynb` implements a **Mean Field Control Game (MFCG)** — a model that *mixes* the competitive and cooperative settings. Agents are organized so that they **cooperate within their own population (MFC-like)** while **competing across populations (MFG-like)**.

This is realized with **three timescales**: the value function `Q` (fastest), a **local** mean field `μ_local` (intermediate), and a **global** mean field `μ_global` (slowest). The notebook tracks the convergence of both mean fields:

```
ω_Q = 0.55   <   ω_μ,local = 0.75   <   ω_μ,global = 0.95
(Q fastest)        (local μ slower)       (global μ slowest)
```

The result is a convergence plot of the local and global occupancy measures, showing the mixed equilibrium settling as the three iterates separate cleanly across timescales.

## 3. Basic Mean Field Control — 2×2 toy model

`src/mfc_2x2_example.py` is a minimal, fully self-contained **MFC** example: two states with two actions (`stay` / `move`) and a congestion cost that grows with the occupancy of the destination. Soft-min Q-learning is run with the distribution `μ` on the **fast** timescale — the **mean-field control (cooperative)** regime — and learns both the stationary occupancy measure and the optimal policy.

![2x2 MFC occupancy-measure convergence](results/mfc_2x2_convergence.png)


## Repository structure

```
rl-mfg-mfc/
├── notebooks/
│   ├── 04_mixed_mfg_mfc.ipynb            # Mixed MFG/MFC — Mean Field Control Game
│   ├── 05_price_impact_mfg_rl.ipynb      # ★ MFG finale: price impact learned via RL (1M episodes)
│   └── 06_price_impact_mfg_idealized.ipynb # Idealized MFG fixed-point reference + convergence
├── src/
│   ├── mfc_2x2_example.py                # Basic 2×2 mean-field CONTROL example (self-contained)
│   ├── control_overlay_plots.py          # Builds the headline MFG overlay figure
│   
├── results/
│   ├── controls_overlay.png              # Learned vs. idealized vs. theoretical (MFG)
│   ├── execution_learned_vs_optimal.png
│   ├── mfc_2x2_convergence.png           # 2×2 MFC occupancy-measure convergence
│   └── learned_controls.csv              # Learned vs. theoretical controls (tabular)
├── docs/                                 # Poster and presentation

```

The repository also contains foundational notebooks (a tabular-RL gridworld and introductory mean-field examples) and an MFC formulation of the execution problem; these are supporting material and are not covered above.

## About

Undergraduate research project on reinforcement learning for mean field games and control.

*Author: Elliott Hull and Derrick Chan* · *Advisor: Professor John Pierre Fouque*
