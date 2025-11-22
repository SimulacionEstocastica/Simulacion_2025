# About this project

## Physics–Informed Neural SDEs: Implementation of GBM and Heston

## Based on the Stochastic Physics-Informed Neural ODEs methodology (O’Leary, Paulson & Mesbah, 2023)

### 1. Introduction

This repository implements a simplified and pedagogical version of the Stochastic Physics-Informed Neural Ordinary Differential Equations (SPIN-ODEs) framework originally proposed by O’Leary, Paulson, and Mesbah (2023).

The original work introduces a general methodology for training neural models to learn stochastic dynamics while enforcing physical or structural constraints derived from stochastic differential equations (SDEs).

In our case, we extend the base example provided in the reference code —the Lotka–Volterra system— and implement two additional experiments based on widely used models in quantitative finance:

- Geometric Brownian Motion (GBM)

- Heston Stochastic Volatility Model

The goal is to demonstrate that the framework is modular, allowing the underlying dynamical system to be replaced without altering the core training architecture.

### 2. Motivation

The primary motivation behind this project is to explore whether the SPIN-ODE methodology can:

Reliably separate drift and diffusion within a neural training pipeline,

Learn parameters and trajectories even when the noise source is not directly observable,

Extend beyond the Lotka–Volterra example while preserving numerical stability, stochastic fidelity, and computational efficiency.

This repository serves as a pedagogical laboratory, where several elements of the original framework are simplified to facilitate study, visualization, and reproducibility.

### 3. Connection to the Original Work

Our implementation is built upon the reference application provided by Jared O’Leary, associated with the SPIN-ODE publication:

O'Leary, J., Paulson, J.A., & Mesbah, A. (2023).
Stochastic Physics-Informed Neural Ordinary Differential Equations.
University of California, Berkeley / The Ohio State University.

In particular:

We preserve the modular training structure,

We follow the same philosophy of splitting learning into train_g1 (drift) and train_g2 (diffusion),

We adapt the data generation pipeline to support GBM and Heston,

We simplify selected components to make the experiments easier to interpret and replicate.

### 4. General Framework Structure

#### 4.1 Drift–Diffusion Decomposition

Following the methodology of O’Leary et al., the training pipeline is divided into two complementary phases:

- Phase G1 (train_g1)

Learning the deterministic component of the SDE (the drift) using observed trajectories.

- Phase G2 (train_g2)

Learning the stochastic component (the diffusion) using the residual increments between the learned drift and the true state transitions.

This decomposition is valid only when the SDE admits a representation of the form:

$$
\begin{aligned}
dX_t &= f(X_t,t)\,dt + g(X_t,t)\,dW_t
\end{aligned}
$$

and when an explicit Euler–Maruyama simulator is available so that the drift and diffusion components can be separated from synthetic data.

### 5. Implemented Models

5.1 Geometric Brownian Motion (GBM)

$$
\begin{aligned}
dS_t &= \mu S_t dt + \sigma S_t dW_t
\end{aligned}
$$

Relevant characteristics:

Drift and diffusion are fully separable.

Serves as a baseline for numerical stability and accuracy of the framework.

Enables direct comparison with the closed-form analytical solution.

### 5.2 Heston Model

$$
\begin{aligned}
dS_t &= \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^{(1)},\\
dv_t &= \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^{(2)}
\end{aligned}
$$

​
with correlation

$$
\begin{aligned}
\rho &= \mathrm{corr}(dW_t^{(1)}, dW_t^{(2)})
\end{aligned}
$$

Key features:

Two-dimensional stochastic system with structured diffusion.

Requires consistency between correlated noise sources.

Tests the framework’s ability to learn coupled stochastic dynamics.

### 6. Code Architecture

The repository is structured as follows:

```bash
  SPINODE_extended/
├── src/
│   ├── dynamics.py          # CSA/LVE/SIR + BS + Heston
│   ├── train.py             # funciones genéricas de entrenamiento g1/g2
│   ├── reconstruct.py       # reconstruct_CSA + reconstruct_LVE/BS/Heston
│   ├── utils.py
│   └── __init__.py          # opcional, si quieres tratarlo como paquete
│
├── experiments/
│   ├── LVE/
│   │   ├── run_LVE.ipynb
│   │   ├── generate_LVE_data.ipynb
│   │   ├── LVE_data.ipynb
│   │   ├── data/            # .npy, .npz, etc. (ignorados en git si son pesados)
│   │   └── results/         # figuras, rmse, etc.
│   ├── BS/
│   │   ├── run_BS.ipynb
│   │   ├── generate_BS_data.ipynb
│   │   ├── BS_data.ipynb
│   │   ├── data/
│   │   └── results/
│   └── Heston/
│       ├── run_Heston.ipynb
│       ├── generate_Heston_data.ipynb
│       ├── Heston_data.ipynb
│       ├── data/
│       └── results/
│
├── README.md
├── requirements.txt / env.yml
└── .gitignore
```

#### 6.1 train_g1

This module trains the drift by fitting:

$$
\begin{aligned}
\widehat{f}_{\theta}(x,t) &\approx \frac{X_{t+\Delta t}-X_t}{\Delta t}
\end{aligned}
$$

#### 6.2 train_g2

With the drift already learned, the module estimates the noise structure, recovering:

$$
\begin{aligned}
\widehat{g}_{\phi}(x,t) &\approx
\frac{X_{t+\Delta t}-X_t-\widehat{f}_{\theta}(x,t)\,\Delta t}{\sqrt{\Delta t}}
\end{aligned}
$$

### 7. Expected Results

In GBM, the method recovers parameters close to the true
𝜇
μ and
𝜎
σ.

In Heston, the method reproduces coherent trajectories and captures the volatility-driven noise structure.

The approach is numerically stable and extensible to other separable drift–diffusion models.

### 8. Known Limitations

This methodology is not directly applicable when:

Drift and diffusion cannot be cleanly separated.

The diffusion term is not driven by Gaussian noise.

The dynamics involve jumps or non-Euler–Maruyama noise (e.g., Lévy processes).

The system exhibits degenerate diffusion that cannot be extracted from finite-difference increments.

### 9. Bibliographic Reference

This project draws upon and integrates ideas from the following works:

- **O'Leary, J., Paulson, J.A., & Mesbah, A. (2023).**  
  _Stochastic Physics-Informed Neural Ordinary Differential Equations._  
  University of California, Berkeley / The Ohio State University.

- **El Janati Elidrissi, Y., & Efstathiadis, G. (2023).**  
  _PINN-Based SDE Solver._  
  Harvard T.H. Chan School of Public Health.

- **Olguín, D. (2024).**  
  _The Math Behind the Magic: Neural Networks, Theory and Practice._  
  Encuentro Nacional de Ingeniería Matemática 2024,  
  with J. Fontbona, J. Maass, and C. Muñoz.
