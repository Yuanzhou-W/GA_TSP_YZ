# Adaptive Operator Genetic Algorithm for TSP (cn130)

> 🎯 **Project Focus**: Adaptive Operator Strategies in Genetic Algorithms
> 📌 **Problem**: Traveling Salesman Problem (TSP) – TSPLIB instance **cn130**
> 🧠 **Core Topic**: *Adaptive Selection, Crossover, and Mutation Operators*
> 🐍 **Language**: Python

---

## 1. Project Motivation

Traditional Genetic Algorithms (GAs) for TSP usually rely on **fixed operator strategies**, such as:

* Fixed crossover probability (Pc)
* Fixed mutation probability (Pm)
* Single selection method (e.g. roulette wheel or tournament)

However, recent studies indicate that such fixed strategies often suffer from:

* Premature convergence
* Loss of population diversity
* Inefficiency on medium-to-large TSP instances

📚 **Recent research (2023–2025)** shows that **adaptive operator strategies** significantly improve GA robustness and performance:

* Adaptive selection mechanisms better maintain diversity (PPHMJ)
* Dynamic Pc/Pm adjustment helps avoid early stagnation (技师学院学报)

This project aims to **systematically study, implement, and visualize** these adaptive strategies using the cn130 benchmark.

---

## 2. Research Objectives

This project is designed as a **learning + experimental + comparative** GitHub repository.

### Primary Objectives

1. Implement a **baseline GA** for TSP (fixed operators)
2. Design and implement **adaptive operator strategies**
3. Compare adaptive vs non-adaptive GA in terms of:

   * Convergence speed
   * Solution quality
   * Population diversity
   * Stability across runs

### Secondary Objectives

* Provide **clear visualization** of GA dynamics
* Offer **modular, extensible Python code**
* Serve as a **reference project** for evolutionary computation learning

---

## 3. Problem Description: TSP cn130

* Source: TSPLIB
* Number of cities: 130
* Distance type: Euclidean
* Known optimal solution exists (used for evaluation, not optimization)

The cn130 instance is large enough to:

* Expose premature convergence problems
* Highlight differences between operator strategies
* Remain computationally feasible in Python

---

## 4. Algorithm Overview

### 4.1 Baseline Genetic Algorithm (GA)

The baseline GA consists of:

1. **Encoding**: Permutation encoding (city order)
2. **Initialization**: Random permutation population
3. **Selection**: Fixed roulette wheel / tournament
4. **Crossover**: Fixed-probability PMX / OX
5. **Mutation**: Fixed-probability swap / inversion
6. **Replacement**: Elitism + generational replacement

This baseline serves as the **control group**.

---

## 5. Adaptive Operator Strategies (Core Contribution)

This project focuses on **Section 2.2: Adaptive Operators**.

### 5.1 Adaptive Selection Strategy

Instead of using a single fixed selection operator, we adopt a **multi-strategy adaptive selection**:

* Roulette Wheel Selection
* Stochastic Universal Sampling (SUS)

#### Strategy Mixing

At each generation:

* Selection methods are chosen probabilistically
* Probabilities are adjusted based on:

  * Population fitness variance
  * Improvement rate of best fitness

📌 **Motivation**:

* Roulette wheel → strong exploitation
* SUS → better diversity preservation

---

### 5.2 Adaptive Crossover Probability (Pc)

Rather than a fixed Pc, crossover probability is **dynamically adjusted**.

#### Typical Rule

* High diversity → lower Pc
* Low diversity / stagnation → higher Pc

Example:

```
Pc(t) = Pc_max - (Pc_max - Pc_min) * (σ_f / σ_f_max)
```

Where:

* σ_f = current population fitness standard deviation

📌 **Effect**:

* Encourages exploration when population becomes homogeneous
* Reduces disruption when diversity is sufficient

---

### 5.3 Adaptive Mutation Probability (Pm)

Mutation probability increases when the algorithm stagnates.

#### Trigger Conditions

* Best fitness unchanged for N generations
* Rapid loss of diversity

Example:

```
Pm(t) = Pm_min + (Pm_max - Pm_min) * stagnation_ratio
```

📌 **Effect**:

* Helps escape local optima
* Prevents early convergence

---

### 5.4 Diversity Measurement

Population diversity is explicitly measured using:

* Average pairwise Hamming distance
* Fitness variance

These metrics drive **adaptive decisions**.

---

## 6. Experimental Design

### 6.1 Experiment Groups

| Group            | Selection | Pc       | Pm       |
| ---------------- | --------- | -------- | -------- |
| GA-Basic         | Fixed     | Fixed    | Fixed    |
| GA-Adaptive-P    | Fixed     | Adaptive | Adaptive |
| GA-Adaptive-S    | Adaptive  | Fixed    | Fixed    |
| GA-Full-Adaptive | Adaptive  | Adaptive | Adaptive |

---

### 6.2 Evaluation Metrics

Each algorithm variant is evaluated over **multiple independent runs**:

1. Best tour length
2. Average tour length
3. Convergence speed
4. Population diversity curve
5. Stability (std over runs)

---

### 6.3 Visualization

Planned visual outputs:

* Fitness vs generation
* Diversity vs generation
* Pc / Pm evolution curves
* Best route visualization

---

## 7. Project Structure

```
Adaptive-Operator-GA-for-TSP/
│
├── data/
│   └── cn130.tsp
│
├── ga/
│   ├── encoding.py
│   ├── population.py
│   ├── selection.py
│   ├── crossover.py
│   ├── mutation.py
│   ├── adaptive.py
│   └── metrics.py
│
├── experiments/
│   ├── run_basic_ga.py
│   ├── run_adaptive_ga.py
│   └── config.yaml
│
├── visualization/
│   └── plots.py
│
├── results/
│
├── README.md
└── requirements.txt
```

---

## 8. Dependencies

* Python ≥ 3.9
* NumPy
* Matplotlib
* Pandas
* tqdm

---

## 9. References

* Adaptive multi-strategy selection in GA, PPHMJ
* Genetic algorithm with dynamic Pc/Pm, 技师学院学报
* Goldberg, *Genetic Algorithms in Search, Optimization, and Machine Learning*
* TSPLIB documentation



> 📌 This repository is intended for **educational, experimental, and research demonstration purposes**, with a strong emphasis on *understanding how adaptive operators change GA dynamics*.
