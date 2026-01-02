# 🧬 GA-TSP-cn130

**A Comparative Study of Classical and Adaptive Genetic Algorithms on TSP**

> 本项目以 TSPLIB 中的 **cn130 Traveling Salesman Problem** 为测试算例，
> 系统性对比了多种经典遗传算法与自适应遗传算法在 **收敛性能、稳定性与路径结构** 方面的差异。

---

## ✨ 项目亮点（Why this project matters）

* ✅ **四种 GA 的严格对照实验设计**
* ✅ **自适应算子策略（Adaptive Operators）**
* ✅ **多次运行下的路径结构稳定性分析**
* ✅ **研究级可视化（收敛、路径、边频率）**
* ✅ **工程化实现，可复现实验**

---

## 🧪 算法对比设置

本项目实现并对比了以下四种遗传算法：

| 编号   | 算法名称             | 参数策略 | 选择策略     | 研究目的      |
| ---- | ---------------- | ---- | -------- | --------- |
| GA-1 | Classic GA       | 固定   | Roulette | 基线方法      |
| GA-2 | Classic GA + SUS | 固定   | SUS      | 分析选择机制影响  |
| GA-3 | Semi-Adaptive GA | 自适应  | 固定       | 分析参数自适应影响 |
| GA-4 | Adaptive GA      | 自适应  | 自适应      | 综合改进方法    |

---

## 📁 项目结构

```text
GA_TSP_YZ/
│  main.py
│  README.md
│  requirements.txt
│
├─data/
│   └─ ch130.tsp
│
├─experiment/
│   └─ run_experiment.py          # 一键运行四种 GA
│
├─ga/
│   ├─ engine.py                  # 通用 GA 引擎
│   ├─ selection.py
│   ├─ crossover.py
│   ├─ mutation.py
│   ├─ metrics.py
│   └─ strategies/
│       ├─ base.py
│       ├─ classic.py
│       ├─ classic_sus.py
│       ├─ semi_adaptive.py
│       └─ adaptive.py
│
├─analysis/
│   ├─ analysis.py                # 多算法性能对比图
│   ├─ show_route_and_convergence.py
│   ├─ compare_routes_multi_ga.py
│   ├─ path_stability_overlay.py
│   └─ compare_edge_frequency_multi_ga.py
│
└─results/
    └─ experiments/               # 实验自动输出
```

---

## 🚀 快速开始（One-Command Run）

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

---

### 2️⃣ 一键运行所有 GA 对比实验 ⭐（推荐）

```bash
python experiment/run_experiment.py
```

运行后将自动：

* 在 **同一 cn130 实例** 上
* 依次运行 **4 种遗传算法**
* 保存完整实验日志到：

```text
results/experiments/
├─ ClassicGA/
├─ ClassicGA_SUS/
├─ SemiAdaptiveGA/
└─ AdaptiveGA/
```

```


```


---

## 📊 结果分析与可视化

> 所有可视化脚本均 **只读结果文件，不重新跑 GA**

---

### 🔹 1. 收敛曲线 & 稳定性对比

```bash
python analysis/analysis.py
```

生成图像：

* 收敛曲线对比（fitness_convergence）
* 多次运行稳定性箱线图
* 运行时间对比

---

### 🔹 2. 单算法：路径 + 收敛联合展示

```bash
python analysis/show_route_and_convergence.py \
  --tsp data/ch130.tsp \
  --result experiment_results/experiments/AdaptiveGA/run_001.json
```

📌 一张图同时展示：

* 最优路径
* 收敛过程

---

### 🔹 3. 多算法最优路径同图对比（直观）

```bash
python analysis/compare_routes_multi_ga.py \
  --tsp data/ch130.tsp \
  --experiment_results \
    experiment_results/experiments/ClassicGA/run_001.json \
    experiment_results/experiments/ClassicGA_SUS/run_001.json \
    experiment_results/experiments/SemiAdaptiveGA/run_001.json \
    experiment_results/experiments/AdaptiveGA/run_001.json
```

📌 **非常适合答辩 / PPT 展示**

---

### 🔹 4. 多次运行路径稳定性分析（高级）

#### （1）路径叠加透明图

```bash
python analysis/path_stability_overlay.py \
  --tsp data/ch130.tsp \
  --experiment_results experiment_results/experiments/AdaptiveGA \
  --n_runs 10
```

#### （2）不同算法边频率稳定性对比 ⭐⭐⭐

```bash
python analysis/compare_edge_frequency_multi_ga.py \
  --tsp data/ch130.tsp \
  --experiment_results \
    experiment_results/experiments/ClassicGA \
    experiment_results/experiments/ClassicGA_SUS \
    experiment_results/experiments/SemiAdaptiveGA \
    experiment_results/experiments/AdaptiveGA \
  --labels \
    "Classic GA" \
    "Classic GA + SUS" \
    "Semi-Adaptive GA" \
    "Adaptive GA" \
  --n_runs 10
```

📌 该图直观反映：

* 路径结构是否稳定
* 算法是否能识别关键边

---

## 📌 实验结论摘要（示例）

* 自适应遗传算法在 **收敛速度与最终解质量** 上整体优于经典 GA
* SUS 选择策略在一定程度上改善了多样性，但不足以替代自适应机制
* 自适应 GA 在多次运行中表现出 **更高的路径结构稳定性**
* 边频率分析表明，自适应机制有助于稳定保留 TSP 的关键连接关系

---

## 🔧 可扩展方向

* 更大规模 TSP（pcb442 / pr1002）
* 与 ACO / SA 等算法对比
* 自适应算子学习（RL-based operator selection）
* 并行 GA / 多种群 GA

---

## 📜 License

MIT License

