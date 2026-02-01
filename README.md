# 🧩 MagicSolver: AI-Powered Rubik's Cube Solver
### *It's not magic. It's Artificial Intelligence.*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Model-Random_Forest-green)
![Status](https://img.shields.io/badge/Status-Academic_Project-orange)

**MagicSolver** is a Machine Learning-based Rubik's Cube solver developed as a project for "Fondamenti di Intelligenza Artificiale" course at University of Salerno (UNISA).

Unlike traditional solvers that rely on complex, pre-computed lookup tables and mathematical group theory, MagicSolver learns to estimate the distance to the solution by observing millions of scrambled states. It effectively derives a heuristic function purely from data, guiding an adaptive search algorithm to solve the puzzle.

---

## 🏗 The Architecture

The system combines a **Random Forest Regressor** (acting as the learned heuristic function $h(s)$) with an **Adaptive Beam Search** algorithm to navigate the state space. 

This project implements **two distinct data representation pipelines** to benchmark "Raw Learning" vs. "Domain Knowledge":

### 🔹 Pipeline 1: The Raw Approach (One-Hot Encoding)
A pure data-driven approach that assumes **zero prior knowledge** of the cube's geometry.
* **Input:** The cube state is flattened into a sparse vector of **324 binary features** (54 faces × 6 colors).
* **Logic:** The model must autonomously abstract spatial rules, piece interdependencies, and color relationships solely from bit patterns.
* **Goal:** To demonstrate that AI can learn complex combinatorial rules without manual feature engineering.

### 🔹 Pipeline 2: The Domain Approach (Manhattan Features)
A hybrid approach that leverages geometric intuition to compress the input space.
* **Input:** The state is reduced to **7 continuous features**, representing the 3D Manhattan Distance of corners and edges from their solved positions.
* **Logic:** The model acts as a "refiner," learning to correct the errors of the mathematical distance to provide a more accurate heuristic than the raw math alone.
* **Goal:** To optimize training efficiency and reduce memory footprint for constrained hardware.

---

## 📊 Data Generation & Methodology

Since no public dataset was suitable for this specific regression task, MagicSolver generates its own training data procedurally:

1.  **Reverse Scrambling:** Starting from a solved cube, the system applies $k$ random moves (backward) to generate pairs of `(State, Distance)`.
2.  **Dataset Size:** 2,000,000 unique samples were generated to ensure statistical significance.
3.  **Format:** Data is serialized in compressed `.npz` format to handle large-scale I/O efficiently.

### Performance Benchmarks
The models were evaluated on a held-out Test Set of 660,000 samples.

| Pipeline | MAE (Mean Absolute Error) | R² Score | Key Insight |
| :--- | :---: | :---: | :--- |
| **Pipeline 1 (OHE)** | **1.615 moves** | **75.14%** | Surprisingly, the raw data model matches the engineered one. |
| **Pipeline 2 (Manhattan)** | 1.624 moves | 74.80% | Highly efficient in memory, similar accuracy. |

---

## 🚀 Installation & Usage

### 1. Prerequisites
Clone the repository and set up a Python environment:

```bash
git clone [https://github.com/YOUR_USERNAME/MagicSolver.git](https://github.com/a-stefanile/MagicSolver.git)
```
### 2. Generate Training Data
Before training, you need to generate the dataset. You can generate it, running the DataSetGenerator.py file

### 3. Train the model
You can choose which pipeline to train. The script will save the trained model as a .joblib file.
You can run TrainModel.py and TrainModelManhattan.py, it depends on which pipeline you want to train

### 4. Run the MagicSolverGUI
At this point you can run MagicSolverGUI.py and you can finally resolve the old cubes that that have been sitting on your shelf for years! :)

---
## 👥 Authors

The autors of the project are:
- Gerardo Russo
- Giuseppe Sepe
- Andrea Stefanile
- Pasquale Tangredi


