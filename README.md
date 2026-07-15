# 🧬 The Peak Predictor: LDA Pharmacokinetics Engine

An interactive Supervised Learning simulation designed to teach **Linear Discriminant Analysis (LDA)**, **Pooled Variance Matrices**, and **Gaussian Discriminant Functions** from scratch. You play as a Clinical Pharmacokinetics Data Scientist building a spatial mapping dashboard that monitors a patient's continuous renal marker (Creatinine Clearance Rate) to mathematically categorize and prevent toxic drug concentrations before an infusion pump runs.

## 🎓 Learning Objectives

This project focuses on teaching:
* **Linear Discriminant Analysis (LDA):** Modeling feature distributions as class-specific Gaussians to calculate structural linear decision partitions.
* **Pooled Variance Optimization:** Assuming uniform, shared feature variance ($\sigma^2$) across distinct labels to reduce parameter volatility and prevent overfitting.
* **Discriminant Function Mapping:** Evaluating mathematical scoring logs ($\delta_k(x)$) that weigh prior class probabilities, standard deviations, and mean distances concurrently.
* **Gaussian Spatial Boundaries:** Using class-specific conditional averages to map high-dimensional patient telemetry profiles into objective binary risk levels.

---

## ✨ Features

* **Intensive Care Pharmacokinetics Scenario:** Contextualizes multi-class Gaussian logic networks inside a real-world critical care workspace, dosage calibration pipeline, and toxicity prevention workflow.
* **Transparent Statistical Audits:** Prints computed conditional means, shared system variance scales, and output discriminant weights dynamically at runtime.
* **Log-Prior Component Weighting:** Accounts for training sample distributions automatically using natural logs of prior probabilities inside the calculation loop.
* **Zero External Overhead:** Coded completely using basic data loops and the native Python math library—no heavy math wrappers or external frameworks required.

---

## 🚀 How to Run the Game

### 1. Prerequisites
You only need **Python 3** installed.

### 2. Setup and Execution
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/peak-predictor-lda.git](https://github.com/YOUR_USERNAME/peak-predictor-lda.git)
    cd peak-predictor-lda
    ```
2.  **Save the Code:** Save the provided script as `peak_predictor.py`.
3.  **Run the Script:**
    ```bash
    python peak_predictor.py
    ```

### 3. Gameplay Instructions
1.  **Inspect Metabolic Baseline Records:** Review historical data points tracking patient creatinine clearance alongside binary peak target categories ($0 = \text{Safe/Therapeutic}$, $1 = \text{Toxic Exposure}$).
2.  **Analyze Probability Curves:** Observe the software as it builds class distributions based on mean feature spreads and pooled sample deviations.
3.  **Process a Live Critical Case:** Watch the engine handle an incoming diagnostic profile showing a borderline clearance rate of $63.0\text{ mL/min}$.
4.  **Audit the Discriminant Matrix:** Track the final numerical scores ($\delta_1$ vs. $\delta_0$) to see if the model successfully flags the toxicity risk and scales down the dosage volume.

---

## 🧠 Code Structure Highlights

### Pooled Variance Computation
The algorithm enforces identical distribution shapes across target labels by calculating variance vectors relative to their specific conditional means.

```python
# Pooled Variance: squared_diffs / (total_samples - total_classes)
squared_diffs = 0.0
for c in history:
    if c["toxic"] == 1:
        squared_diffs += (c["clearance"] - mean_toxic) ** 2
    else:
        squared_diffs += (c["clearance"] - mean_safe) ** 2

pooled_variance = squared_diffs / (total_records - 2)

