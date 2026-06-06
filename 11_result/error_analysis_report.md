# Final Report: Validation of Rule-Based vs. Machine Learning Models under Leakage-Free CV

This report presents the final, rigorous comparison between the handcrafted if-else rule (**Rule 2**) and the **Machine Learning models** (SVM, XGBoost, Random Forest). 

We discovered and corrected a critical **data leakage** issue: in the original pipeline, **ComBat batch correction was run globally on the entire dataset before splitting into training and test sets**. This leaked test set distribution information into the training phase, artificially inflating the performance of the handcrafted rule.

Below are the results after running a **proper, leakage-free Leave-One-Out Cross-Validation (LOO-CV)** where ComBat batch correction was fit *strictly* on training samples inside each cross-validation fold.

---

## 📊 Performance Comparison: Leaky vs. Leakage-Free CV

| Model / Rule | Leaky CV Accuracy (Original) | Leakage-Free CV Accuracy (Proper) | Change |
| :--- | :---: | :---: | :---: |
| **SVM (RBF)** | 95.10% | **95.15%** | **+0.05%** (Stable) |
| **XGBoost** | 95.10% | **94.17%** | **-0.93%** (Stable) |
| **Random Forest** | 92.16% | **92.23%** | **+0.07%** (Stable) |
| **Rule 2 (3-Gene Nested)** | **98.04%** | **86.41%** | 🔴 **-11.63%** (Collapsed) |

---

## 🔍 Key Scientific Insights

### 1. Why did the If-Else Rule collapse (from 98.04% to 86.41%)?
* **Fragility of Hardcoded Thresholds**: Rule 2 relies on fixed, hardcoded thresholds (`<= 2.75`, `<= 2.71`, `<= 5.77`). 
* **Effect of Proper Batch Correction**: When ComBat is run properly inside the CV loop, the exact mean and variance adjustments calculated for each fold vary slightly depending on the excluded test sample. 
* **Threshold Misalignment**: Although the biological signal remains, these slight batch correction shifts cause the uncorrected coordinates of the samples to slide. Because Rule 2's thresholds are rigid, it misclassifies samples near the boundary.

### 2. Why did the Machine Learning Models remain stable?
* **Dynamic Threshold Learning**: The ML models (SVM, XGBoost, Random Forest) do not use hardcoded thresholds. In each fold, they receive the ComBat-corrected training set and dynamically calculate the optimal decision boundary.
* **Robustness to Shifts**: If ComBat shifts the feature scales slightly from fold to fold, the ML model automatically adapts its internal weights and thresholds during training, ensuring high generalization accuracy (**95.15% for SVM**).

---

## 💡 Final Recommendation
* **Use SVM (RBF) or XGBoost** as the final classifier. They represent the true biological generalization signal on combat-corrected counts, achieving **94-95% accuracy** without leakage.
* **Discard Rule 2** for production, as its 98% accuracy was a mathematical artifact of global batch correction leakage.
