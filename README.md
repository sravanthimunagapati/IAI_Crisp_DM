# IAI_Crisp_DM

Industrial AI mini-projects applying structured, evaluation-driven workflows to real manufacturing, energy, and sensor-data problems, using classical ML and deep learning where each is the right tool for the data.

## Projects

### 1. Bearing RUL Prediction — `B5_RUL_Predictive_Maintenance`
**Problem:** Predict Remaining Useful Life (RUL) and health stage of ball bearings from raw vibration signals, using the FEMTO/PRONOSTIA (IEEE PHM 2012 Challenge) dataset.

**Approach:** Bandpass-filtered raw vibration windows to isolate bearing fault frequencies, engineered a fused RMS + kurtosis health indicator via PCA, and built a hybrid CNN + SVR pipeline for RUL regression alongside a Random Forest classifier for four-stage health assessment. Validated with leave-one-bearing-out (LOBO) cross-validation to test generalization across unseen bearings. Included an outlier check on run-to-failure lifetimes across the fleet, and a self-contained interactive HTML dashboard for maintenance decision support.

**Results:** R² 0.88 (best validation fold), mean R² 0.42 (LOBO), F1-macro 0.36 for health-stage classification (+0.23 over baseline).

**Stack:** Python, PyTorch, scikit-learn, CRISP-DM

---

### 2. Injection Moulding Quality Prediction — `B5_Injection_Moulding_Quality`
**Problem:** Automate post-ejection quality inspection for injection-moulded parts by predicting six defect targets from process sensor data (ProBayes dataset, 564 parts, 9 sensor sources).

**Approach:** Five LightGBM classifiers on scalar process features, plus a Temporal Convolutional Network (TCN) trained directly on raw cavity-pressure waveforms. Evaluated under GroupKFold to prevent cross-condition leakage. SHAP analysis on one defect target ("Streaks") revealed severity levels weren't ordinal but reflected three physically distinct failure mechanisms (cold/slow fill, pressure instability, overheating) — restructuring the target around this improved performance across all three related classes.

**Results:** F1 0.742 → 0.915 after the Streaks restructuring.

**Stack:** Python, LightGBM, TCN, SHAP, CRISP-DM

---

### 3. Industrial PLC Performance Loss Detection — `B5_OEE_Performance_Loss_Detection`
**Problem:** Identify unexplained performance losses (OEE) on a production line from raw PLC controller logs, across drilling, warehouse, and conveyor subsystems.

**Approach:** Merged 11,480+ time-series records from two different PLC controller types (Siemens S7 and Beckhoff) into a unified 5,058 × 197 cycle-level feature dataset. Applied Z-score distance and Isolation Forest as complementary anomaly-scoring methods to flag and characterize performance-loss events, attributing each to timestamp, duration, and likely root cause via per-cycle feature attribution.

**Results:** Five distinct performance-loss events identified with root-cause attribution across subsystems.

**Stack:** Python, Isolation Forest, CRISP-DM

---

### 4. Metal Nut Defect Classification — `DL_Mini_Metal_Nut`
**Problem:** Classify metal nuts as Good or Defective from image data, covering four defect types (bent, colour variation, flipped orientation, surface scratches).

**Approach:** Built and compared a Dense/MLP baseline against a CNN built from scratch (3 convolutional blocks), including data augmentation, class-weight handling for class imbalance, and hyperparameter sweeps over learning rate, batch size, and filter count. Since missing a real defect is the costliest error in quality inspection, tuned the decision threshold (via F1-maximization on validation data) to prioritize defect recall over the default 0.5 cutoff.

**Results:** CNN AUC-ROC 0.94 (vs. MLP 0.745); tuned-threshold defect recall 0.86 vs. 0.57 at default threshold (catching 12/14 vs. 8/14 test-set defects).

**Stack:** Python, TensorFlow/Keras, CNN

---

## Methodology

The RUL, Injection Moulding, and PLC Performance Loss projects follow **CRISP-DM** (business understanding → data understanding → data preparation → modeling → evaluation → deployment). The Metal Nut project follows a standard deep-learning experimentation workflow (preprocessing → augmentation → model development → hyperparameter tuning → threshold optimization). Across all four, the consistent thread is questioning default assumptions before trusting a model — e.g., catching non-ordinal target structure via SHAP, or fleet-level outliers via lifetime distribution checks.