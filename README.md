### 4. Metal Nut Defect Classification — `DL_Mini_Metal_Nut`
**Problem:** Classify metal nuts as Good or Defective from image data, covering four defect types (bent, colour variation, flipped orientation, surface scratches).

**Approach:** Built and compared a Dense/MLP baseline against a CNN built from scratch (3 convolutional blocks), including data augmentation, class-weight handling for class imbalance, and hyperparameter sweeps over learning rate, batch size, and filter count. Since missing a real defect is the costliest error in quality inspection, tuned the decision threshold (via F1-maximization on validation data) to prioritize defect recall over the default 0.5 cutoff.

**Results:** CNN AUC-ROC 0.94 (vs. MLP 0.745); tuned-threshold defect recall 0.86 vs. 0.57 at default threshold (catching 12/14 vs. 8/14 test-set defects).

**Stack:** Python, TensorFlow/Keras, CNN