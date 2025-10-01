# Anomaly Detection on Energy Plant Sensor Data

**Project Goal:**  
Detect anomalies from sensor readings collected at an energy manufacturing plant. These anomalies may indicate unusual operating conditions or faulty sensor behavior. Identifying them early can help in preventive maintenance, improved safety, and process efficiency.

**Evaluation Criteria:**  
Submissions are evaluated using **F1-score per class** (class 0 and class 1 separately) and **accuracy per class**. Because anomalies are extremely rare, focusing on F1-score is essential to avoid trivial solutions that always predict the majority class.



## Problem Statement

Industrial plants often use a large number of sensors to continuously monitor machine health and energy flow. While most sensor readings are normal, a small fraction can represent **rare anomalies**. These anomalies can be caused by operational faults, sensor malfunctions, or unexpected external events.  

The challenge is to build a model that can **detect anomalies (class 1)** from large-scale time-stamped sensor data.  



## Dataset Summary

- **Train shape:** 1,639,424 rows × 7 columns  
- **Test shape:** 409,856 rows × 7 columns  

**Features:**
- `Date` → timestamp  
- `X1–X5` → continuous sensor readings  
- `target` → anomaly indicator (0 = normal, 1 = anomaly)

**Class Imbalance:**
- Class 0 (normal): ~99.14%  
- Class 1 (anomaly): ~0.86%  

This imbalance means that a naïve model predicting all zeros would achieve ~99% accuracy but **F1-score for anomalies would be zero**. Handling this imbalance was one of the core challenges.  



## Exploratory Data Analysis (EDA)

Key findings from EDA:  

1. **Severe class imbalance**  
   - Majority of the data belongs to class 0.  
   - Visualization confirmed extremely rare anomalies.
     ![Class Distribution]() 

2. **Sensor behavior and outliers**  
   - `X3` and `X4` showed extremely large spikes (up to 80–90), far outside the main distribution.  
   - Applying `log1p` transformation significantly reduced skewness.  
   ![Box plot]()
 


## Preprocessing Steps

1. **Datetime Features**
   - Extracted `year`, `month`, `day`, `day_of_week`, `day_of_year`.  
   - Dropped the raw `Date` column for modeling.

2. **Outlier Handling**
   - Applied `np.log1p()` to `X3` and `X4` to compress large outlier values.

3. **Scaling**
   - Used `RobustScaler` (less sensitive to outliers than StandardScaler).  
   - Ensured that linear models (e.g., Logistic, SVM) worked better.

4. **Feature Engineering**
   - Added `day_of_week` (cyclical pattern).
   - Added `X3_roll_mean_3` and `X3_roll_std_3` (rolling statistics with window=3).

5. **Cross-Validation**
   - Used **StratifiedKFold (n=5)**.  
   - Stratification preserved anomaly ratio in every fold, preventing biased validation splits.  
   - Out-of-Fold (OoF) predictions were stored for each model to evaluate performance consistently.

6. **Threshold Tuning**
   - Since anomalies are rare, a fixed 0.5 threshold gave poor F1.  
   - A custom function searched thresholds between 0.01–0.99 to find the one that maximized F1-score on OoF predictions.



## Modeling Approach

To address imbalance and capture different perspectives, I trained several models:  

- **Logistic Regression** → Simple baseline with `class_weight="balanced"`.  
- **XGBoost** → Gradient boosting, tuned with `scale_pos_weight`.  
- **CatBoost** → Robust boosting, used `auto_class_weights="Balanced"`.    

### Ensembling
- **Blending:** Averaged predictions of XGBoost + CatBoost.  
- **Weighted Average:** Tuned weights; best was 50% XGB, 50% CatBoost.  
- **Stacking:** Logistic regression as meta-model (weaker here, overfitting minority class).  



## Results

All metrics are reported on **out-of-fold (OoF)** validation sets.  
The table shows the best F1 and threshold per model.

| Model              | Best Threshold | F1-score | ROC AUC | Notes |
|--------------------|----------------|----------|---------|-------|
| Logistic Regression | 0.970          | 0.309    | 0.87    | Baseline, underfit |
| XGBoost            | 0.970          | 0.742    | 0.99    | Strong performance |
| CatBoost           | 0.970          | 0.734    | 0.993   | Very competitive |
| Blend (XGB+CAT)    | 0.960          | 0.745    | 0.993   | Slight improvement |
| Weighted Average   | 0.960          | 0.745    | 0.993   | Best final choice |
| Stacking           | 0.990          | 0.680    | 0.985   | Overfitting issues |

**Final Submission**: Weighted Average (XGB + CatBoost, 50–50).  

Predicted anomaly rate on test set: **~0.84%** → consistent with training distribution.



## Conclusion

- Boosting models (XGB & CatBoost) captured anomalies best.  
- Ensemble (weighted average) slightly improved F1-score compared to individual models.  
- Logistic regression provided a simple baseline but was too weak for imbalanced, non-linear data.  

**Final F1-score (OoF): ~0.745**  



## Next Steps

- Try **LightGBM** with tuned class weights.    
- Add more **rolling/statistical features** to capture temporal shifts.  
- Regularize stacking ensemble to prevent minority class overfitting.
