# Weather_Prediction

An intelligent, data-driven machine learning system that uses historical climatic profiles to predict daily weather conditions.
This system leverages an optimized `HistGradientBoostingClassifier` along with advanced feature engineering and synthetic over-sampling techniques to resolve severe data imbalances and provide actionable, real-time agricultural management advisories.

---

## 🚀 Key Features
* **Full 5-Class Granularity:** Keeps all categories (`sun`, `rain`, `fog`, `drizzle`, `snow`) isolated without grouping structural targets.
* **Advanced Feature Engineering:** Uses 3-day temporal rolling windows to inject multi-day atmospheric trends into single-day metrics.
* **Imbalance & Overlap Resolution:** Integrates **SMOTE (Synthetic Minority Over-sampling Technique)** to mathematically eliminate majority-class bias for rare events.
* **Interactive CLI Terminal:** A live Command Line Interface allowing agricultural operators to manually test conditions and receive instant automated crop warnings.

---

## 📊 Engineered Feature Space
The model evaluates data points using 9 distinct mathematical dimensions:
1. `precipitation`: Daily liquid moisture fall (mm).
2. `temp_max`: Maximum recorded temperature (°C).
3. `temp_min`: Minimum recorded temperature (°C).
4. `wind`: Average wind velocity metrics.
5. `month`: Extracted integer (1–12) tracking macro-seasonal variance.
6. `temp_range`: Diurnal temperature variation ($T_{max} - T_{min}$).
7. `temp_max_roll3`: Historical 3-day sliding average of maximum temperatures.
8. `precip_roll3`: Historical 3-day sliding accumulation of moisture.
9. `wind_roll3`: Historical 3-day sliding average of wind vectors.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed (Tested successfully up to Python 3.14). 

### 2. Install Required Libraries
Run the following command in your terminal to install the data science and resampling packages:
```bash
pip install numpy pandas scikit-learn imbalanced-learn
```
### 3. Run Code
Execute the main file through your terminal application:
```bash
python weather.py
