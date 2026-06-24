import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# =====================================================================
# 1. DATA INGESTION & ROLLING FEATURE ENGINEERING
# =====================================================================
print("🔄 Training the Agricultural Weather Model...")
df = pd.read_csv(r"C:\Users\Priyansh\Downloads\seattle-weather.csv")

df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['temp_range'] = df['temp_max'] - df['temp_min'] 

# 3-day sliding historical context
df['temp_max_roll3'] = df['temp_max'].rolling(window=3, min_periods=1).mean()
df['precip_roll3'] = df['precipitation'].rolling(window=3, min_periods=1).mean()
df['wind_roll3'] = df['wind'].rolling(window=3, min_periods=1).mean()

features = ['precipitation', 'temp_max', 'temp_min', 'wind', 'month', 'temp_range', 
            'temp_max_roll3', 'precip_roll3', 'wind_roll3']
X = df[features]
y = df['weather'] 

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into training and testing sets first (Never apply SMOTE to test data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# =====================================================================
# 2. SYNTHETIC MINORITY OVER-SAMPLING (SMOTE)
# =====================================================================
# This generates synthetic mathematical rows for drizzle and snow to balance the classes
smote = SMOTE(k_neighbors=3, random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# =====================================================================
# 3. MODEL TRAINING
# =====================================================================
model = HistGradientBoostingClassifier(
    max_iter=200,              
    max_depth=6,               
    learning_rate=0.05,        
    min_samples_leaf=5,        
    random_state=42
)
model.fit(X_train_balanced, y_train_balanced)
print("✅ Balanced Model successfully trained!\n")

# =====================================================================
# 4. PERFORMANCE EVALUATION REPORT
# =====================================================================
y_pred = model.predict(X_test)
print("="*60)
print(f"🏆 EVALUATION REPORT: OVERALL TESTING ACCURACY: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("="*60)
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("="*60 + "\n")

# =====================================================================
# 5. INTERACTIVE USER INPUT LOOP
# =====================================================================
last_row = df.iloc[-1] 

while True:
    print("🌍 ENTER DAILY WEATHER CONDITIONS FOR REAL-TIME PREDICTION")
    print("-" * 60)
    
    user_choice = input("Press Enter to test a day (or type 'exit'): ").strip().lower()
    if user_choice == 'exit':
        break
        
    try:
        precipitation = float(input("🔹 Enter Precipitation (in mm): "))
        temp_max = float(input("🔹 Enter Max Temperature (°C): "))
        temp_min = float(input("🔹 Enter Min Temperature (°C): "))
        wind = float(input("🔹 Enter Wind Speed: "))
        month = int(input("🔹 Enter Month Number (1-12): "))
        
        temp_range = temp_max - temp_min
        
        input_data = pd.DataFrame([{
            'precipitation': precipitation,
            'temp_max': temp_max,
            'temp_min': temp_min,
            'wind': wind,
            'month': month,
            'temp_range': temp_range,
            'temp_max_roll3': (temp_max + last_row['temp_max']) / 2,
            'precip_roll3': (precipitation + last_row['precipitation']) / 2,
            'wind_roll3': (wind + last_row['wind']) / 2
        }])
        
        pred_id = model.predict(input_data)[0]
        final_weather = le.inverse_transform([pred_id])[0]
        
        print("\n" + "*" * 40)
        print(f"🔮 PREDICTED WEATHER: {final_weather.upper()}")
        print("*" * 40 + "\n")
        
    except ValueError:
        print("\n❌ Input Error: Enter numbers only.\n")
