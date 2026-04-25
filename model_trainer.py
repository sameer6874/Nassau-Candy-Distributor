import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def train_models(file_path):
    df = pd.read_csv(file_path)
    
    # Feature Engineering
    # We need to predict Lead Time given: Product, Origin Factory, Destination Region, Ship Mode
    features = ['Product Name', 'Origin Factory', 'Region', 'Ship Mode', 'Division', 'Sales', 'Units']
    target = 'Lead Time'
    
    X = df[features].copy()
    y = df[target]
    
    # Encoding categorical variables
    encoders = {}
    for col in ['Product Name', 'Origin Factory', 'Region', 'Ship Mode', 'Division']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model_name = ""
    best_r2 = -float('inf')
    best_model_obj = None
    
    print("Model Evaluation Results:")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        results[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
        print(f"\n{name}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_obj = model
            
    print(f"\nBest Model: {best_model_name} with R2: {best_r2:.4f}")
    
    # Save best model and encoders
    joblib.dump(best_model_obj, 'best_lead_time_model.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    joblib.dump(features, 'features_list.pkl')
    
    return results

if __name__ == "__main__":
    train_models("cleaned_nassau_candy.csv")
