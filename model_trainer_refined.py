import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def train_refined_models(file_path):
    df = pd.read_csv(file_path)
    
    # Feature Engineering
    # Add more features: State, Division
    categorical_features = ['Product Name', 'Origin Factory', 'Region', 'State/Province', 'Ship Mode', 'Division']
    numeric_features = ['Sales', 'Units']
    
    X = df[categorical_features + numeric_features].copy()
    y = df['Lead Time']
    
    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ('num', StandardScaler(), numeric_features)
        ]
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    best_model = None
    best_r2 = -float('inf')
    
    print("Refined Model Evaluation:")
    for name, model in models.items():
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        
        print(f"\n{name}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2: {r2:.4f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipe
            
    print(f"\nBest Refined Model R2: {best_r2:.4f}")
    
    # Save the pipeline
    joblib.dump(best_model, 'best_lead_time_pipeline.pkl')
    joblib.dump(categorical_features, 'cat_features.pkl')
    joblib.dump(numeric_features, 'num_features.pkl')
    
    return best_model

if __name__ == "__main__":
    train_refined_models("cleaned_nassau_candy.csv")
