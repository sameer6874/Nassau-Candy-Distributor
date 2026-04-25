import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import joblib

def build_engine(file_path):
    df = pd.read_csv(file_path)
    
    # 1. Route Performance Matrix (Average Lead Time per Factory-Region pair)
    route_matrix = df.groupby(['Origin Factory', 'Region'])['Lead Time'].mean().unstack()
    print("Route Performance Matrix (Avg Lead Time):")
    print(route_matrix)
    
    # Fill missing values with global mean for that factory or region
    route_matrix = route_matrix.fillna(df['Lead Time'].mean())
    route_matrix.to_csv("route_performance_matrix.csv")
    
    # 2. Route Clustering
    # We'll cluster regions based on their lead time performance across factories
    pivot_for_clustering = df.pivot_table(index='Region', columns='Origin Factory', values='Lead Time', aggfunc='mean').fillna(df['Lead Time'].mean())
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    pivot_for_clustering['Cluster'] = kmeans.fit_predict(pivot_for_clustering)
    
    print("\nRegion Clusters:")
    print(pivot_for_clustering['Cluster'])
    pivot_for_clustering.to_csv("region_clusters.csv")
    
    # 3. Simulation Function
    def simulate_reallocation(product_name, current_factory, target_region):
        # In a real scenario, this would use the ML model if it were accurate.
        # Here we'll use the Route Matrix as a reliable historical benchmark.
        results = []
        factories = df['Origin Factory'].unique()
        
        for factory in factories:
            predicted_lt = route_matrix.loc[factory, target_region]
            # Estimate profit impact (simulated for now, keeping it simple)
            # Assume profit stays same or varies slightly based on factory efficiency
            results.append({
                'Factory': factory,
                'Predicted Lead Time': predicted_lt,
                'Is Current': (factory == current_factory)
            })
            
        return pd.DataFrame(results).sort_values(by='Predicted Lead Time')

    # 4. Recommendation Logic
    recommendations = []
    # Identify products with high current lead times
    product_stats = df.groupby(['Product Name', 'Origin Factory', 'Region'])['Lead Time'].mean().reset_index()
    product_stats = product_stats[product_stats['Lead Time'] > df['Lead Time'].mean()]
    
    for _, row in product_stats.head(20).iterrows():
        sim = simulate_reallocation(row['Product Name'], row['Origin Factory'], row['Region'])
        best_opt = sim.iloc[0]
        if not best_opt['Is Current']:
            improvement = row['Lead Time'] - best_opt['Predicted Lead Time']
            if improvement > 50: # Only recommend if improvement > 50 days
                recommendations.append({
                    'Product': row['Product Name'],
                    'Region': row['Region'],
                    'Current Factory': row['Origin Factory'],
                    'Recommended Factory': best_opt['Factory'],
                    'Current LT': row['Lead Time'],
                    'Predicted LT': best_opt['Predicted Lead Time'],
                    'Improvement': improvement
                })
                
    rec_df = pd.DataFrame(recommendations).sort_values(by='Improvement', ascending=False)
    print("\nTop Recommendations:")
    print(rec_df.head(10))
    rec_df.to_csv("recommendations.csv", index=False)
    
    return rec_df

if __name__ == "__main__":
    build_engine("cleaned_nassau_candy.csv")
