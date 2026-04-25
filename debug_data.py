import pandas as pd

def debug_data(file_path):
    df = pd.read_csv(file_path)
    
    # Check variance within a specific route
    route = df[(df['Region'] == 'Pacific') & (df['Origin Factory'] == 'Sugar Shack')]
    print(f"Stats for Pacific - Sugar Shack (Count: {len(route)}):")
    print(route['Lead Time'].describe())
    
    route2 = df[(df['Region'] == 'Atlantic') & (df['Origin Factory'] == 'Sugar Shack')]
    print(f"\nStats for Atlantic - Sugar Shack (Count: {len(route2)}):")
    print(route2['Lead Time'].describe())

    # Check correlations
    print("\nCorrelations with Lead Time:")
    print(df[['Lead Time', 'Sales', 'Units', 'Gross Profit', 'Cost']].corr()['Lead Time'])

if __name__ == "__main__":
    debug_data("cleaned_nassau_candy.csv")
