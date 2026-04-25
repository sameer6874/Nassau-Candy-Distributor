import pandas as pd
import numpy as np
from datetime import datetime

def load_and_preprocess(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Parse dates
    df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d-%m-%Y')
    df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='%d-%m-%Y')
    
    # Calculate Lead Time
    df['Lead Time'] = (df['Ship Date'] - df['Order Date']).dt.days
    
    # Remove outliers (Lead Time should be reasonable, e.g., 0-30 days)
    # Browsing the file earlier showed some dates in 2026 for 2024 orders? 
    # Let's check the distribution.
    
    # Product to Factory Mapping
    factory_map = {
        'Wonka Bar - Nutty Crunch Surprise': "Lot's O' Nuts",
        'Wonka Bar - Fudge Mallows': "Lot's O' Nuts",
        'Wonka Bar -Scrumdiddlyumptious': "Lot's O' Nuts",
        'Wonka Bar - Milk Chocolate': "Wicked Choccy's",
        'Wonka Bar - Triple Dazzle Caramel': "Wicked Choccy's",
        'Laffy Taffy': 'Sugar Shack',
        'SweeTARTS': 'Sugar Shack',
        'Nerds': 'Sugar Shack',
        'Fun Dip': 'Sugar Shack',
        'Fizzy Lifting Drinks': 'Sugar Shack',
        'Everlasting Gobstopper': 'Secret Factory',
        'Hair Toffee': 'The Other Factory',
        'Lickable Wallpaper': 'Secret Factory',
        'Wonka Gum': 'Secret Factory',
        'Kazookles': 'The Other Factory'
    }
    
    df['Origin Factory'] = df['Product Name'].map(factory_map)
    
    # Factory Coordinates
    factory_coords = {
        "Lot's O' Nuts": (32.881893, -111.768036),
        "Wicked Choccy's": (32.076176, -81.088371),
        "Sugar Shack": (48.11914, -96.18115),
        "Secret Factory": (41.446333, -90.565487),
        "The Other Factory": (35.1175, -89.971107)
    }
    
    # For simulation later, we'll need these.
    
    return df

if __name__ == "__main__":
    file_path = "nassau_candy_distributor.csv"
    data = load_and_preprocess(file_path)
    print(data.info())
    print("\nLead Time Description:")
    print(data['Lead Time'].describe())
    print("\nFirst 5 rows with Lead Time and Factory:")
    print(data[['Order Date', 'Ship Date', 'Lead Time', 'Product Name', 'Origin Factory']].head())
    
    # Save cleaned data
    data.to_csv("cleaned_nassau_candy.csv", index=False)
