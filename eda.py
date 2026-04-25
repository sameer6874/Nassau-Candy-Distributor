import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(file_path):
    df = pd.read_csv(file_path)
    
    # 1. Lead Time by Region
    plt.figure(figsize=(12, 6))
    region_lt = df.groupby('Region')['Lead Time'].mean().sort_values()
    sns.barplot(x=region_lt.index, y=region_lt.values, palette='viridis')
    plt.title('Average Lead Time by Region')
    plt.ylabel('Days')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('lt_by_region.png')
    
    # 2. Lead Time by Ship Mode
    plt.figure(figsize=(10, 5))
    mode_lt = df.groupby('Ship Mode')['Lead Time'].mean().sort_values()
    sns.barplot(x=mode_lt.index, y=mode_lt.values, palette='magma')
    plt.title('Average Lead Time by Ship Mode')
    plt.ylabel('Days')
    plt.tight_layout()
    plt.savefig('lt_by_mode.png')
    
    # 3. Profit vs Lead Time
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Lead Time', y='Gross Profit', alpha=0.3)
    plt.title('Gross Profit vs Lead Time')
    plt.tight_layout()
    plt.savefig('profit_vs_lt.png')
    
    # 4. Product Performance
    plt.figure(figsize=(15, 8))
    prod_lt = df.groupby('Product Name')['Lead Time'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=prod_lt.values, y=prod_lt.index, palette='coolwarm')
    plt.title('Top 10 Products with Highest Lead Time')
    plt.xlabel('Average Days')
    plt.tight_layout()
    plt.savefig('top_slow_products.png')
    
    # Textual Summaries
    print("EDA Insights Summary:")
    print(f"Total Orders: {len(df)}")
    print(f"Overall Avg Lead Time: {df['Lead Time'].mean():.2f} days")
    print(f"Most Profitable Region: {df.groupby('Region')['Gross Profit'].sum().idxmax()}")
    print(f"Highest Lead Time Region: {df.groupby('Region')['Lead Time'].mean().idxmax()}")
    
    # Identify bottlenecks
    bottlenecks = df.groupby(['Region', 'Origin Factory'])['Lead Time'].mean().reset_index()
    bottlenecks = bottlenecks.sort_values(by='Lead Time', ascending=False)
    print("\nSlowest Routes (Region - Factory):")
    print(bottlenecks.head(5))

if __name__ == "__main__":
    perform_eda("cleaned_nassau_candy.csv")
