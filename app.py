import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page Configuration
st.set_page_config(page_title="Nassau Candy - Optimization System", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }
    h1, h2, h3 {
        color: #1e293b;
        font-weight: 700;
    }
    .stSidebar {
        background-color: #0f172a;
    }
    /* Style for Plotly containers */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Data and Resources
@st.cache_data
def load_all_data():
    df = pd.read_csv("cleaned_nassau_candy.csv")
    recs = pd.read_csv("recommendations.csv")
    route_matrix = pd.read_csv("route_performance_matrix.csv", index_col=0)
    return df, recs, route_matrix

# App Header
st.title("🍬 Nassau Candy Distributor")
st.subheader("Factory Reallocation & Shipping Optimization Recommendation System")

try:
    df, recs, route_matrix = load_all_data()
except Exception as e:
    # Handle if index_col was actually needed
    df = pd.read_csv("cleaned_nassau_candy.csv")
    recs = pd.read_csv("recommendations.csv")
    route_matrix = pd.read_csv("route_performance_matrix.csv", index_col=0)

# Sidebar
st.sidebar.header("Control Panel")
module = st.sidebar.selectbox("Choose Module", 
    ["Dashboard Overview", "Factory Optimization Simulator", "What-If Scenario Analysis", "Recommendation Dashboard", "Risk & Impact Panel"])

# --- Dashboard Overview ---
if module == "Dashboard Overview":
    st.header("Operational Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Orders", f"{len(df):,}")
    col2.metric("Avg Lead Time", f"{df['Lead Time'].mean():.1f} Days")
    col3.metric("Total Profit", f"${df['Gross Profit'].sum()/1e6:.1f}M")
    col4.metric("Potential LT Savings", f"{recs['Improvement'].sum():.0f} Days")
    
    st.divider()
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.write("### Lead Time by Region")
        region_lt = df.groupby('Region')['Lead Time'].mean().reset_index()
        fig = px.bar(region_lt, x='Region', y='Lead Time', color='Region',
                     title="Avg Lead Time (Click a bar for region details)",
                     color_discrete_sequence=px.colors.qualitative.Prism,
                     labels={'Lead Time': 'Avg Days'})
        fig.update_layout(showlegend=False, clickmode='event+select')
        
        # Enable selection
        selected_points = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
    
    with col_r:
        st.write("### Profit Distribution")
        profit_data = df.groupby('Region')['Gross Profit'].sum().reset_index()
        fig = px.pie(profit_data, values='Gross Profit', names='Region',
                     title="Profit by Region",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    # Show details based on selection
    if selected_points and 'selection' in selected_points and selected_points['selection']['points']:
        selected_region = selected_points['selection']['points'][0]['x']
        st.write(f"### 🔍 Detailed Orders for: {selected_region}")
        filtered_df = df[df['Region'] == selected_region].head(20)
        st.dataframe(filtered_df[['Order ID', 'Product Name', 'Lead Time', 'Sales', 'Gross Profit']], 
                     use_container_width=True, hide_index=True)
    else:
        st.write("### 🔍 Recent Orders (Interactive - Select a region above)")
        st.info("The charts above provide rich insights on hover. **Click a bar** in the Lead Time chart to see specific orders for that region below.")
        st.dataframe(df[['Order ID', 'Product Name', 'Region', 'Lead Time', 'Sales', 'Gross Profit']].head(10), 
                     use_container_width=True, hide_index=True)

# --- Simulator ---
elif module == "Factory Optimization Simulator":
    st.header("⚡ Factory Optimization Simulator")
    st.write("Analyze how different factory assignments affect product lead times.")
    
    selected_prod = st.selectbox("Select Product", df['Product Name'].unique())
    
    product_data = df[df['Product Name'] == selected_prod]
    current_factory = product_data['Origin Factory'].iloc[0]
    current_region = product_data['Region'].iloc[0]
    
    st.info(f"Current Assignment: **{selected_prod}** produced at **{current_factory}** for **{current_region}**.")
    
    st.write("### Predicted Performance Across Factories")
    sim_data = []
    for factory in route_matrix.index:
        try:
            pred_lt = route_matrix.loc[factory, current_region]
        except:
            pred_lt = df['Lead Time'].mean()
            
        sim_data.append({
            "Factory": factory,
            "Predicted Lead Time (Days)": pred_lt,
            "Status": "Recommended" if factory != current_factory else "Current"
        })
    sim_df = pd.DataFrame(sim_data).sort_values(by="Predicted Lead Time (Days)")
    
    current_lt = sim_df[sim_df['Status'] == "Current"]["Predicted Lead Time (Days)"].values[0]
    best_lt = sim_df["Predicted Lead Time (Days)"].min()
    potential_saving = current_lt - best_lt
    
    col1, col2 = st.columns(2)
    col1.metric("Current Sync Time", f"{current_lt:.1f} Days")
    col2.metric("Best Possible Time", f"{best_lt:.1f} Days", f"-{(potential_saving/current_lt*100):.1f}%")

    # Interactive Bar Chart
    fig = px.bar(sim_df, x='Factory', y='Predicted Lead Time (Days)', color='Status',
                 title=f"Lead Time Comparison for {selected_prod}",
                 color_discrete_map={"Current": "#ef4444", "Recommended": "#3b82f6"},
                 hover_data=['Predicted Lead Time (Days)'])
    fig.add_hline(y=current_lt, line_dash="dash", line_color="red", annotation_text="Current Baseline")
    st.plotly_chart(fig, use_container_width=True)
    
    st.table(sim_df[['Factory', 'Predicted Lead Time (Days)', 'Status']])

# --- What-If Analysis ---
elif module == "What-If Scenario Analysis":
    st.header("🛠️ What-If Scenario Analysis")
    st.write("Compare current assignment performance vs. the recommended optimized model.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Current State")
        st.metric("Avg Lead Time", f"{df['Lead Time'].mean():.1f} Days")
        st.metric("Total Shipping Latency", f"{df['Lead Time'].sum():,} Days")
    
    with col2:
        st.write("### Recommended State")
        opt_avg_lt = (df['Lead Time'].sum() - recs['Improvement'].sum()) / len(df)
        st.metric("Optimized Avg LT", f"{opt_avg_lt:.1f} Days", f"-{(df['Lead Time'].mean() - opt_avg_lt):.1f} Days")
        st.metric("Total Latency Reduction", f"{recs['Improvement'].sum():,.0f} Days")

    st.divider()
    st.write("### Visualize Lead-Time Improvements by Region")
    reg_imp = recs.groupby('Region')['Improvement'].sum().reset_index().sort_values('Improvement', ascending=False)
    
    fig = px.bar(reg_imp, x='Region', y='Improvement', color='Improvement',
                 title="Cumulative Lead Time Savings by Region",
                 color_continuous_scale='Greens',
                 labels={'Improvement': 'Days Saved'})
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("View Improvement Details by Product"):
        st.dataframe(recs[['Product', 'Region', 'Improvement']].sort_values('Improvement', ascending=False), use_container_width=True)

# --- Recommendation Dashboard ---
elif module == "Recommendation Dashboard":
    st.header("📊 Ranked Reassignment Suggestions")
    st.write("Priority list of factory reallocations to maximize efficiency.")
    
    recs['Efficiency Gain (%)'] = (recs['Improvement'] / recs['Current LT'] * 100).round(1)
    
    # Hierarchy Chart: Product vs Improvement
    fig = px.treemap(recs, path=['Region', 'Product'], values='Improvement',
                     color='Improvement', color_continuous_scale='RdYlGn',
                     title="Recommendation Priority Matrix (Size = Improvement)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(recs[['Product', 'Region', 'Current Factory', 'Recommended Factory', 'Current LT', 'Predicted LT', 'Improvement', 'Efficiency Gain (%)']], 
                 use_container_width=True, hide_index=True)
    
    st.download_button("Export Full Recommendation Report", recs.to_csv(index=False), "nassau_optimization_report.csv", "text/csv")

# --- Risk & Impact Panel ---
elif module == "Risk & Impact Panel":
    st.header("⚠️ Risk & Impact Assessment")
    st.write("Analyze potential trade-offs and operational risks associated with reassignments.")
    
    # Interactive Risk Scatter Plot
    recs['Risk Score'] = np.random.uniform(1, 10, size=len(recs)) # Simulated risk
    fig = px.scatter(recs, x='Improvement', y='Risk Score', size='Improvement', color='Region',
                     hover_name='Product', title="Risk vs. Lead Time Reward (Click/Hover Dots)",
                     labels={'Improvement': 'Lead Time Reward (Days)', 'Risk Score': 'Operational Risk'},
                     template="plotly_white")
    fig.add_hline(y=5, line_dash="dot", annotation_text="Moderate Risk Threshold")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Profit Impact Alerts")
        high_risk_factories = recs[recs['Recommended Factory'].isin(['Secret Factory', 'The Other Factory'])].head(5)
        for _, row in high_risk_factories.iterrows():
            st.error(f"**{row['Product']}**: Reallocating to {row['Recommended Factory']} may pressure margins.")
    
    with col2:
        st.subheader("Regional Volatility Warnings")
        volatile_regions = ['International', 'West']
        risk_recs = recs[recs['Region'].isin(volatile_regions)].head(5)
        for _, row in risk_recs.iterrows():
            st.warning(f"**{row['Region']}**: Potential route congestion for '{row['Product']}'.")

# Footer
st.divider()
st.caption("Decision Intelligence System | Developed for Nassau Candy Distributor Sustainability Project")
