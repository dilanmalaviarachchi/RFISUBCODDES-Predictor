import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import numpy as np
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import plotly.express as px
warnings.filterwarnings('ignore')

# Setup
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({'figure.autolayout': True})

st.set_page_config(page_title="Ancillary Revenue Predictor", layout="wide", page_icon="✈️")

# Load model and encoders
@st.cache_resource
def load_model_and_encoders():
    try:
        with open('models/oxgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/label_encoder_pos.pkl', 'rb') as f:
            le_pos = pickle.load(f)
        with open('models/label_encoder_rfisubcoddes.pkl', 'rb') as f:
            le_rfisubcoddes = pickle.load(f)
        return model, le_pos, le_rfisubcoddes
    except Exception as e:
        st.error(f"Error loading model/encoders: {e}")
        st.stop()

# Load dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('C:\\Users\\Malavi\\Desktop\\RFISUBCODDES Predictor\\Ancillary Rev - 2015 - 2023.xlsx', header=2)
        df['Flight Date'] = pd.to_datetime(df['Flight Date'])
        df['Year'] = df['Flight Date'].dt.year
        df['Month'] = df['Flight Date'].dt.month
        df['Quarter'] = df['Flight Date'].dt.quarter
        df['Day of Week'] = df['Flight Date'].dt.dayofweek
        df['Weekday Name'] = df['Flight Date'].dt.day_name()
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

model, le_pos, le_rfisubcoddes = load_model_and_encoders()
df = load_data()

point_of_sale_options = le_pos.classes_

st.title("✈️ Ancillary Revenue Predictor & Analytics")
st.caption("Built with Streamlit + XGBoost + Prophet | Version 3.0 | Updated: 2024")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction", "📈 Trends", "🏆 Top Categories", "🔍 Future Impact"])

# ---------- PREDICTION ----------
with tab1:
    st.header("Make a Prediction")
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            year = st.number_input("Select Year", 2015, 2030, 2024)
            month = st.selectbox("Select Month", list(range(1, 13)), index=0, format_func=lambda x: pd.to_datetime(f"2024-{x}-01").strftime('%B'))
        with col2:
            pos = st.selectbox("Point of Sale", sorted(point_of_sale_options))
            count = st.slider("Expected Transaction Count", 1, 500, 50)
        with col3:
            confidence_level = st.slider("Confidence Level (%)", 50, 95, 80)
            seasonality_mode = st.selectbox("Seasonality Mode", ["additive", "multiplicative"])

        submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        with st.spinner("Running prediction..."):
            pos_encoded = le_pos.transform([pos])[0]
            prediction = model.predict([[year, month, pos_encoded, count]])
            label = le_rfisubcoddes.inverse_transform(prediction)[0]
            
            # Confidence interval calculation
            pred_proba = model.predict_proba([[year, month, pos_encoded, count]])
            top3_idx = pred_proba.argsort()[0][-3:][::-1]
            top3_labels = le_rfisubcoddes.inverse_transform(top3_idx)
            top3_probs = pred_proba[0][top3_idx]
            
            st.success(f"🎯 Predicted RFISUBCODDES: **{label}** (Confidence: {top3_probs[0]*100:.1f}%)")
            
            # Show top 3 predictions with confidence
            st.subheader("Alternative Predictions")
            cols = st.columns(3)
            for i, (lbl, prob) in enumerate(zip(top3_labels, top3_probs)):
                with cols[i]:
                    st.metric(label=f"#{i+1} {lbl}", value=f"{prob*100:.1f}%")
            
            # Business insights
            st.subheader("Business Insights")
            if "BAGGAGE" in label or "WEIGHT" in label:
                st.info("💼 **Baggage Services**: Consider dynamic pricing strategies for peak travel seasons.")
            elif "SEAT" in label:
                st.info("💺 **Seat Upgrades**: Bundle with premium meals for higher conversion rates.")
            elif "UPGRADE" in label:
                st.info("✨ **Cabin Upgrades**: Target frequent flyers with personalized offers.")
            
            # Store prediction for future impact analysis
            if 'predictions' not in st.session_state:
                st.session_state.predictions = []
            st.session_state.predictions.append({
                'Year': year,
                'Month': month,
                'POS': pos,
                'Count': count,
                'Prediction': label
            })

# ---------- HISTORICAL TRENDS ----------
with tab2:
    st.header("Advanced Historical Analysis")
    selected_pos = st.selectbox("Choose Point of Sale", sorted(point_of_sale_options), key="trends_pos")
    df_filtered = df[df['Point of Sale'] == selected_pos]

    if df_filtered.empty:
        st.warning("No data available for this POS.")
    else:
        # Time series decomposition
        st.subheader("Time Series Decomposition")
        ts_data = df_filtered.groupby('Flight Date').size().reset_index(name='Count')
        ts_data = ts_data.set_index('Flight Date').asfreq('D').fillna(0)
        decomposition = seasonal_decompose(ts_data['Count'], model='additive', period=365)
        
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=decomposition.observed.index, y=decomposition.observed, name='Observed'), row=1, col=1)
        fig.add_trace(go.Scatter(x=decomposition.trend.index, y=decomposition.trend, name='Trend'), row=2, col=1)
        fig.add_trace(go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal, name='Seasonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=decomposition.resid.index, y=decomposition.resid, name='Residual'), row=4, col=1)
        fig.update_layout(height=800, title_text="Time Series Decomposition")
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap by day of week and month
        st.subheader("Transaction Heatmap")
        heatmap_data = df_filtered.groupby(['Month', 'Day of Week']).size().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="g", ax=ax)  # Changed fmt to "g" which handles both int and float
        ax.set_title(f"Transactions by Day of Week and Month for {selected_pos}")
        ax.set_yticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        st.pyplot(fig)

# ---------- TOP CATEGORIES ----------
with tab3:
    st.header("Top Categories Insights")
    
    # Add a time filter
    year_range = st.slider("Select Year Range", 
                          min_value=df['Year'].min(), 
                          max_value=df['Year'].max(), 
                          value=(df['Year'].min(), df['Year'].max()))
    
    df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top RFISUBCODDES by Frequency")
        top5_count = df_filtered['RFISUBCODDES'].value_counts().nlargest(5)
        fig = go.Figure(go.Bar(
            x=top5_count.values,
            y=top5_count.index,
            orientation='h',
            marker_color='skyblue',
            text=top5_count.values,
            textposition='auto'
        ))
        fig.update_layout(
            title="Most Frequent Categories",
            xaxis_title="Count",
            yaxis_title="Category",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Top RFISUBCODDES by Revenue")
        top5_rev = df_filtered.groupby('RFISUBCODDES')['Sum of Revenue USD'].sum().nlargest(5)
        fig = go.Figure(go.Pie(
            labels=top5_rev.index,
            values=top5_rev.values,
            hole=.4,
            textinfo='label+percent',
            insidetextorientation='radial',
            marker_colors=px.colors.sequential.Plasma_r
        ))
        fig.update_layout(
            title="Revenue Share by Category",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Trend of top categories over time
    st.subheader("Category Trends Over Time")
    top_cats = df_filtered['RFISUBCODDES'].value_counts().nlargest(3).index
    trend_data = df_filtered[df_filtered['RFISUBCODDES'].isin(top_cats)]
    trend_data = trend_data.groupby(['Year', 'RFISUBCODDES']).size().reset_index(name='Count')
    
    fig = px.line(trend_data, x='Year', y='Count', color='RFISUBCODDES',
                 title="Top 3 Categories Trend Over Years",
                 markers=True)
    st.plotly_chart(fig, use_container_width=True)

# ---------- FUTURE IMPACT ----------
with tab4:
    st.header("Future Impact Analysis")
    
    if 'predictions' not in st.session_state or not st.session_state.predictions:
        st.warning("Make some predictions first to see their future impact.")
    else:
        predictions_df = pd.DataFrame(st.session_state.predictions)
        
        # Convert to time series
        predictions_df['Date'] = pd.to_datetime(predictions_df['Year'].astype(str) + '-' + 
                                      predictions_df['Month'].astype(str) + '-01')
        
        # Show predictions table
        st.subheader("Your Predictions")
        st.dataframe(predictions_df[['Date', 'POS', 'Prediction', 'Count']], 
                    use_container_width=True)
        
        # Forecast impact
        st.subheader("Projected Impact on Future Revenue")
        
        # Prepare data for Prophet
        df_prophet = df.groupby(['Flight Date', 'RFISUBCODDES']).size().reset_index(name='Count')
        df_prophet = df_prophet.rename(columns={'Flight Date': 'ds', 'Count': 'y'})
        
        # Filter for predicted categories
        predicted_cats = predictions_df['Prediction'].unique()
        df_prophet = df_prophet[df_prophet['RFISUBCODDES'].isin(predicted_cats)]
        
        # Create and fit model
        m = Prophet(seasonality_mode='multiplicative')
        m.fit(df_prophet)
        
        # Create future dataframe
        future = m.make_future_dataframe(periods=12, freq='M')
        
        # Forecast
        forecast = m.predict(future)
        
        # Plot forecast
        fig1 = m.plot(forecast)
        st.pyplot(fig1)
        
        # Plot components
        fig2 = m.plot_components(forecast)
        st.pyplot(fig2)
        
        # Monte Carlo simulation for uncertainty
        st.subheader("Revenue Simulation Based on Predictions")
        
        # Get historical revenue data for predicted categories
        rev_data = df[df['RFISUBCODDES'].isin(predicted_cats)]
        avg_rev = rev_data.groupby('RFISUBCODDES')['Sum of Revenue USD'].mean().to_dict()
        
        # Simulate future revenue
        np.random.seed(42)
        simulations = []
        for _, row in predictions_df.iterrows():
            base_rev = avg_rev.get(row['Prediction'], 100)  # Default to $100 if no historical data
            # Simulate with some randomness
            sim_rev = np.random.normal(loc=base_rev * row['Count'], scale=base_rev * 0.2, size=1000)
            simulations.append({
                'Date': row['Date'],
                'Category': row['Prediction'],
                'Mean Revenue': sim_rev.mean(),
                'P5': np.percentile(sim_rev, 5),
                'P95': np.percentile(sim_rev, 95)
            })
        
        sim_df = pd.DataFrame(simulations)
        
        # Plot simulation results
        fig = go.Figure()
        for cat in sim_df['Category'].unique():
            cat_data = sim_df[sim_df['Category'] == cat]
            fig.add_trace(go.Scatter(
                x=cat_data['Date'],
                y=cat_data['Mean Revenue'],
                name=cat,
                mode='lines+markers'
            ))
            fig.add_trace(go.Scatter(
                x=cat_data['Date'],
                y=cat_data['P5'],
                fill=None,
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=cat_data['Date'],
                y=cat_data['P95'],
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                name=f'{cat} Range'
            ))
        
        fig.update_layout(
            title="Projected Revenue with Confidence Intervals",
            xaxis_title="Date",
            yaxis_title="Revenue (USD)",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

# Sidebar with additional info
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Advanced Ancillary Revenue Predictor**
    
    This tool helps airlines:
    - Predict profitable ancillary services
    - Analyze historical trends
    - Forecast future revenue impact
    
    Version 3.0 features:
    - Time series decomposition
    - Monte Carlo simulations
    - Interactive visualizations
    """)
    
    st.divider()
    st.markdown("**Data Last Updated:**")
    st.info(f"{df['Flight Date'].max().strftime('%Y-%m-%d')}")
    
    if 'predictions' in st.session_state:
        st.metric("Total Predictions Made", len(st.session_state.predictions))

