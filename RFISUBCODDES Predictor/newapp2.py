import mysql.connector
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- DB CONFIG ---
MYSQL_HOST = "localhost"
MYSQL_USER = "mysqluser"
MYSQL_PASSWORD = "dilan1213"
MYSQL_DB = "predictions"

# Connect to MySQL
def connect_to_mysql():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB
    )

# Load model and encoders
with open("C:\\Users\\Malavi\\Desktop\\RFISUBCODDES Predictor\\new models\\xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("C:\\Users\\Malavi\\Desktop\\RFISUBCODDES Predictor\\new models\\le_pos.pkl", "rb") as f:
    le_pos = pickle.load(f)

with open("C:\\Users\\Malavi\\Desktop\\RFISUBCODDES Predictor\\new models\\le_target.pkl", "rb") as f:
    le_target = pickle.load(f)

avg_count = 5.0
avg_rpc = 10.0

# Streamlit UI
st.set_page_config(page_title="Ancillary Revenue Predictor", layout="wide", page_icon="📊")

# CSS Styling
st.markdown("""
<style>
    html, body, .main {
        background-color: #f4f6f8;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton > button {
        background-color: #0066cc;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #004999;
    }
    
    .prediction-card {
    background-color: #121212;           
    color: #f0f0f0;                       
    border-left: 6px solid #1e90ff;       
    border-radius: 12px;
    padding: 20px 24px;
    margin: 16px 0;
    box-shadow: 0 4px 12px rgba(30, 144, 255, 0.3); 
    transition: transform 0.2s ease;
    }

    .prediction-card:hover {
    transform: translateX(5px);
    box-shadow: 0 6px 16px rgba(30, 144, 255, 0.6);
    }
    progress {
        width: 100%;
        height: 20px;
        appearance: none;
    }
    progress::-webkit-progress-bar {
        background-color: #f3f3f3;
        border-radius: 10px;
    }
    progress::-webkit-progress-value {
        background-color: #0066cc;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.title("Ancillary Revenue Predictor")
    st.markdown("Enter details to get the top predicted ancillary types with visualizations.")
    st.subheader("🛠️ Input Parameters")

    point_of_sale = st.selectbox("Point of Sale", le_pos.classes_)
    year = st.number_input("Year", min_value=2015, max_value=2030, step=1)
    month = st.selectbox("Month", list(range(1, 13)))

    if st.button("Predict"):
        encoded_pos = le_pos.transform([point_of_sale])[0]
        input_df = pd.DataFrame([{
            'Point_of_Sale_encoded': encoded_pos,
            'Year': int(year),
            'Month': int(month),
            'Count': avg_count,
            'Revenue_per_Count': avg_rpc
        }])

        prediction_proba = model.predict_proba(input_df)
        top_3_idx = prediction_proba[0].argsort()[-3:][::-1]
        top_3_classes = model.classes_[top_3_idx]
        top_3_probs = prediction_proba[0][top_3_idx]
        top_3_labels = le_target.inverse_transform(top_3_classes)

        st.session_state['predictions'] = {
            'labels': top_3_labels,
            'probs': top_3_probs,
            'all_probs': prediction_proba[0],
            'all_labels': le_target.inverse_transform(model.classes_)
        }

        # Save to MySQL
        try:
            conn = connect_to_mysql()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions 
                (pos, year, month, top1, top1_prob, top2, top2_prob, top3, top3_prob)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                point_of_sale, year, month,
                top_3_labels[0], float(top_3_probs[0]),
                top_3_labels[1], float(top_3_probs[1]),
                top_3_labels[2], float(top_3_probs[2])
            ))
            conn.commit()
            conn.close()
            st.success("✅ Prediction saved successfully!")
        except Exception as e:
            st.error(f"MySQL Error: {e}")

# Display predictions
if 'predictions' in st.session_state:
    predictions = st.session_state['predictions']
    with col2:
        st.subheader("Prediction Results")

        with st.expander("📌 Top Predictions", expanded=True):
            colors = ['#28a745', '#ffc107', '#dc3545']
            for i, (label, prob) in enumerate(zip(predictions['labels'], predictions['probs'])):
                progress_value = float(prob) * 100
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>#{i+1} <span style="background-color:{colors[i]}; color:white; padding:4px 10px; border-radius:6px;">{label}</span></h3>
                    <p>Confidence: {round(progress_value, 2)}%</p>
                    <progress value="{progress_value}" max="100"></progress>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs([
            "Probability Distribution", 
            "Top Categories", 
            "Monthly Trend", 
            "Recent Predictions"
        ])

        with tab1:
            st.subheader("Probability Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                x=predictions['all_probs'] * 100,
                y=predictions['all_labels'],
                ax=ax,
                palette="viridis"
            )
            ax.set_xlabel("Probability (%)")
            ax.set_ylabel("Ancillary Type")
            ax.set_title("Prediction Probabilities for All Categories")
            st.pyplot(fig)

        with tab2:
            st.subheader("Top Categories Breakdown")
            fig, ax = plt.subplots(figsize=(8, 8))
            explode = [0.1 if x == predictions['labels'][0] else 0 for x in predictions['labels']]
            ax.pie(
                predictions['probs'] * 100,
                labels=predictions['labels'],
                autopct='%1.1f%%',
                startangle=90,
                explode=explode,
                shadow=True,
                colors=sns.color_palette("pastel")
            )
            ax.axis('equal')
            st.pyplot(fig)

        with tab3:
            st.subheader(" Monthly Trend Simulation")
            months = list(range(1, 13))
            simulated_data = {
                'Month': months,
                predictions['labels'][0]: np.random.normal(loc=predictions['probs'][0]*100, scale=5, size=12),
                predictions['labels'][1]: np.random.normal(loc=predictions['probs'][1]*100, scale=5, size=12),
                predictions['labels'][2]: np.random.normal(loc=predictions['probs'][2]*100, scale=5, size=12),
            }
            df_simulated = pd.DataFrame(simulated_data)
            fig, ax = plt.subplots(figsize=(10, 6))
            for label in predictions['labels']:
                ax.plot(df_simulated['Month'], df_simulated[label], marker='o', label=label)
            ax.set_xlabel("Month")
            ax.set_ylabel("Simulated Confidence (%)")
            ax.set_title("Simulated Monthly Trends for Top Categories")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        with tab4:
            st.subheader("Recent Predictions from DB")
            try:
                conn = connect_to_mysql()
                df_recent = pd.read_sql(
                    "SELECT pos, year, month, top1, top1_prob, top2, top2_prob, top3, top3_prob, pred_time FROM predictions ORDER BY pred_time DESC",
                    con=conn
                )
                conn.close()
                st.dataframe(df_recent, use_container_width=True)
            except Exception as e:
                st.error(f"MySQL Fetch Error: {e}")

# Sidebar
with st.sidebar:
    st.markdown("## 📘 About")
    st.info("This tool predicts likely ancillary revenue types based on POS, year, and month using an ML model.")

    st.markdown("## 📊 Model Details")
    st.metric("Target Classes", len(le_target.classes_))
    st.metric("POS Options", len(le_pos.classes_))

    st.markdown("## 🧭 How to Use")
    st.code("1. Select POS\n2. Set Year and Month\n3. Click Predict")
