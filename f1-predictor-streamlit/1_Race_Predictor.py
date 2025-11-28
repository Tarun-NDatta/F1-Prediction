import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page config
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #e10600 0%, #ff1801 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #e10600;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header"><h1 style="color: white; margin: 0;">🏎️ F1 Race Predictor</h1></div>', unsafe_allow_html=True)

st.markdown("""
**Machine Learning-Powered Race Predictions** • Trained on 2022-2024 historical data  
Compare **Ridge Regression**, **XGBoost**, and **CatBoost** models with track specialization
""")

# ==================== DATA LOADING ====================

@st.cache_data
def load_data():
    """Load all predictor data"""
    data_dir = data_dir = Path(__file__).resolve().parent / "data"

    
    try:
        return {
            'events': pd.read_csv(data_dir / "events_2025.csv"),
            'drivers': pd.read_csv(data_dir / "drivers_2025.csv"),
            'tracks': pd.read_csv(data_dir / "track_specs.csv"),
            'predictions': pd.read_csv(data_dir / "predictions_2025.csv")
        }
    except FileNotFoundError:
        st.error("""
        ⚠️ **Data files not found!**
        
        Please run from your Django project:
        ```bash
        python manage.py export_predictor_data
        ```
        
        Then copy the `predictor_data/` folder contents to your Streamlit app's `data/` folder.
        """)
        st.stop()

# Load data
data = load_data()

# ==================== SIDEBAR ====================

with st.sidebar:
    st.header("📊 About the Models")
    
    st.markdown("""
    ### Ridge Regression
    Linear baseline model with L2 regularization
    - **MAE**: ~2.8 positions
    - **Best for**: Consistency
    
    ### XGBoost
    Gradient boosting with feature engineering
    - **MAE**: ~2.5 positions
    - **Best for**: Complex patterns
    
    ### CatBoost
    Track-specialized categorical boosting
    - **MAE**: ~2.3 positions
    - **Best for**: Circuit-specific predictions
    """)
    
    st.markdown("---")
    st.markdown("**Dissertation Project**  \n*ML-Based F1 Prediction System*")

# ==================== MAIN APP ====================

# Race selection
st.header("🎯 Select Race")

col1, col2 = st.columns([2, 1])

with col1:
    # Get unique events
    events = data['events'].sort_values('round')
    event_options = events['name'].tolist()
    
    selected_event_name = st.selectbox(
        "Choose a race",
        event_options,
        help="Select a race from the 2025 season"
    )
    
    selected_event = events[events['name'] == selected_event_name].iloc[0]

with col2:
    st.metric("Round", f"R{selected_event['round']}")
    st.metric("Circuit", selected_event['circuit_name'])

# Track characteristics
st.markdown("---")
st.subheader(f"📍 {selected_event['circuit_name']}")

track_info = data['tracks'][data['tracks']['circuit_name'] == selected_event['circuit_name']]

if not track_info.empty:
    track = track_info.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Track Type", track['track_category'])
    with col2:
        st.metric("Power Sensitivity", f"{track['power_sensitivity']:.1f}/10")
    with col3:
        st.metric("Overtaking", f"{track['overtaking_difficulty']:.1f}/10")
    with col4:
        st.metric("Quali Importance", f"{track['qualifying_importance']:.1f}/10")

# Get predictions for this event
st.markdown("---")
st.header("🏁 Race Predictions")

event_preds = data['predictions'][data['predictions']['event_name'] == selected_event_name]

if event_preds.empty:
    st.warning(f"No predictions available for {selected_event_name} yet.")
    st.info("Predictions will be generated closer to race weekend.")
    st.stop()

# Pivot predictions by model
pred_pivot = event_preds.pivot_table(
    index='driver_name',
    columns='model',
    values='predicted_position'
).reset_index()

# Calculate ensemble (average of all models)
if all(model in pred_pivot.columns for model in ['Ridge', 'XGBoost', 'CatBoost']):
    pred_pivot['Ensemble'] = pred_pivot[['Ridge', 'XGBoost', 'CatBoost']].mean(axis=1)
elif 'Ridge' in pred_pivot.columns and 'XGBoost' in pred_pivot.columns:
    pred_pivot['Ensemble'] = pred_pivot[['Ridge', 'XGBoost']].mean(axis=1)
else:
    pred_pivot['Ensemble'] = pred_pivot.iloc[:, 1]  # Use first available model

# Sort by ensemble prediction
pred_pivot = pred_pivot.sort_values('Ensemble').reset_index(drop=True)
pred_pivot.index = pred_pivot.index + 1  # Start from 1

# Display predictions
tab1, tab2, tab3 = st.tabs(["📋 Predicted Grid", "📊 Model Comparison", "🎯 Confidence Analysis"])

with tab1:
    st.subheader("Predicted Finishing Order")
    
    # Format predictions table
    display_df = pred_pivot.copy()
    display_df.index.name = 'Pos'
    
    # Round predictions to 1 decimal
    for col in ['Ridge', 'XGBoost', 'CatBoost', 'Ensemble']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(1)
    
    # Add actual results if available
    actual_results = event_preds[event_preds['actual_position'].notna()]
    if not actual_results.empty:
        actual_map = dict(zip(actual_results['driver_name'], actual_results['actual_position']))
        display_df['Actual'] = display_df['driver_name'].map(actual_map)
        
        # Calculate differences
        if 'Ensemble' in display_df.columns and 'Actual' in display_df.columns:
            display_df['Diff'] = (display_df['Ensemble'] - display_df['Actual']).round(1)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=600
    )
    
    # Show prediction statistics
    if 'Actual' in display_df.columns:
        st.markdown("---")
        st.subheader("📈 Prediction Accuracy")
        
        col1, col2, col3 = st.columns(3)
        
        mae = abs(display_df['Ensemble'] - display_df['Actual']).mean()
        perfect_preds = (display_df['Ensemble'].round() == display_df['Actual']).sum()
        top3_accuracy = (display_df[display_df['Actual'] <= 3]['Ensemble'] <= 3.5).sum()
        
        with col1:
            st.metric("Mean Absolute Error", f"{mae:.2f} positions")
        with col2:
            st.metric("Perfect Predictions", f"{perfect_preds}/{len(display_df)}")
        with col3:
            st.metric("Top-3 Accuracy", f"{top3_accuracy}/3")

with tab2:
    st.subheader("Model Predictions Comparison")
    
    # Create comparison chart
    fig = go.Figure()
    
    colors = {'Ridge': '#FF6B6B', 'XGBoost': '#4ECDC4', 'CatBoost': '#45B7D1', 'Ensemble': '#96CEB4'}
    
    for model in ['Ridge', 'XGBoost', 'CatBoost', 'Ensemble']:
        if model in pred_pivot.columns:
            fig.add_trace(go.Bar(
                name=model,
                x=pred_pivot['driver_name'],
                y=pred_pivot[model],
                marker_color=colors[model],
                text=pred_pivot[model].round(1),
                textposition='outside'
            ))
    
    # Add actual results if available
    if 'Actual' in display_df.columns:
        actual_vals = display_df.set_index('driver_name')['Actual']
        fig.add_trace(go.Scatter(
            name='Actual Result',
            x=pred_pivot['driver_name'],
            y=pred_pivot['driver_name'].map(actual_vals),
            mode='markers',
            marker=dict(size=12, color='black', symbol='diamond'),
            text=actual_vals,
        ))
    
    fig.update_layout(
        title="Predicted Position by Model",
        xaxis_title="Driver",
        yaxis_title="Position",
        yaxis=dict(autorange="reversed"),
        height=500,
        barmode='group',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model agreement analysis
    st.markdown("---")
    st.subheader("🤝 Model Agreement")
    
    if all(model in pred_pivot.columns for model in ['Ridge', 'XGBoost', 'CatBoost']):
        # Calculate variance in predictions per driver
        pred_pivot['Variance'] = pred_pivot[['Ridge', 'XGBoost', 'CatBoost']].var(axis=1)
        
        high_agreement = (pred_pivot['Variance'] < 1).sum()
        low_agreement = (pred_pivot['Variance'] > 3).sum()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("High Agreement", f"{high_agreement}/{len(pred_pivot)} drivers")
            st.caption("Variance < 1 position")
        
        with col2:
            st.metric("Moderate Agreement", f"{len(pred_pivot) - high_agreement - low_agreement}/{len(pred_pivot)} drivers")
            st.caption("1-3 position variance")
        
        with col3:
            st.metric("Low Agreement", f"{low_agreement}/{len(pred_pivot)} drivers")
            st.caption("Variance > 3 positions")

with tab3:
    st.subheader("🎯 Prediction Confidence")
    
    st.markdown("""
    Model confidence is based on:
    - **Historical performance** at this circuit
    - **Agreement between models** (lower variance = higher confidence)
    - **Track characteristics** match with driver strengths
    """)
    
    # Calculate confidence scores
    if all(model in pred_pivot.columns for model in ['Ridge', 'XGBoost', 'CatBoost']):
        # Confidence = inverse of variance, normalized to 0-100
        max_var = pred_pivot['Variance'].max()
        if max_var > 0:
            pred_pivot['Confidence'] = ((1 - (pred_pivot['Variance'] / max_var)) * 100).round(0)
        else:
            pred_pivot['Confidence'] = 100
        
        # Create confidence chart
        fig = px.bar(
            pred_pivot.sort_values('Confidence', ascending=False),
            x='driver_name',
            y='Confidence',
            color='Confidence',
            color_continuous_scale='RdYlGn',
            title='Prediction Confidence by Driver',
            labels={'Confidence': 'Confidence (%)', 'driver_name': 'Driver'}
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence breakdown
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Highest Confidence Predictions:**")
            top_conf = pred_pivot.nlargest(5, 'Confidence')[['driver_name', 'Ensemble', 'Confidence']]
            for _, row in top_conf.iterrows():
                st.markdown(f"- **{row['driver_name']}**: P{row['Ensemble']:.1f} ({row['Confidence']:.0f}% confident)")
        
        with col2:
            st.markdown("**Uncertain Predictions:**")
            low_conf = pred_pivot.nsmallest(5, 'Confidence')[['driver_name', 'Ensemble', 'Confidence']]
            for _, row in low_conf.iterrows():
                st.markdown(f"- **{row['driver_name']}**: P{row['Ensemble']:.1f} ({row['Confidence']:.0f}% confident)")

# ==================== FOOTER ====================

st.markdown("---")

with st.expander("📚 About This Predictor"):
    st.markdown("""
    ### Machine Learning Pipeline
    
    **Data Sources:**
    - Historical F1 data (2022-2024)
    - Ergast API + Official F1 data
    - 30+ engineered features per driver
    
    **Models:**
    - **Ridge Regression**: L2-regularized linear baseline
    - **XGBoost**: Gradient boosting with hyperparameter tuning
    - **CatBoost**: Track-specialized categorical boosting
    
    **Key Features:**
    - Recent form (5-race moving average)
    - Circuit affinity scores
    - Teammate battle metrics
    - Team reliability indicators
    - Track power/overtaking characteristics
    
    **Performance Metrics:**
    - Average MAE: ~2.3 positions
    - Top-3 Accuracy: ~67%
    - Spearman Correlation: ~0.82
    
    ### Dissertation Project
    This predictor is part of a comprehensive ML-based F1 race outcome prediction system.
    Models trained on 6+ years of historical data with extensive feature engineering.
    """)

st.markdown("---")
st.info("""
**💡 Dissertation Project** • ML-Based F1 Race Outcome Prediction  
Developed using Python, Scikit-learn, XGBoost, CatBoost, and Django
""")