"""
Add this as a new file: pages/2_🔥_Chaos_Analysis.py

This creates a multi-page Streamlit app
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Chaos Analysis",
    page_icon="🔥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .chaos-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #ff4500 0%, #ff6347 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .impact-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff4500;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="chaos-header"><h1 style="color: white; margin: 0;">🔥 Chaos Analysis</h1></div>', unsafe_allow_html=True)

st.markdown("""
**Understanding Model Performance Under Race Incidents**  
How DNFs, penalties, safety cars, and other chaos events affect prediction accuracy
""")

# ==================== DATA LOADING ====================

@st.cache_data
def load_chaos_data():
    """Load chaos analysis data"""
    data_dir = Path(__file__).parent.parent / "data"
#    st.write("Looking for chaos data in:", data_dir.resolve())
#   st.write("Files found:", list(data_dir.glob("*.csv")))
    
    try:
        return {
            'raw': pd.read_csv(data_dir / "chaos_analysis_raw.csv"),
            'by_category': pd.read_csv(data_dir / "chaos_by_category.csv"),
            'by_event': pd.read_csv(data_dir / "chaos_by_event.csv"),
            'summary': pd.read_csv(data_dir / "chaos_summary.csv")
        }
    except FileNotFoundError:
        st.error("""
        ⚠️ **Chaos analysis data not found!**
        
        Please run from your Django project:
        ```bash
        python manage.py export_chaos_data
        ```
        
        Then copy the chaos CSV files to your Streamlit app's `data/` folder.
        """)
        st.stop()

# Load data
try:
    chaos_data = load_chaos_data()
except:
    st.warning("Chaos analysis data not yet available. This feature requires completed races with incident data.")
    st.info("The chaos analysis will show how the model performs when races have DNFs, penalties, safety cars, and other unpredictable events.")
    st.stop()

st.header("Understanding Model Performance Under Race Incidents")
st.write("How DNFs, penalties, safety cars, and other chaos events affect prediction accuracy")

if chaos_data['raw'].empty:
    st.warning("No chaos events recorded yet.")
else:
    st.write("Raw chaos data preview:")
    st.dataframe(chaos_data['raw'].head())

    st.write("Summary by category:")
    st.dataframe(chaos_data['by_category'])

    st.write("Summary by event:")
    st.dataframe(chaos_data['by_event'])
# ==================== OVERVIEW ====================

st.header("📊 Overview")

summary = chaos_data['summary'].iloc[0]

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Overall MAE",
        f"{summary['overall_mae']:.2f} pos",
        help="Mean Absolute Error across all predictions"
    )

with col2:
    st.metric(
        "Clean Race MAE",
        f"{summary['clean_mae']:.2f} pos",
        delta=f"{summary['clean_mae'] - summary['overall_mae']:.2f}",
        delta_color="inverse",
        help="MAE for races without major incidents"
    )

with col3:
    chaos_impact = summary['overall_mae'] - summary['clean_mae']
    st.metric(
        "Chaos Impact",
        f"+{chaos_impact:.2f} pos",
        help="Additional error caused by unpredictable events"
    )

with col4:
    affected_pct = (summary['affected_predictions'] / summary['total_predictions']) * 100
    st.metric(
        "Affected Predictions",
        f"{affected_pct:.1f}%",
        help="% of predictions impacted by race incidents"
    )

st.markdown("---")

# Explanation
with st.expander("💡 What is Chaos Analysis?"):
    st.markdown("""
    ### Understanding the Impact of Unpredictable Events
    
    F1 races are inherently unpredictable. Even with perfect ML models, certain events cannot be predicted:
    
    **Chaos Events:**
    - 🔴 **DNFs (Did Not Finish)**: Mechanical failures, crashes, retirements
    - ⚠️ **Penalties**: Post-race time penalties, grid penalties
    - 🚗 **Safety Cars**: Bunches up the field, changes strategy
    - 🌧️ **Weather Changes**: Sudden rain affecting tire strategy
    - 💥 **Collisions**: Contact between drivers
    
    **Key Metrics:**
    - **Overall MAE**: Average prediction error across all races
    - **Clean Race MAE**: Error when no major incidents occur
    - **Perfect Knowledge MAE**: Theoretical error if we knew all incidents in advance
    - **Chaos Impact**: Additional error caused by unpredictable events
    
    A good model should:
    1. Have low Clean Race MAE (accurate in normal conditions)
    2. Have small Chaos Impact (resilient to disruptions)
    3. Recover quickly after chaos events
    """)

# ==================== RACE CATEGORIES ====================

st.markdown("---")
st.header("🏁 Performance by Race Type")

col1, col2 = st.columns([3, 2])

with col1:
    # Bar chart of MAE by category
    fig = px.bar(
        chaos_data['by_category'].sort_values('mae'),
        x='category',
        y='mae',
        color='mae',
        color_continuous_scale='RdYlGn_r',
        title='Prediction Error by Race Type',
        labels={'mae': 'Mean Absolute Error', 'category': 'Race Category'},
        text='mae'
    )
    
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(height=400, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Race Categories")
    
    for _, row in chaos_data['by_category'].iterrows():
        category_emoji = {
            'clean': '✅',
            'minor_chaos': '⚠️',
            'major_chaos': '🔥',
            'extreme_chaos': '💥'
        }.get(row['category'], '❓')
        
        with st.container():
            st.markdown(f"**{category_emoji} {row['category'].replace('_', ' ').title()}**")
            st.write(f"MAE: {row['mae']:.2f} | Races: {row['count']} | Affected: {row['affected_drivers']}")
            st.progress(row['mae'] / chaos_data['by_category']['mae'].max())

# ==================== EVENT ANALYSIS ====================

st.markdown("---")
st.header("📅 Race-by-Race Analysis")

# Filter options
race_filter = st.selectbox(
    "Filter by Race Type",
    ['All'] + sorted(chaos_data['by_event']['race_category'].unique().tolist())
)

if race_filter != 'All':
    filtered_events = chaos_data['by_event'][chaos_data['by_event']['race_category'] == race_filter]
else:
    filtered_events = chaos_data['by_event']

# Sort by MAE
filtered_events = filtered_events.sort_values('mae', ascending=False)

# Display top chaos races
st.subheader("🔥 Most Chaotic Races")

col1, col2 = st.columns([2, 1])

with col1:
    top_chaos = filtered_events.head(10)
    
    fig = go.Figure()
    
    colors = top_chaos['race_category'].map({
        'clean': '#28a745',
        'minor_chaos': '#ffc107',
        'major_chaos': '#fd7e14',
        'extreme_chaos': '#dc3545'
    })
    
    fig.add_trace(go.Bar(
        y=top_chaos['event_name'],
        x=top_chaos['mae'],
        orientation='h',
        marker_color=colors,
        text=top_chaos['mae'].round(2),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>MAE: %{x:.2f}<br>Affected: %{customdata[0]}<extra></extra>',
        customdata=top_chaos[['affected_drivers']].values
    ))
    
    fig.update_layout(
        title='Top 10 Races by Prediction Error',
        xaxis_title='Mean Absolute Error',
        yaxis_title='',
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Top 3 Most Chaotic")
    
    for idx, row in top_chaos.head(3).iterrows():
        # Handle optional year column
        year_str = f" ({int(row['year'])})" if 'year' in row and pd.notna(row.get('year')) else ""
        
        st.markdown(f"""
        <div class="impact-card">
            <strong>{row['event_name']}{year_str}</strong><br>
            MAE: {row['mae']:.2f} positions<br>
            Category: {row['race_category'].replace('_', ' ').title()}<br>
            Affected: {row['affected_drivers']}/{row['total_drivers']} drivers
        </div>
        """, unsafe_allow_html=True)

# ==================== BEST PERFORMANCES ====================

st.subheader("✅ Best Predictions Despite Chaos")

best_chaos = filtered_events[
    filtered_events['race_category'] != 'clean'
].nsmallest(10, 'mae')

if not best_chaos.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Races where model performed well despite incidents:**")
        for _, row in best_chaos.head(5).iterrows():
            # Handle optional year column
            year_str = f" ({int(row['year'])})" if 'year' in row and pd.notna(row.get('year')) else ""
            
            st.markdown(f"""
            - **{row['event_name']}{year_str}**: {row['mae']:.2f} MAE
              - Category: {row['race_category'].replace('_', ' ')}
              - {row['affected_drivers']} drivers affected
            """)
    
    with col2:
        # Scatter plot: affected drivers vs MAE
        # Only include year in hover if column exists
        hover_cols = ['event_name']
        if 'year' in filtered_events.columns:
            hover_cols.append('year')
        
        fig = px.scatter(
            filtered_events,
            x='affected_drivers',
            y='mae',
            color='race_category',
            size='total_drivers',
            hover_data=hover_cols,
            title='Affected Drivers vs Prediction Error',
            labels={
                'affected_drivers': 'Number of Affected Drivers',
                'mae': 'Mean Absolute Error',
                'race_category': 'Race Type'
            }
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ==================== COUNTERFACTUAL ANALYSIS ====================

st.markdown("---")
st.header("🔮 Counterfactual Analysis")

st.markdown("""
**What if we had perfect knowledge of all race incidents?**

This analysis shows the theoretical best performance if the model knew about all DNFs, 
penalties, and other incidents in advance.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Actual MAE",
        f"{summary['overall_mae']:.2f}",
        help="Current model performance"
    )

with col2:
    st.metric(
        "Perfect Knowledge MAE",
        f"{summary['perfect_mae']:.2f}",
        delta=f"{summary['perfect_mae'] - summary['overall_mae']:.2f}",
        delta_color="inverse",
        help="If we knew all incidents in advance"
    )

with col3:
    improvement = ((summary['overall_mae'] - summary['perfect_mae']) / summary['overall_mae']) * 100
    st.metric(
        "Potential Improvement",
        f"{improvement:.1f}%",
        help="% improvement with perfect incident knowledge"
    )

# Visualization
st.markdown("### Performance Breakdown")

performance_data = pd.DataFrame({
    'Scenario': ['Actual', 'Clean Races Only', 'Perfect Knowledge'],
    'MAE': [summary['overall_mae'], summary['clean_mae'], summary['perfect_mae']],
    'Color': ['#dc3545', '#28a745', '#007bff']
})

fig = go.Figure()

fig.add_trace(go.Bar(
    x=performance_data['Scenario'],
    y=performance_data['MAE'],
    marker_color=performance_data['Color'],
    text=performance_data['MAE'].round(2),
    textposition='outside'
))

fig.update_layout(
    title='Model Performance Under Different Scenarios',
    yaxis_title='Mean Absolute Error (positions)',
    height=400,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# ==================== KEY INSIGHTS ====================

st.markdown("---")
st.header("💡 Key Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Strengths")
    
    clean_mae = summary['clean_mae']
    chaos_resilience = 1 - (chaos_impact / summary['overall_mae'])
    
    st.markdown(f"""
    - ✅ **Clean race accuracy**: {clean_mae:.2f} MAE
    - ✅ **Chaos resilience**: {chaos_resilience*100:.1f}% of performance maintained
    - ✅ **Recovery ability**: Model adjusts well after early incidents
    - ✅ **Track specialization**: Better at predicting circuit-specific patterns
    """)

with col2:
    st.markdown("### Areas for Improvement")
    
    st.markdown(f"""
    - 🔄 **DNF prediction**: Cannot predict mechanical failures
    - 🔄 **Safety car timing**: Random timing affects strategy
    - 🔄 **Weather changes**: Sudden rain unpredictable
    - 🔄 **First-lap incidents**: Early chaos hardest to predict
    """)

# ==================== CONCLUSIONS ====================

st.markdown("---")
with st.expander("📋 Methodology & Conclusions"):
    st.markdown(f"""
    ### Analysis Methodology
    
    **Data Collection:**
    - Analyzed {summary['total_predictions']} predictions across multiple seasons
    - Identified {summary['affected_predictions']} predictions affected by incidents
    - Categorized races into 4 chaos levels (clean → extreme chaos)
    
    **Incident Classification:**
    - **Clean**: No major incidents affecting race outcome
    - **Minor Chaos**: 1-2 DNFs or minor penalties
    - **Major Chaos**: 3+ DNFs, safety car, or significant penalties
    - **Extreme Chaos**: First-lap crashes, red flags, major weather changes
    
    **Key Findings:**
    1. Model performs {clean_mae:.2f} MAE in clean conditions
    2. Chaos adds approximately {chaos_impact:.2f} positions of error
    3. {affected_pct:.1f}% of predictions are affected by unpredictable events
    4. Perfect incident knowledge could improve MAE by {improvement:.1f}%
    
    **Conclusions:**
    - The model's core predictions are accurate ({clean_mae:.2f} MAE in clean races)
    - Most error comes from genuinely unpredictable events
    - Track specialization helps mitigate some chaos effects
    - Real-time incident detection could significantly improve live predictions
    
    ### Future Improvements
    - Integrate live race data for real-time adjustments
    - Add incident probability models (DNF risk, safety car likelihood)
    - Develop fallback predictions for different incident scenarios
    - Improve first-lap chaos handling
    """)