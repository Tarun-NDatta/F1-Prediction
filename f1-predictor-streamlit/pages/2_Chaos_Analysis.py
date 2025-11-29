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

# ==================== DATA LOADING ====================

@st.cache_data
def load_chaos_data():
    """Load chaos analysis data"""
    data_dir = Path(__file__).parent.parent / "data"
    
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

# ==================== SIDEBAR WITH THEME ====================

if 'theme' not in st.session_state:
    st.session_state.theme = "Night Mode"

with st.sidebar:
    # Theme selection at the top with session state
    theme = st.selectbox(
        "🌙 Theme", 
        ["Day Mode", "Night Mode"], 
        index=0 if st.session_state.theme == "Day Mode" else 1,
        key="theme_selector"
    )
    
    # Update session state when theme changes
    if theme != st.session_state.theme:
        st.session_state.theme = theme
    
    st.markdown("---")
    
    st.header("📊 About Chaos Analysis")
    
    st.markdown("""
    ### What is Chaos?
    
    **Unpredictable race events:**
    - 🔴 DNFs & Retirements
    - ⚠️ Penalties
    - 🚗 Safety Cars
    - 🌧️ Weather Changes
    - 💥 Collisions
    
    ### Categories
    - **Clean**: No incidents
    - **Minor**: 1-2 incidents
    - **Major**: 3+ incidents
    - **Extreme**: Red flags
    """)
    
    st.markdown("---")
    st.markdown("**Dissertation Project**  \n*ML-Based F1 Prediction System*")

# ==================== THEME CONFIGURATION ====================

# Set theme variables based on selection
if theme == "Night Mode":
    plotly_template = "plotly_dark"
    app_bg_color = "#0e1117"
    sidebar_bg_color = "#262730"
    text_color = "#fafafa"
    header_gradient = "linear-gradient(90deg, #cc0000 0%, #ff4500 100%)"
    metric_bg = "#1e1e1e"
    metric_border = "#ff4500"
    card_bg = "#262730"
    input_bg = "#1e1e1e"
    input_text = "#fafafa"
    table_bg = "#1e1e1e"
    table_header_bg = "#262730"
    border_color = "#404040"
    impact_card_bg = "#262730"
    impact_card_border = "#ff4500"
else:
    plotly_template = "plotly_white"
    app_bg_color = "#ffffff"
    sidebar_bg_color = "#f0f2f6"
    text_color = "#262730"
    header_gradient = "linear-gradient(90deg, #e10600 0%, #ff1801 100%)"
    metric_bg = "#f0f2f6"
    metric_border = "#e10600"
    card_bg = "#f8f9fa"
    input_bg = "#ffffff"
    input_text = "#262730"
    table_bg = "#ffffff"
    table_header_bg = "#f0f2f6"
    border_color = "#e0e0e0"
    impact_card_bg = "#f8f9fa"
    impact_card_border = "#e10600"

# Apply theme CSS
st.markdown(f"""
<style>
    /* Main app container */
    [data-testid="stAppViewContainer"] {{
        background-color: {app_bg_color};
        transition: background-color 0.3s ease;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg_color};
        transition: background-color 0.3s ease;
    }}
    
    /* All text elements */
    .stMarkdown, .stText, p, span, label {{
        color: {text_color} !important;
        transition: color 0.3s ease;
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
        transition: color 0.3s ease;
    }}
    
    /* Main header with animation */
    .chaos-header {{
        text-align: center;
        padding: 1.5rem 0;
        background: {header_gradient};
        border-radius: 10px;
        margin-bottom: 2rem;
        animation: slideDown 0.5s ease-out;
    }}
    
    @keyframes slideDown {{
        from {{
            opacity: 0;
            transform: translateY(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}
    
    .chaos-header h1 {{
        color: white !important;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    /* Impact cards with animation */
    .impact-card {{
        background: {impact_card_bg};
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid {impact_card_border};
        margin: 1rem 0;
        color: {text_color};
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideIn 0.5s ease-out;
    }}
    
    .impact-card:hover {{
        transform: translateX(5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    
    @keyframes slideIn {{
        from {{
            opacity: 0;
            transform: translateX(-20px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    
    /* Metric cards */
    [data-testid="stMetricValue"] {{
        color: {text_color} !important;
        font-size: 1.5rem !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {text_color} !important;
    }}
    
    .metric-card {{
        background: {metric_bg} !important;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid {metric_border};
        animation: slideIn 0.5s ease-out;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        color: {text_color} !important;
        background-color: {card_bg};
        transition: all 0.3s ease;
    }}
    
    .stTabs [data-baseweb="tab"]:hover {{
        transform: translateY(-2px);
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {metric_border} !important;
        color: white !important;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }}
    
    .streamlit-expanderHeader:hover {{
        border-color: {metric_border} !important;
    }}
    
    .streamlit-expanderContent {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }}
    
    /* Info/Warning boxes */
    .stAlert {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border-radius: 8px;
    }}
    
    /* Select boxes */
    [data-baseweb="select"] {{
        background-color: {input_bg} !important;
    }}
    
    [data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
    }}
    
    /* Dropdown menu */
    [data-baseweb="popover"] {{
        background-color: {input_bg} !important;
    }}
    
    [role="listbox"] {{
        background-color: {input_bg} !important;
    }}
    
    [role="option"] {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
        transition: background-color 0.2s ease;
    }}
    
    [role="option"]:hover {{
        background-color: {card_bg} !important;
    }}
    
    /* Dataframe styling */
    [data-testid="stDataFrame"] {{
        background-color: {table_bg} !important;
    }}
    
    [data-testid="stDataFrame"] div[role="gridcell"] {{
        background-color: {table_bg} !important;
        color: {text_color} !important;
    }}
    
    [data-testid="stDataFrame"] div[role="columnheader"] {{
        background-color: {table_header_bg} !important;
        color: {text_color} !important;
    }}
    
    /* Progress bars */
    .stProgress > div > div {{
        background-color: {metric_border} !important;
    }}
    
    /* Container animations */
    .element-container {{
        animation: fadeIn 0.6s ease-out;
    }}
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {{
        .chaos-header h1 {{
            font-size: 1.5rem;
        }}
        
        .impact-card {{
            margin: 0.5rem 0;
            padding: 0.75rem;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ==================== TITLE ====================

st.markdown('<div class="chaos-header"><h1 style="color: white; margin: 0;">🔥 Chaos Analysis</h1></div>', unsafe_allow_html=True)

st.markdown("""
**Understanding Model Performance Under Race Incidents**  
How DNFs, penalties, safety cars, and other chaos events affect prediction accuracy
""")

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
        text='mae',
        template=plotly_template
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
        yaxis={'categoryorder': 'total ascending'},
        template=plotly_template
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
            },
            template=plotly_template
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
    showlegend=False,
    template=plotly_template
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

st.markdown("---")
st.info("""
**💡 Dissertation Project** • ML-Based F1 Race Outcome Prediction  
Developed using Python, Scikit-learn, XGBoost, CatBoost, and Django
""")