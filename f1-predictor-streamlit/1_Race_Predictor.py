import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import time

# Page config
st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide"
)

# ==================== TEAM COLORS & DRIVER NUMBERS ====================

TEAM_COLORS = {
    'Red Bull Racing': '#3671C6',
    'Ferrari': '#E8002D',
    'Mercedes': '#27F4D2',
    'McLaren': '#FF8000',
    'Aston Martin': '#229971',
    'Alpine': '#FF87BC',
    'Williams': '#64C4FF',
    'AlphaTauri': '#5E8FAA',
    'Alfa Romeo': '#C92D4B',
    'Haas': '#B6BABD',
    'RB': '#6692FF',
    'Kick Sauber': '#52E252'
}

RACE_FLAGS = {
    'Bahrain': '🇧🇭', 'Saudi Arabia': '🇸🇦', 'Australia': '🇦🇺', 'Japan': '🇯🇵',
    'China': '🇨🇳', 'Miami': '🇺🇸', 'Emilia Romagna': '🇮🇹', 'Monaco': '🇲🇨',
    'Canada': '🇨🇦', 'Spain': '🇪🇸', 'Austria': '🇦🇹', 'Great Britain': '🇬🇧',
    'Hungary': '🇭🇺', 'Belgium': '🇧🇪', 'Netherlands': '🇳🇱', 'Italy': '🇮🇹',
    'Azerbaijan': '🇦🇿', 'Singapore': '🇸🇬', 'United States': '🇺🇸', 'Mexico': '🇲🇽',
    'Brazil': '🇧🇷', 'Las Vegas': '🇺🇸', 'Qatar': '🇶🇦', 'Abu Dhabi': '🇦🇪'
}

# ==================== DATA LOADING ====================

@st.cache_data(show_spinner=False)
def load_data():
    """Load all predictor data"""
    data_dir = Path(__file__).resolve().parent / "data"
    
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

# Load data with spinner
with st.spinner('🏎️ Loading F1 data...'):
    data = load_data()
    time.sleep(0.5)  # Brief pause for smooth UX

# ==================== HELPER FUNCTIONS ====================

def get_team_color(driver_name, drivers_df):
    """Get team color for a driver"""
    driver_row = drivers_df[drivers_df['full_name'] == driver_name]
    if not driver_row.empty and 'team' in driver_row.columns:
        team = driver_row.iloc[0]['team']
        return TEAM_COLORS.get(team, '#45B7D1')
    return '#45B7D1'

def get_driver_number(driver_name, drivers_df):
    """Get driver number"""
    driver_row = drivers_df[drivers_df['full_name'] == driver_name]
    if not driver_row.empty and 'number' in driver_row.columns:
        return int(driver_row.iloc[0]['number'])
    return ''

def format_driver_name(driver_name, drivers_df):
    """Format driver name with number"""
    number = get_driver_number(driver_name, drivers_df)
    if number:
        return f"#{number} {driver_name}"
    return driver_name

def get_race_flag(event_name):
    """Get flag emoji for race"""
    for country, flag in RACE_FLAGS.items():
        if country.lower() in event_name.lower():
            return flag
    return '🏁'

# ==================== SIDEBAR ====================

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
    
    st.header("📊 About the Models")
    
    with st.expander("ℹ️ Model Details"):
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
        
        ### Ensemble
        Average of all three models for robust predictions
        """)
    
    st.markdown("---")
    
    # Driver search and multiselect
    st.subheader("🔍 Filter Drivers")
    driver_search = st.text_input("Search drivers", placeholder="Type driver name...")
    
    # Filter drivers based on search
    available_drivers = data['drivers']['full_name'].tolist()
    if driver_search:
        available_drivers = [d for d in available_drivers if driver_search.lower() in d.lower()]
    
    selected_drivers = st.multiselect(
        "Highlight drivers",
        options=available_drivers,
        default=[]
    )
    
    st.markdown("---")
    
    # Export options
    st.subheader("💾 Export Options")
    if st.button("📥 Download Predictions CSV"):
        st.info("CSV download feature coming soon!")
    
    if st.button("📊 Generate Race Report"):
        st.info("PDF report feature coming soon!")
    
    st.markdown("---")
    st.markdown("**Dissertation Project**  \n*ML-Based F1 Prediction System*")
    
    # Glossary
    with st.expander("📖 F1 Glossary"):
        st.markdown("""
        **MAE**: Mean Absolute Error - average prediction error in positions
        
        **Ensemble**: Combined prediction from all models
        
        **DNF**: Did Not Finish - retirement from race
        
        **Confidence**: How much models agree (high = low variance)
        
        **Track Affinity**: Historical driver performance at circuit
        """)

# ==================== THEME CONFIGURATION ====================

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
    podium_gold = "#FFD700"
    podium_silver = "#C0C0C0"
    podium_bronze = "#CD7F32"
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
    podium_gold = "#FFD700"
    podium_silver = "#C0C0C0"
    podium_bronze = "#CD7F32"

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
    .main-header {{
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
    
    .main-header h1 {{
        color: white !important;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    /* Podium styling */
    .podium-container {{
        display: flex;
        justify-content: center;
        align-items: flex-end;
        gap: 20px;
        margin: 2rem 0;
        padding: 2rem;
        background: {card_bg};
        border-radius: 12px;
        animation: fadeIn 0.8s ease-out;
        flex-wrap: wrap; 
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    
    .podium-place {{
        text-align: center;
        padding: 1.5rem 1rem;
        border-radius: 8px;
        flex: 0 0 auto;
        min-width: 150px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: popIn 0.5s ease-out backwards;
    }}
    
    .podium-place:hover {{
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }}
    
    @keyframes popIn {{
        from {{
            transform: scale(0.8);
            opacity: 0;
        }}
        to {{
            transform: scale(1);
            opacity: 1;
        }}
    }}
    
    .podium-1 {{
        background: linear-gradient(135deg, {podium_gold}, #FFA500);
        animation-delay: 0.2s;
    }}
    
    .podium-2 {{
        background: linear-gradient(135deg, {podium_silver}, #A8A8A8);
        animation-delay: 0.1s;
    }}
    
    .podium-3 {{
        background: linear-gradient(135deg, {podium_bronze}, #A0522D);
        animation-delay: 0.3s;
    }}
    
    .podium-trophy {{
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }}
    
    .podium-position {{
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }}
    
    .podium-driver {{
        font-size: 1.1rem;
        font-weight: bold;
        color: white;
        margin-top: 0.5rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }}
    
    /* Metric cards with animation */
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
    
    /* Custom table styling */
    .prediction-table {{
        background-color: {table_bg};
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid {border_color};
        animation: fadeIn 0.6s ease-out;
    }}
    
    .prediction-table th {{
        background-color: {table_header_bg};
        color: {text_color};
        padding: 12px;
        text-align: left;
        font-weight: bold;
        border-bottom: 2px solid {border_color};
        position: sticky;
        top: 0;
        z-index: 10;
    }}
    
    .prediction-table td {{
        background-color: {table_bg};
        color: {text_color};
        padding: 10px 12px;
        border-bottom: 1px solid {border_color};
        transition: background-color 0.2s ease;
    }}
    
    .prediction-table tr:hover td {{
        background-color: {card_bg};
    }}
    
    .highlighted-driver {{
        background-color: #ffd700 !important;
        color: #000000 !important;
        font-weight: bold;
    }}
    
    /* Team color indicators */
    .team-color-dot {{
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        vertical-align: middle;
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
    
    /* Multi-select pills */
    [data-baseweb="tag"] {{
        background-color: {metric_border} !important;
        color: white !important;
    }}
    
    /* Input fields */
    input {{
        background-color: {input_bg} !important;
        color: {input_text} !important;
    }}
    /* Button styling */
    .stButton > button {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
    }}
    
    .stButton > button:hover {{
        background-color: {metric_border} !important;
        color: white !important;
        border-color: {metric_border} !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2) !important;
    }}
    
    .stButton > button:active {{
        transform: translateY(0) !important;
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
    
    /* Loading skeleton */
    .skeleton {{
        background: linear-gradient(90deg, {card_bg} 25%, {metric_bg} 50%, {card_bg} 75%);
        background-size: 200% 100%;
        animation: loading 1.5s infinite;
        border-radius: 4px;
        height: 20px;
        margin: 10px 0;
    }}
    
    @keyframes loading {{
        0% {{ background-position: 200% 0; }}
        100% {{ background-position: -200% 0; }}
    }}
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {{
        .podium-container {{
            flex-direction: column;
            align-items: center;
        }}
        
        .podium-place {{
            min-width: 100%;
            margin-bottom: 1rem;
        }}
        
        .prediction-table {{
            overflow-x: auto;
            display: block;
        }}
        
        .prediction-table table {{
            min-width: 600px;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# ==================== TITLE ====================

st.markdown('<div class="main-header"><h1 style="color: white; margin: 0;">🏎️ F1 Race Predictor</h1></div>', unsafe_allow_html=True)

st.markdown("""
**Machine Learning-Powered Race Predictions** • Trained on 2022-2024 historical data  
Compare **Ridge Regression**, **XGBoost**, and **CatBoost** models with track specialization
""")

# ==================== RACE SELECTION ====================

st.header("🎯 Select Race")

col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    # Get unique events
    events = data['events'].sort_values('round')
    
    # Add flags to event names
    event_display_names = []
    event_actual_names = []
    for _, event in events.iterrows():
        flag = get_race_flag(event['name'])
        display_name = f"{flag} {event['name']}"
        event_display_names.append(display_name)
        event_actual_names.append(event['name'])
    
    selected_display = st.selectbox(
        "Choose a race",
        event_display_names,
        help="Select a race from the 2025 season"
    )
    
    # Get actual event name
    selected_idx = event_display_names.index(selected_display)
    selected_event_name = event_actual_names[selected_idx]
    selected_event = events[events['name'] == selected_event_name].iloc[0]

with col2:
    st.metric("Round", f"R{selected_event['round']}", help="Race round number")

with col3:
    # Compare races option
    if st.button("📊 Compare Races"):
        st.info("Multi-race comparison feature coming soon!")

# Track characteristics
st.markdown("---")
st.subheader(f"📍 {selected_event['circuit_name']}")

track_info = data['tracks'][data['tracks']['circuit_name'] == selected_event['circuit_name']]

if not track_info.empty:
    track = track_info.iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Track Type", 
            track['track_category'],
            help="Circuit classification: Street, Road, or Hybrid"
        )
    with col2:
        st.metric(
            "Power Sensitivity ⚡", 
            f"{track['power_sensitivity']:.1f}/10",
            help="How much engine power affects lap time"
        )
    with col3:
        st.metric(
            "Overtaking 🏁", 
            f"{track['overtaking_difficulty']:.1f}/10",
            help="Difficulty of overtaking (higher = harder)"
        )
    with col4:
        st.metric(
            "Quali Importance 🏆", 
            f"{track['qualifying_importance']:.1f}/10",
            help="How much qualifying position affects race result"
        )

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

# Calculate ensemble
if all(model in pred_pivot.columns for model in ['Ridge', 'XGBoost', 'CatBoost']):
    pred_pivot['Ensemble'] = pred_pivot[['Ridge', 'XGBoost', 'CatBoost']].mean(axis=1)
elif 'Ridge' in pred_pivot.columns and 'XGBoost' in pred_pivot.columns:
    pred_pivot['Ensemble'] = pred_pivot[['Ridge', 'XGBoost']].mean(axis=1)
else:
    pred_pivot['Ensemble'] = pred_pivot.iloc[:, 1]

# Sort by ensemble prediction
pred_pivot = pred_pivot.sort_values('Ensemble').reset_index(drop=True)
pred_pivot.index = pred_pivot.index + 1

# ==================== PODIUM VISUALIZATION ====================

if len(pred_pivot) >= 3:
    st.subheader("🏆 Predicted Podium")
    
    top3 = pred_pivot.head(3)
    
    # Use Streamlit columns for podium (2nd, 1st, 3rd)
    col1, col2, col3 = st.columns(3)
    
    with col1:  # 2nd place
        st.markdown(f'''
        <div class="podium-place podium-2">
            <div class="podium-trophy">🥈</div>
            <div class="podium-position">2nd</div>
            <div class="podium-driver">{top3.iloc[1]['driver_name']}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:  # 1st place
        st.markdown(f'''
        <div class="podium-place podium-1">
            <div class="podium-trophy">🥇</div>
            <div class="podium-position">1st</div>
            <div class="podium-driver">{top3.iloc[0]['driver_name']}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:  # 3rd place
        st.markdown(f'''
        <div class="podium-place podium-3">
            <div class="podium-trophy">🥉</div>
            <div class="podium-position">3rd</div>
            <div class="podium-driver">{top3.iloc[2]['driver_name']}</div>
        </div>
        ''', unsafe_allow_html=True)

# ==================== TABS ====================

tab1, tab2, tab3, tab4 = st.tabs(["📋 Predicted Grid", "📊 Model Comparison", "🎯 Confidence Analysis", "📈 Driver Trends"])

with tab1:
    st.subheader("Predicted Finishing Order")
    
    # Format predictions table
    display_df = pred_pivot.copy()
    display_df.index.name = 'Pos'
    
    # Round predictions
    for col in ['Ridge', 'XGBoost', 'CatBoost', 'Ensemble']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(1)
    
    # Add actual results if available
    actual_results = event_preds[event_preds['actual_position'].notna()]
    if not actual_results.empty:
        actual_map = dict(zip(actual_results['driver_name'], actual_results['actual_position']))
        display_df['Actual'] = display_df['driver_name'].map(actual_map)
        
        if 'Ensemble' in display_df.columns and 'Actual' in display_df.columns:
            display_df['Diff'] = (display_df['Ensemble'] - display_df['Actual']).round(1)
    
    # Create HTML table with team colors
    html_table = '<div class="prediction-table"><table style="width:100%"><thead><tr>'
    html_table += '<th style="width:8%">Pos</th><th style="width:30%">Driver</th>'
    
    if 'Ridge' in display_df.columns:
        html_table += '<th style="width:12%">Ridge</th>'
    if 'XGBoost' in display_df.columns:
        html_table += '<th style="width:12%">XGBoost</th>'
    if 'CatBoost' in display_df.columns:
        html_table += '<th style="width:12%">CatBoost</th>'
    if 'Ensemble' in display_df.columns:
        html_table += '<th style="width:12%">Ensemble</th>'
    if 'Actual' in display_df.columns:
        html_table += '<th style="width:10%">Actual</th><th style="width:8%">Diff</th>'
    
    html_table += '</tr></thead><tbody>'
    
    for idx, row in display_df.iterrows():
        highlight_class = 'highlighted-driver' if row['driver_name'] in selected_drivers else ''
        team_color = get_team_color(row['driver_name'], data['drivers'])
        driver_num = get_driver_number(row['driver_name'], data['drivers'])
        
        driver_display = f'<span class="team-color-dot" style="background-color: {team_color};"></span>'
        if driver_num:
            driver_display += f'#{driver_num} '
        driver_display += row['driver_name']
        
        html_table += f'<tr><td class="{highlight_class}">{idx}</td><td class="{highlight_class}">{driver_display}</td>'
        
        if 'Ridge' in display_df.columns:
            html_table += f'<td class="{highlight_class}">{row["Ridge"]:.1f}</td>'
        if 'XGBoost' in display_df.columns:
            html_table += f'<td class="{highlight_class}">{row["XGBoost"]:.1f}</td>'
        if 'CatBoost' in display_df.columns:
            html_table += f'<td class="{highlight_class}">{row["CatBoost"]:.1f}</td>'
        if 'Ensemble' in display_df.columns:
            html_table += f'<td class="{highlight_class}"><strong>{row["Ensemble"]:.1f}</strong></td>'
        if 'Actual' in display_df.columns:
            html_table += f'<td class="{highlight_class}">{row["Actual"]:.0f}</td>'
            diff_sign = '+' if row["Diff"] > 0 else ''
            html_table += f'<td class="{highlight_class}">{diff_sign}{row["Diff"]:.1f}</td>'
        
        html_table += '</tr>'
    
    html_table += '</tbody></table></div>'
    
    st.markdown(html_table, unsafe_allow_html=True)
    
    # Show prediction statistics
    if 'Actual' in display_df.columns:
        st.markdown("---")
        st.subheader("📈 Prediction Accuracy")
        
        col1, col2, col3, col4 = st.columns(4)
        
        mae = abs(display_df['Ensemble'] - display_df['Actual']).mean()
        perfect_preds = (display_df['Ensemble'].round() == display_df['Actual']).sum()
        top3_accuracy = (display_df[display_df['Actual'] <= 3]['Ensemble'] <= 3.5).sum()
        max_error = abs(display_df['Ensemble'] - display_df['Actual']).max()
        
        with col1:
            st.metric("Mean Absolute Error", f"{mae:.2f} pos", help="Average prediction error")
        with col2:
            st.metric("Perfect Predictions", f"{perfect_preds}/{len(display_df)}", help="Exact position matches")
        with col3:
            st.metric("Top-3 Accuracy", f"{top3_accuracy}/3", help="Correct podium predictions")
        with col4:
            st.metric("Max Error", f"{max_error:.1f} pos", help="Largest prediction miss")

with tab2:
    st.subheader("Model Predictions Comparison")
    
    comparison_df = pred_pivot.copy()
    
    # Create bar chart with team colors
    driver_colors = [get_team_color(d, data['drivers']) for d in comparison_df['driver_name']]
    highlighted = [d in selected_drivers for d in comparison_df['driver_name']]
    bar_colors = ['#FFD700' if h else c for h, c in zip(highlighted, driver_colors)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=comparison_df['driver_name'],
        y=comparison_df['Ensemble'],
        marker_color=bar_colors,
        text=comparison_df['Ensemble'].round(1),
        textposition='outside',
        textfont=dict(size=11),
        name='Ensemble Prediction',
        hovertemplate='<b>%{x}</b><br>Predicted: P%{y:.1f}<extra></extra>'
    ))
    
    # Add actual results if available
    if 'Actual' in display_df.columns:
        actual_vals = display_df.set_index('driver_name')['Actual']
        fig.add_trace(go.Scatter(
            x=comparison_df['driver_name'],
            y=comparison_df['driver_name'].map(actual_vals),
            mode='markers',
            marker=dict(size=12, color='#e10600', symbol='diamond'),
            name='Actual Result',
            hovertemplate='<b>%{x}</b><br>Actual: P%{y:.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        title="Ensemble Predicted Position by Driver",
        xaxis_title="Driver",
        yaxis_title="Position",
        height=600,
        template=plotly_template,
        hovermode='x unified',
        showlegend=True,
        yaxis=dict(autorange='reversed')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Model breakdown
    st.markdown("---")
    st.subheader("Individual Model Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if 'Ridge' in comparison_df.columns:
            st.markdown("**🔵 Ridge Model:**")
            ridge_df = comparison_df[['driver_name', 'Ridge']].sort_values('Ridge').head(10)
            for i, (idx, row) in enumerate(ridge_df.iterrows(), 1):
                st.markdown(f"{i}. {row['driver_name']}: P{row['Ridge']:.1f}")
    
    with col2:
        if 'XGBoost' in comparison_df.columns:
            st.markdown("**🟢 XGBoost Model:**")
            xgb_df = comparison_df[['driver_name', 'XGBoost']].sort_values('XGBoost').head(10)
            for i, (idx, row) in enumerate(xgb_df.iterrows(), 1):
                st.markdown(f"{i}. {row['driver_name']}: P{row['XGBoost']:.1f}")
    
    with col3:
        if 'CatBoost' in comparison_df.columns:
            st.markdown("**🟡 CatBoost Model:**")
            cat_df = comparison_df[['driver_name', 'CatBoost']].sort_values('CatBoost').head(10)
            for i, (idx, row) in enumerate(cat_df.iterrows(), 1):
                st.markdown(f"{i}. {row['driver_name']}: P{row['CatBoost']:.1f}")

with tab3:
    st.subheader("🎯 Prediction Confidence")
    
    st.markdown("""
    Model confidence is based on **agreement between models**. Lower variance = higher confidence.
    """)
    
    # Calculate confidence scores
    if all(model in pred_pivot.columns for model in ['Ridge', 'XGBoost', 'CatBoost']):
        pred_pivot['Variance'] = pred_pivot[['Ridge', 'XGBoost', 'CatBoost']].var(axis=1)
        
        max_var = pred_pivot['Variance'].max()
        if max_var > 0:
            pred_pivot['Confidence'] = ((1 - (pred_pivot['Variance'] / max_var)) * 100).round(0)
        else:
            pred_pivot['Confidence'] = 100
        
        sorted_df = pred_pivot.sort_values('Confidence', ascending=False).copy()
        
        # Confidence bar chart
        driver_colors_conf = [get_team_color(d, data['drivers']) for d in sorted_df['driver_name']]
        bar_colors_conf = []
        for idx, row in sorted_df.iterrows():
            if row['driver_name'] in selected_drivers:
                bar_colors_conf.append('#FFD700')
            elif row['Confidence'] >= 80:
                bar_colors_conf.append('#00cc00')
            elif row['Confidence'] <= 40:
                bar_colors_conf.append('#ff6b6b')
            else:
                bar_colors_conf.append(driver_colors_conf[len(bar_colors_conf)])
        
        fig = go.Figure(go.Bar(
            x=sorted_df['driver_name'],
            y=sorted_df['Confidence'],
            marker_color=bar_colors_conf,
            text=sorted_df['Confidence'].astype(int),
            textposition='outside',
            texttemplate='%{text}%',
            textfont=dict(size=11),
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.0f}%<br>Variance: %{customdata:.2f}<extra></extra>',
            customdata=sorted_df['Variance']
        ))
        
        fig.update_layout(
            title='Prediction Confidence by Driver',
            xaxis_title='Driver',
            yaxis_title='Confidence (%)',
            height=600,
            template=plotly_template,
            showlegend=False,
            yaxis=dict(range=[0, 110])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**💡 Color Guide:** 🟢 High (80%+) • 🔴 Low (≤40%) • Team colors (Moderate) • 🟡 Selected drivers")
        
        # Confidence breakdown
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✅ Highest Confidence:**")
            top_conf = pred_pivot.nlargest(5, 'Confidence')[['driver_name', 'Ensemble', 'Confidence', 'Variance']]
            for _, row in top_conf.iterrows():
                st.markdown(f"- **{row['driver_name']}**: P{row['Ensemble']:.1f} ({row['Confidence']:.0f}% • var: {row['Variance']:.2f})")
        
        with col2:
            st.markdown("**⚠️ Uncertain Predictions:**")
            low_conf = pred_pivot.nsmallest(5, 'Confidence')[['driver_name', 'Ensemble', 'Confidence', 'Variance']]
            for _, row in low_conf.iterrows():
                st.markdown(f"- **{row['driver_name']}**: P{row['Ensemble']:.1f} ({row['Confidence']:.0f}% • var: {row['Variance']:.2f})")

with tab4:
    st.subheader("📈 Driver Performance Trends")
    
    st.info("🚧 Feature coming soon: Season-long performance trends, track affinity scores, and head-to-head comparisons")
    
    # Placeholder for future feature
    if selected_drivers:
        st.markdown(f"**Selected Drivers:** {', '.join(selected_drivers)}")
        st.markdown("This section will show:")
        st.markdown("- Season performance trajectory")
        st.markdown("- Historical performance at this circuit")
        st.markdown("- Recent form (last 5 races)")
        st.markdown("- Comparison with teammates")
    else:
        st.markdown("Select drivers in the sidebar to see their performance trends here.")

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
    - **Ensemble**: Average of all three for robust predictions
    
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