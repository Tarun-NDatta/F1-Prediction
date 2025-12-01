import streamlit as st
from pathlib import Path
from PIL import Image

# Page config
st.set_page_config(
    page_title="Django Project Showcase",
    page_icon="📸",
    layout="wide"
)
# Initialize session state for theme persistence
if 'theme' not in st.session_state:
    st.session_state.theme = "Day Mode"

# Add sidebar with theme selector
with st.sidebar:
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
    st.markdown("**Navigate using the sidebar menu above** ☝️")

# Use the theme from session state
theme = st.session_state.theme

# Theme variables (rest of your existing code)
if theme == "Night Mode":
    app_bg_color = "#0e1117"
    ...

# Load the same theme from session state
if 'theme' not in st.session_state:
    st.session_state.theme = "Day Mode"


# Theme variables
if theme == "Night Mode":
    app_bg_color = "#0e1117"
    sidebar_bg_color = "#262730"  
    text_color = "#fafafa"
    card_bg = "#262730"
    border_color = "#404040"
    header_gradient = "linear-gradient(90deg, #cc0000 0%, #ff4500 100%)"
else:
    app_bg_color = "#ffffff"
    sidebar_bg_color = "#f0f2f6"  
    text_color = "#262730"
    card_bg = "#f8f9fa"
    border_color = "#e0e0e0"
    header_gradient = "linear-gradient(90deg, #e10600 0%, #ff1801 100%)"

# Apply theme CSS
st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{
        background-color: {app_bg_color};
        transition: background-color 0.3s ease;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {sidebar_bg_color};
        transition: background-color 0.3s ease;
    }}
    
    .stMarkdown, .stText, p, span, h1, h2, h3, h4, h5, h6 {{
        color: {text_color} !important;
        transition: color 0.3s ease;
    }}
    
    .main-header {{
        text-align: center;
        padding: 1.5rem 0;
        background: {header_gradient};
        border-radius: 10px;
        margin-bottom: 2rem;
    }}
    
    .main-header h1 {{
        color: white !important;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }}
    
    .screenshot-container {{
        background: {card_bg};
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid {border_color};
        margin-bottom: 2rem;
    }}
    
    .stAlert {{
        background-color: {card_bg} !important;
        color: {text_color} !important;
    }}
</style>
""", unsafe_allow_html=True)

# Page content
st.markdown('<div class="main-header"><h1>📸 Full Django Project</h1></div>', unsafe_allow_html=True)

st.markdown("""
**Primary Development Platform**

This is the comprehensive Django web application mentioned on the home page - the full-stack implementation that serves as the primary development and testing platform.

### 🎯 Complete ML-Based F1 Prediction System
This Streamlit interface is a **live demo** of predictions from the full Django application.

**Full Django Project Features:**
- 🔐 User authentication & role-based access control
- 📊 Complete admin dashboard for data management
- 🤖 Model training & hyperparameter tuning interface
- 📈 Historical race analysis & visualization tools
- 🔌 RESTful API for external integrations
- 💾 PostgreSQL database with optimized queries
- 📡 Real-time data updates from Ergast & FastF1 APIs
""")

st.markdown("---")

# Path to screenshots
img_path = Path(r"C:\Users\tarun\diss\td188\diss pictures")

# Get all images in the folder
image_files = list(img_path.glob("*.png")) + list(img_path.glob("*.jpg")) + list(img_path.glob("*.jpeg"))

if image_files:
    st.subheader("📷 Project Screenshots")
    
    st.markdown("---")
    
    # Display each image
    for idx, img_file in enumerate(sorted(image_files)):
        try:
            # Use PIL to open and display the image
            from PIL import Image
            img = Image.open(img_file)
            st.image(img, use_column_width=True)
        except Exception as e:
            st.error(f"Could not load image: {img_file.name}")
            st.error(f"Error details: {str(e)}")
        
        if idx < len(image_files) - 1:
            st.markdown("---")
else:
    st.warning(f"No images found in: {img_path}")
    st.info("Please add screenshots to the folder to display them here.")

st.markdown("---")

# Tech stack
st.subheader("🛠️ Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **Backend & Framework:**
    - Django 4.2+
    - Django REST Framework
    - PostgreSQL / SQLite
    """)

with col2:
    st.markdown("""
    **Machine Learning:**
    - Scikit-learn
    - XGBoost
    - CatBoost
    - Pandas & NumPy
    - Feature Engineering Pipeline
    """)

with col3:
    st.markdown("""
    **Data Sources:**
    - Ergast API
    - FastF1 Library
    - Official F1 Data
    """)

st.markdown("---")

# Footer
st.info("""
💡 **About This Demo**  
This Streamlit app showcases the prediction engine from the full Django system. 
The complete Django application includes data management, user authentication, API endpoints, 
and comprehensive admin tools for model training and evaluation.
""")

st.markdown("---")
st.markdown("**Dissertation Project** • *ML-Based F1 Race Outcome Prediction System*")