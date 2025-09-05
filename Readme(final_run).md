F1 Race Position Prediction System

AI/ML Approach to Formula 1 Race Outcome Prediction Using Historical and Real-Time Data
Project Overview
This Django-based system implements machine learning models to predict Formula 1 race finishing positions using a combination of historical race data and real-time race information. The project employs ensemble methods combining Ridge Regression, XGBoost, and CatBoost algorithms to achieve optimal prediction accuracy.
Research Objective
To evaluate the effectiveness of different machine learning approaches in predicting F1 race positions, with particular focus on:

Historical performance patterns and driver/team characteristics
Real-time race data integration via API calls
Chaos theory application to race unpredictability analysis
Ensemble model performance comparison

Technical Architecture
Core Technologies

Framework: Django 4.x
Database: SQLite/PostgreSQL (configurable)
ML Libraries:

scikit-learn (Ridge Regression, preprocessing)
XGBoost (Gradient boosting)
CatBoost (Categorical boosting)
FastF1 (Historical F1 data)


Real-time Data: HypRace API via RapidAPI
Optimization: Bayesian optimization (scikit-optimize)
Analysis: matplotlib, pandas, numpy

Data Sources

Historical Data: FastF1 library (2022-2025 seasons)
Real-time Data: HypRace API for live race telemetry
Feature Engineering: Driver performance metrics, team characteristics, circuit affinity

Installation & Setup
Prerequisites
bashPython 3.8+
pip
Git
Environment Setup
bash# Clone the repository
git clone <repository-url>
cd td188

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Django setup
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser
Configuration

Update ROOT_DIR path in training commands to match your project directory
Configure RapidAPI key for HypRace API access
Set up FastF1 cache directory for historical data

Data Collection & Preparation
Historical Data Collection
bash# Fetch qualifying results (2022-2025)
python manage.py qualifying --years 2022 2023 2024 2025

# Fetch race results and pit stop data
python manage.py results --years 2022 2023 2024

# Generate engineered features
python manage.py extract_features --date 2025-12-31
Feature Engineering
The system creates enhanced features including:

Driver Features: Reliability score, qualifying average, position variance, circuit affinity
Team Features: Development slope, pit stop performance, reliability metrics
Temporal Features: Performance momentum, seasonal trends with decay weighting

Model Training
Baseline Ridge Regression
bashpython manage.py train_ridge --save-model
XGBoost Ensemble with Bayesian Optimization
bashpython manage.py train_ensemble_model

Implements Ridge + XGBoost stacking ensemble
Bayesian hyperparameter optimization (40 iterations)
Sample weighting with temporal decay (α = 0.05)
Saves models to project root directory

CatBoost with Track Specialization
bash# Initialize track specialization data
python manage.py catboost_pipeline --mode initialize

# Train CatBoost model
python manage.py predict_catboost --mode train --save-model
Prediction System
Offline Predictions
bash# Ridge baseline predictions
python manage.py predict_baseline --model ridge_baseline.pkl --year 2025 --round 10 --compare

# XGBoost ensemble predictions
python manage.py predict_xgboost --year 2025 --round 10 --compare 

# CatBoost predictions with track specialization
python manage.py catboost_pipeline --mode predict --year 2025 --round 10 --compare
Real-Time Live Predictions
bash# Live prediction system using HypRace API
python manage.py run_live_predictions --interval 10 --final-lap 15 --grand-prix-id <id>

# Test API connection
python manage.py run_live_predictions --test-connection

# Check API quota usage
python manage.py run_live_predictions --quota-status
Analysis & Evaluation
Chaos Theory Analysis
bash# Analyze prediction errors and chaos impact
python manage.py analyze_chaos_impact --season 2025 --model catboost_ensemble

# Regression analysis: chaos score vs model MAE
python manage.py analyze_chaos --year 2025 --model catboost_ensemble
Model Performance Metrics

RMSE: Root Mean Square Error
MAE: Mean Absolute Error
R²: Coefficient of determination
Spearman Correlation: Rank correlation for position predictions

Web Interface
Development Server
bashpython manage.py runserver
Available Endpoints

Admin interface: /admin/
Prediction results dashboard
Real-time race monitoring
Model performance visualization

Key Features
Advanced ML Pipeline

Ensemble Learning: Combines Ridge, XGBoost, and CatBoost
Bayesian Optimization: Automated hyperparameter tuning
Temporal Weighting: Recent seasons weighted more heavily
Feature Engineering: 15+ engineered features per driver/team

Real-Time Integration

Live Data Streaming: HypRace API integration
Dynamic Predictions: Updates during race progression
API Quota Management: Efficient usage tracking

Research Analysis

Chaos Theory: Quantifies race unpredictability
Counterfactual Analysis: "What-if" scenario modeling
Performance Benchmarking: Cross-model comparison

API Dependencies
External Services

RapidAPI/HypRace: Live F1 telemetry and timing data
FastF1: Historical race data and session information

Rate Limits

HypRace API: 40 requests/month (configurable)
FastF1: No rate limits (cached locally)
There is a strong chance that i will not be purchasing the api calls leaving the project with one maybe 2 calls left(will add a video of it working in presentation)

Research Applications
This system supports analysis of:

Machine learning model effectiveness in motorsport prediction
Impact of chaos theory on prediction accuracy
Real-time data integration benefits
Ensemble method performance in time-series prediction
Feature importance in F1 race outcome prediction

