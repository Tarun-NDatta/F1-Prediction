# Live F1 Prediction System

This system integrates with the OpenF1 API to provide real-time F1 race predictions using machine learning models. It continuously updates predictions throughout a race until 15 laps to go, when it makes the final prediction.

## Features

- **Real-time Data Integration**: Connects to OpenF1 API for live race data
- **Multiple ML Models**: Uses Ridge Regression, XGBoost, and CatBoost ensemble
- **Live Feature Engineering**: Extracts features from live race data
- **Continuous Updates**: Updates predictions every 30 seconds during races
- **Final Prediction**: Makes definitive prediction with 15 laps remaining
- **Database Integration**: Saves all predictions to Django database
- **Comprehensive Logging**: Detailed logging for monitoring and debugging

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   OpenF1 API    │───▶│ Live Prediction │───▶│ Django Database │
│                 │    │    System       │    │                 │
│ • Lap Times     │    │                 │    │ • Ridge Results │
│ • Positions     │    │ • Feature Ext.  │    │ • XGBoost Results│
│ • Weather       │    │ • ML Models     │    │ • CatBoost Results│
│ • Tire Data     │    │ • Predictions   │    │ • Live Updates  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

1. **Django Project**: Must be running in the td188 directory
2. **ML Models**: Trained models must be available in the `models/` directory
3. **Database**: All required models must be migrated
4. **Python Dependencies**:
   ```bash
   pip install aiohttp asyncio pandas numpy scikit-learn
   ```

## Installation

1. **Ensure all files are in place**:
   - `live_prediction_system.py` (main system)
   - `live_prediction_config.py` (configuration)
   - `prediction/management/commands/run_live_predictions.py` (Django command)

2. **Verify ML models exist**:
   ```bash
   ls models/
   # Should see files like:
   # f1_v20250723_222045_ridge.pkl
   # f1_v20250723_222045_xgboost.pkl
   # f1_v20250723_222045_stacked_model.pkl
   # f1_v20250723_222045_preprocessor.pkl
   # f1_v20250723_222045_features.pkl
   ```

3. **Run database migrations**:
   ```bash
   python manage.py migrate
   ```

## Usage

### Basic Usage

Run the live prediction system using the Django management command:

```bash
cd td188
python manage.py run_live_predictions
```

### Advanced Usage

```bash
# Custom prediction interval (60 seconds)
python manage.py run_live_predictions --interval 60

# Custom final prediction lap (10 laps to go)
python manage.py run_live_predictions --final-lap 10

# Dry run mode (no database saves)
python manage.py run_live_predictions --dry-run

# Debug logging
python manage.py run_live_predictions --log-level DEBUG

# Specific event ID
python manage.py run_live_predictions --event-id 123
```

### Direct Python Execution

You can also run the system directly:

```bash
cd td188
python live_prediction_system.py
```

## Configuration

The system is configured through `live_prediction_config.py`. Key settings:

### Prediction Settings
```python
PREDICTION_CONFIG = {
    'prediction_interval': 30,  # seconds between predictions
    'final_prediction_lap': 15,  # make final prediction with 15 laps to go
    'max_prediction_laps': 50,  # maximum laps to run predictions for
    'min_drivers_required': 15,  # minimum drivers needed for prediction
}
```

### Model Settings
```python
MODEL_CONFIG = {
    'models_directory': 'models',
    'ensemble_weights': {
        'ridge': 0.2,
        'xgboost': 0.3,
        'catboost': 0.5,
    },
    'confidence_threshold': 0.7,
}
```

## How It Works

### 1. Initialization
- Loads all trained ML models from the `models/` directory
- Connects to OpenF1 API
- Identifies current race event from database

### 2. Live Data Collection
The system continuously collects:
- **Driver Positions**: Current race positions
- **Lap Times**: Latest lap times and sector times
- **Weather Data**: Temperature, humidity, rain conditions
- **Tire Data**: Tire compound information

### 3. Feature Engineering
For each driver, extracts features including:
- Current race position
- Recent lap times
- Historical performance data
- Team performance metrics
- Weather conditions
- Track characteristics

### 4. Prediction Generation
Uses three ML models:
1. **Ridge Regression**: Linear model for baseline predictions
2. **XGBoost**: Gradient boosting for complex patterns
3. **CatBoost Ensemble**: Final ensemble combining all models

### 5. Database Storage
Saves predictions to three tables:
- `ridgeregression` (with `model_name='ridge_regression_live'`)
- `xgboostprediction`
- `CatBoostPrediction` (with `model_name='catboost_ensemble_live'`)

### 6. Final Prediction
When `total_laps - current_lap <= 15`, the system:
- Makes the final prediction
- Saves it with `used_live_data=True`
- Exits gracefully

## Data Flow

```
OpenF1 API → Live Data Collection → Feature Engineering → ML Models → Database
     ↓              ↓                    ↓              ↓           ↓
  Positions    Weather Data      Driver Features   Predictions   Storage
  Lap Times    Tire Data        Team Features     Confidence    Logging
  Weather      Sector Times     Track Features    Final Flag    Monitoring
```

## Monitoring and Logging

### Log Files
- `live_prediction.log`: Main log file with all system activity
- Console output: Real-time status updates

### Log Levels
- **INFO**: Normal operation (default)
- **DEBUG**: Detailed debugging information
- **WARNING**: Non-critical issues
- **ERROR**: Critical errors

### Key Log Messages
```
INFO - Starting Live Prediction System
INFO - Current event: Dutch Grand Prix (2025-08-24)
INFO - Active session found: 12345
INFO - Processing lap 5/50
INFO - Made Ridge Regression predictions
INFO - Made XGBoost predictions
INFO - Made CatBoost ensemble predictions
INFO - Saved live predictions to database (Lap 5)
INFO - Making FINAL prediction with 15 laps to go
INFO - Final prediction completed. Exiting live prediction system.
```

## Error Handling

The system includes comprehensive error handling:

1. **API Failures**: Retries with exponential backoff
2. **Model Loading**: Graceful degradation if models missing
3. **Database Errors**: Transaction rollback and retry
4. **Network Issues**: Continues with cached data when possible

## Performance Considerations

### Optimization Tips
1. **Prediction Interval**: 30 seconds is optimal for most races
2. **Database Batching**: Predictions are saved in batches
3. **Memory Management**: Models are loaded once at startup
4. **Network Efficiency**: Uses async HTTP requests

### Resource Requirements
- **CPU**: Moderate (ML inference every 30 seconds)
- **Memory**: ~500MB (loaded models + live data)
- **Network**: ~1MB/minute (OpenF1 API calls)
- **Storage**: ~10MB/day (log files + database)

## Troubleshooting

### Common Issues

1. **"No model files found"**
   ```bash
   # Check if models exist
   ls models/*.pkl
   # Ensure latest timestamp models are present
   ```

2. **"No current race event found"**
   ```bash
   # Check database for events
   python manage.py shell
   >>> from data.models import Event
   >>> Event.objects.filter(year=2025).order_by('date')
   ```

3. **"No active session found"**
   - Verify OpenF1 API is accessible
   - Check if there's an active F1 session
   - Try with `--dry-run` for testing

4. **Database connection errors**
   ```bash
   # Check Django settings
   python manage.py check
   # Verify database migrations
   python manage.py showmigrations
   ```

### Debug Mode
```bash
python manage.py run_live_predictions --log-level DEBUG --dry-run
```

## Integration with Web Interface

The live predictions are automatically available in the web interface:

1. **Live Updates Page**: Shows real-time predictions
2. **Prediction Page**: Displays final predictions
3. **Betting Interface**: Uses live predictions for odds calculation

## Development

### Adding New Features
1. Modify `live_prediction_system.py`
2. Update configuration in `live_prediction_config.py`
3. Test with `--dry-run` flag
4. Update this README

### Testing
```bash
# Test configuration
python live_prediction_config.py

# Test with mock data
python manage.py run_live_predictions --dry-run --log-level DEBUG
```

## API Reference

### OpenF1Client Methods
- `get_current_session()`: Get active F1 session
- `get_lap_times(session_id)`: Get lap times
- `get_driver_positions(session_id)`: Get current positions
- `get_weather_data(session_id)`: Get weather data
- `get_tire_data(session_id)`: Get tire compound data

### LivePredictionSystem Methods
- `run_live_prediction()`: Main execution method
- `_extract_live_features()`: Feature engineering
- `_make_predictions()`: ML model inference
- `_save_predictions_to_db()`: Database storage

## Support

For issues or questions:
1. Check the log files for error messages
2. Verify configuration settings
3. Test with `--dry-run` mode
4. Check OpenF1 API status

## License

This system is part of the F1 Prediction Dashboard project. 