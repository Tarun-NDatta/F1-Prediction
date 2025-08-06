"""
Configuration file for Live F1 Prediction System
"""

import os
from typing import Dict, Any

# OpenF1 API Configuration
OPENF1_CONFIG = {
    'base_url': 'https://api.openf1.org/v1',
    'timeout': 30,  # seconds
    'max_retries': 3,
    'retry_delay': 5,  # seconds
}

# Prediction System Configuration
PREDICTION_CONFIG = {
    'prediction_interval': 30,  # seconds between predictions
    'final_prediction_lap': 15,  # make final prediction with 15 laps to go
    'max_prediction_laps': 50,  # maximum laps to run predictions for
    'min_drivers_required': 15,  # minimum drivers needed for prediction
}

# Model Configuration
MODEL_CONFIG = {
    'models_directory': 'models',
    'latest_timestamp_pattern': r'f1_v(\d+)_',
    'required_models': ['ridge', 'xgboost', 'catboost', 'preprocessor'],
    'ensemble_weights': {
        'ridge': 0.2,
        'xgboost': 0.3,
        'catboost': 0.5,
    },
    'confidence_threshold': 0.7,
}

# Database Configuration
DATABASE_CONFIG = {
    'batch_size': 100,  # number of predictions to save in batch
    'transaction_timeout': 30,  # seconds
    'max_retries': 3,
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'include_weather': True,
    'include_tire_data': True,
    'include_lap_times': True,
    'include_sector_times': True,
    'include_historical_performance': True,
    'include_team_performance': True,
    'include_track_specialization': True,
    'include_driver_specialization': True,
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'live_prediction.log',
    'max_file_size': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# Weather Impact Configuration
WEATHER_CONFIG = {
    'rain_threshold': 0.1,  # mm/h
    'temperature_impact_threshold': 5.0,  # degrees Celsius
    'humidity_impact_threshold': 20.0,  # percentage
    'wind_speed_impact_threshold': 10.0,  # km/h
}

# Tire Strategy Configuration
TIRE_CONFIG = {
    'compounds': ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET'],
    'degradation_rates': {
        'SOFT': 1.5,
        'MEDIUM': 1.0,
        'HARD': 0.7,
        'INTERMEDIATE': 1.2,
        'WET': 1.8,
    },
    'performance_impact': {
        'SOFT': 1.0,
        'MEDIUM': 0.95,
        'HARD': 0.9,
        'INTERMEDIATE': 0.85,
        'WET': 0.8,
    },
}

# Track Specialization Configuration
TRACK_CONFIG = {
    'categories': {
        'POWER': {
            'description': 'Power Circuits',
            'characteristics': ['high_speed', 'long_straights', 'engine_importance'],
        },
        'TECHNICAL': {
            'description': 'Technical Circuits',
            'characteristics': ['slow_corners', 'aero_importance', 'precision'],
        },
        'STREET': {
            'description': 'Street Circuits',
            'characteristics': ['narrow_track', 'low_speed', 'precision'],
        },
        'HYBRID': {
            'description': 'Hybrid Circuits',
            'characteristics': ['mixed_characteristics', 'balanced'],
        },
        'HIGH_SPEED': {
            'description': 'High Speed Circuits',
            'characteristics': ['very_high_speed', 'aero_importance'],
        },
    },
}

# Driver Specialization Configuration
DRIVER_SPEC_CONFIG = {
    'specializations': {
        'OVERTAKING': {
            'description': 'Overtaking Specialist',
            'track_modifiers': {'overtaking_difficulty': -0.2},
        },
        'QUALIFYING': {
            'description': 'Qualifying Specialist',
            'track_modifiers': {'qualifying_importance': 0.2},
        },
        'CONSISTENCY': {
            'description': 'Consistency Specialist',
            'track_modifiers': {'position_variance': -0.2},
        },
        'WET_WEATHER': {
            'description': 'Wet Weather Specialist',
            'weather_modifiers': {'rain': 0.3},
        },
        'TIRE_MANAGEMENT': {
            'description': 'Tire Management Specialist',
            'tire_modifiers': {'degradation_rate': -0.2},
        },
        'TECHNICAL': {
            'description': 'Technical Circuit Specialist',
            'track_modifiers': {'technical_circuit_modifier': 0.2},
        },
        'POWER': {
            'description': 'Power Circuit Specialist',
            'track_modifiers': {'power_circuit_modifier': 0.2},
        },
    },
}

# Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'min_lap_time': 60.0,  # seconds
    'max_lap_time': 300.0,  # seconds
    'min_sector_time': 15.0,  # seconds
    'max_sector_time': 100.0,  # seconds
    'min_position': 1,
    'max_position': 20,
    'min_confidence': 0.1,
    'max_confidence': 1.0,
}

# Error Handling Configuration
ERROR_CONFIG = {
    'max_consecutive_errors': 5,
    'error_cooldown': 60,  # seconds
    'graceful_degradation': True,
    'fallback_predictions': True,
}

# Development/Testing Configuration
DEV_CONFIG = {
    'mock_data': False,
    'mock_session_id': 12345,
    'mock_event_id': 1,
    'test_mode': False,
    'verbose_logging': False,
}

def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        'openf1': OPENF1_CONFIG,
        'prediction': PREDICTION_CONFIG,
        'model': MODEL_CONFIG,
        'database': DATABASE_CONFIG,
        'feature': FEATURE_CONFIG,
        'logging': LOGGING_CONFIG,
        'weather': WEATHER_CONFIG,
        'tire': TIRE_CONFIG,
        'track': TRACK_CONFIG,
        'driver_spec': DRIVER_SPEC_CONFIG,
        'performance': PERFORMANCE_THRESHOLDS,
        'error': ERROR_CONFIG,
        'dev': DEV_CONFIG,
    }

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration values"""
    try:
        # Validate prediction interval
        if config['prediction']['prediction_interval'] < 5:
            raise ValueError("Prediction interval must be at least 5 seconds")
        
        # Validate final prediction lap
        if config['prediction']['final_prediction_lap'] < 1:
            raise ValueError("Final prediction lap must be at least 1")
        
        # Validate model weights sum to 1
        weights_sum = sum(config['model']['ensemble_weights'].values())
        if abs(weights_sum - 1.0) > 0.01:
            raise ValueError("Model ensemble weights must sum to 1.0")
        
        # Validate performance thresholds
        if config['performance']['min_lap_time'] >= config['performance']['max_lap_time']:
            raise ValueError("Min lap time must be less than max lap time")
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

if __name__ == "__main__":
    # Test configuration
    config = get_config()
    if validate_config(config):
        print("Configuration is valid!")
    else:
        print("Configuration validation failed!") 