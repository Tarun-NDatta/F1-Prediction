"""
Live Prediction System for F1 Races
Integrates with OpenF1 API and continuously updates predictions until 15 laps to go
"""

import os
import sys
import django
import asyncio
import aiohttp
import json
import logging
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import random
from dataclasses import dataclass
from enum import Enum
warnings.filterwarnings('ignore')

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Dissertation.settings')
django.setup()

from django.db import transaction
from django.utils import timezone
from data.models import (
    Event, Driver, Team, Circuit, Session, RaceResult, QualifyingResult,
    DriverPerformance, TeamPerformance, TrackCharacteristics,
    ridgeregression, xgboostprediction, CatBoostPrediction,
    TrackSpecialization, DriverSpecialization
)
from prediction.data_prep.utilities import load_model
from prediction.data_prep.pipeline import F1DataPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Mock Race Simulation Classes
class TrackType(Enum):
    POWER = "POWER"
    TECHNICAL = "TECHNICAL"
    STREET = "STREET"
    HYBRID = "HYBRID"
    HIGH_SPEED = "HIGH_SPEED"

class EventType(Enum):
    SAFETY_CAR = "safety_car"
    BAD_PIT_STOP = "bad_pit_stop"
    WEATHER_CHANGE = "weather_change"

@dataclass
class RaceEvent:
    lap: int
    event_type: EventType
    driver_affected: Optional[str] = None
    description: str = ""
    impact_duration: int = 1  # laps

@dataclass
class DriverState:
    name: str
    position: int
    lap_time: float
    tire_compound: str
    tire_age: int
    pit_stops: int
    penalties: int = 0
    dnf: bool = False

class TrackConfiguration:
    """Track-specific configuration for realistic simulation"""

    TRACK_CONFIGS = {
        "Australian Grand Prix": {
            "total_laps": 58,
            "track_type": TrackType.HYBRID,
            "base_lap_time": 82.5,
            "overtaking_difficulty": 6,
            "tire_degradation": 5,
            "safety_car_probability": 0.3,
            "weather_change_probability": 0.2
        },
        "Chinese Grand Prix": {
            "total_laps": 56,
            "track_type": TrackType.TECHNICAL,
            "base_lap_time": 95.2,
            "overtaking_difficulty": 4,
            "tire_degradation": 6,
            "safety_car_probability": 0.25,
            "weather_change_probability": 0.4
        },
        "Dutch Grand Prix": {
            "total_laps": 72,
            "track_type": TrackType.TECHNICAL,
            "base_lap_time": 75.8,
            "overtaking_difficulty": 7,
            "tire_degradation": 4,
            "safety_car_probability": 0.2,
            "weather_change_probability": 0.3
        },
        "Italian Grand Prix": {  # Monza
            "total_laps": 53,
            "track_type": TrackType.POWER,
            "base_lap_time": 81.2,
            "overtaking_difficulty": 3,
            "tire_degradation": 3,
            "safety_car_probability": 0.15,
            "weather_change_probability": 0.1
        },
        "Azerbaijan Grand Prix": {  # Baku
            "total_laps": 51,
            "track_type": TrackType.STREET,
            "base_lap_time": 103.8,
            "overtaking_difficulty": 5,
            "tire_degradation": 7,
            "safety_car_probability": 0.6,
            "weather_change_probability": 0.1
        },
        "Singapore Grand Prix": {
            "total_laps": 61,
            "track_type": TrackType.STREET,
            "base_lap_time": 103.2,
            "overtaking_difficulty": 8,
            "tire_degradation": 8,
            "safety_car_probability": 0.4,
            "weather_change_probability": 0.7
        }
    }

    @classmethod
    def get_config(cls, event_name: str) -> Dict:
        return cls.TRACK_CONFIGS.get(event_name, cls.TRACK_CONFIGS["Dutch Grand Prix"])

class F1Commentary:
    """F1-style race commentary generator"""

    def __init__(self, event_name: str, track_config: Dict):
        self.event_name = event_name
        self.track_config = track_config
        self.commentary_history = []

        # F1 commentator phrases
        self.race_start_phrases = [
            "And it's lights out and away we go!",
            "The formation lap is complete, and we're ready for racing!",
            "Green flag! The {event} is underway!"
        ]

        self.overtake_phrases = [
            "Brilliant move by {overtaker} on {overtaken}!",
            "{overtaker} sweeps past {overtaken} with a fantastic overtake!",
            "What a move! {overtaker} takes the position from {overtaken}!",
            "{overtaker} makes it stick on {overtaken} - superb racing!"
        ]

        self.pit_stop_phrases = [
            "{driver} comes into the pits for a {time:.1f} second stop",
            "That's a {time:.1f} second pit stop for {driver}",
            "{driver} in and out in {time:.1f} seconds - {'excellent' if time < 3.0 else 'steady'} work from the crew"
        ]

        self.weather_phrases = [
            "The weather is changing here at {track}!",
            "Rain is starting to fall - this could shake up the order!",
            "The track is drying out, and lap times are dropping",
            "Conditions are tricky out there with the changing weather"
        ]

        self.safety_car_phrases = [
            "Safety Car deployed! The field bunches up",
            "Yellow flags are out - Safety Car period begins",
            "We have a Safety Car situation - this could change everything!"
        ]

        self.penalty_phrases = [
            "{driver} has been handed a {penalty} second time penalty",
            "The stewards have given {driver} a {penalty} second penalty",
            "That's a {penalty} second penalty for {driver} - costly mistake!"
        ]

    def generate_race_start(self) -> str:
        phrase = random.choice(self.race_start_phrases)
        return phrase.format(event=self.event_name)

    def generate_overtake(self, overtaker: str, overtaken: str, lap: int) -> str:
        phrase = random.choice(self.overtake_phrases)
        commentary = f"Lap {lap}: {phrase.format(overtaker=overtaker, overtaken=overtaken)}"
        self.commentary_history.append(commentary)
        return commentary

    def generate_pit_stop(self, driver: str, lap: int, stop_time: float) -> str:
        phrase = random.choice(self.pit_stop_phrases)
        commentary = f"Lap {lap}: {phrase.format(driver=driver, time=stop_time)}"
        self.commentary_history.append(commentary)
        return commentary

    def generate_weather_change(self, lap: int, weather_type: str) -> str:
        phrase = random.choice(self.weather_phrases)
        track_name = self.event_name.replace(" Grand Prix", "")
        commentary = f"Lap {lap}: {phrase.format(track=track_name)} - {weather_type}"
        self.commentary_history.append(commentary)
        return commentary

    def generate_safety_car(self, lap: int, reason: str = "") -> str:
        phrase = random.choice(self.safety_car_phrases)
        commentary = f"Lap {lap}: {phrase}"
        if reason:
            commentary += f" - {reason}"
        self.commentary_history.append(commentary)
        return commentary

    def generate_penalty(self, driver: str, lap: int, penalty_seconds: int) -> str:
        phrase = random.choice(self.penalty_phrases)
        commentary = f"Lap {lap}: {phrase.format(driver=driver, penalty=penalty_seconds)}"
        self.commentary_history.append(commentary)
        return commentary

    def generate_position_update(self, lap: int, leader: str, gap: float) -> str:
        commentary = f"Lap {lap}: {leader} leads by {gap:.1f} seconds"
        return commentary

    def generate_final_laps(self, lap: int, total_laps: int, leader: str) -> str:
        laps_remaining = total_laps - lap
        if laps_remaining == 10:
            return f"Lap {lap}: Just 10 laps remaining! {leader} still leads"
        elif laps_remaining == 5:
            return f"Lap {lap}: 5 laps to go - the tension is building!"
        elif laps_remaining == 1:
            return f"Lap {lap}: Final lap! {leader} is moments away from victory!"
        return ""

class MockRaceSimulator:
    """Core race simulation engine"""

    def __init__(self, event_name: str, simulation_speed: str = "10min"):
        self.event_name = event_name
        self.simulation_speed = simulation_speed
        self.track_config = TrackConfiguration.get_config(event_name)
        self.total_laps = self.track_config["total_laps"]
        self.current_lap = 0
        self.race_started = False
        self.race_finished = False
        self.ml_updates_stopped = False

        # Commentary system
        self.commentary = F1Commentary(event_name, self.track_config)

        # ML Prediction system
        self.ml_manager = MockMLPredictionManager()
        self.ml_initialized = False
        self.last_ml_update_lap = 0
        self.ml_update_interval = 2  # Update every 2 laps (simulating 30-second intervals)

        # User events (max 2 per race)
        self.user_events_used = 0
        self.max_user_events = 2
        self.pending_user_events = []

        # Race state
        self.drivers = self._initialize_drivers()
        self.weather_condition = "DRY"
        self.safety_car_active = False
        self.safety_car_laps_remaining = 0

        # Timing configuration
        self.lap_interval = self._calculate_lap_interval()

        # Race events log
        self.race_events = []
        self.commentary_feed = []

        # Initialize ML system
        self._initialize_ml_system()

    def _initialize_ml_system(self):
        """Initialize the ML prediction system"""
        try:
            self.ml_initialized = self.ml_manager.initialize_pipelines(self.event_name)
            if self.ml_initialized:
                logger.info(f"ML system initialized for {self.event_name}")
            else:
                logger.warning(f"ML system failed to initialize, using mock predictions")
        except Exception as e:
            logger.error(f"Error initializing ML system: {e}")
            self.ml_initialized = False

    def _should_update_ml_predictions(self) -> bool:
        """Check if ML predictions should be updated this lap"""
        if self.ml_updates_stopped or not self.race_started:
            return False

        # Update every N laps (simulating 30-second intervals)
        if (self.current_lap - self.last_ml_update_lap) >= self.ml_update_interval:
            return True

        return False

    def _update_ml_predictions(self) -> Dict:
        """Update ML predictions based on current race state"""
        try:
            if not self._should_update_ml_predictions():
                return {}

            # Get current race state
            race_state = self.get_race_state()

            # Generate predictions
            predictions = self.ml_manager.generate_predictions(race_state, self.current_lap)

            # Save to database (optional, can be disabled for demo)
            try:
                self.ml_manager.save_predictions_to_database(predictions, self.current_lap)
            except Exception as e:
                logger.warning(f"Could not save predictions to database: {e}")

            # Update tracking
            self.last_ml_update_lap = self.current_lap

            # Add commentary about ML update
            self.commentary_feed.append({
                "lap": self.current_lap,
                "message": f"Lap {self.current_lap}: ML models updated - {len(predictions)} prediction sets generated",
                "timestamp": datetime.now().isoformat(),
                "type": "ml_update"
            })

            return predictions

        except Exception as e:
            logger.error(f"Error updating ML predictions: {e}")
            return {}

    def _calculate_lap_interval(self) -> float:
        """Calculate time between laps based on simulation speed"""
        speed_mapping = {
            "5min": 300 / self.total_laps,   # 5 minutes total
            "10min": 600 / self.total_laps,  # 10 minutes total
            "15min": 900 / self.total_laps   # 15 minutes total
        }
        return speed_mapping.get(self.simulation_speed, speed_mapping["10min"])

    def _initialize_drivers(self) -> List[DriverState]:
        """Initialize driver grid based on typical F1 field"""
        # Realistic F1 2025 driver lineup
        driver_names = [
            "Max Verstappen", "Lando Norris", "Charles Leclerc", "Oscar Piastri",
            "Carlos Sainz", "Lewis Hamilton", "George Russell", "Fernando Alonso",
            "Lance Stroll", "Nico Hulkenberg", "Kevin Magnussen", "Yuki Tsunoda",
            "Daniel Ricciardo", "Pierre Gasly", "Esteban Ocon", "Alexander Albon",
            "Logan Sargeant", "Valtteri Bottas", "Zhou Guanyu", "Nyck de Vries"
        ]

        drivers = []
        base_lap_time = self.track_config["base_lap_time"]

        for i, name in enumerate(driver_names):
            # Add realistic performance variation
            performance_modifier = 0.2 + (i * 0.05)  # Slower drivers have higher lap times
            lap_time = base_lap_time + performance_modifier + random.uniform(-0.1, 0.1)

            drivers.append(DriverState(
                name=name,
                position=i + 1,  # Starting positions 1-20
                lap_time=lap_time,
                tire_compound="MEDIUM",
                tire_age=0,
                pit_stops=0
            ))

        return drivers

    def can_add_user_event(self) -> bool:
        """Check if user can add more events"""
        return self.user_events_used < self.max_user_events and self.current_lap < (self.total_laps - 10)

    def add_user_event(self, event_type: EventType, target_lap: int, driver_name: str = None) -> bool:
        """Add a user-triggered event"""
        if not self.can_add_user_event():
            return False

        if target_lap <= self.current_lap or target_lap > (self.total_laps - 10):
            return False

        event = RaceEvent(
            lap=target_lap,
            event_type=event_type,
            driver_affected=driver_name,
            description=f"User-triggered {event_type.value}"
        )

        self.pending_user_events.append(event)
        self.user_events_used += 1
        return True

    def _process_lap_events(self) -> List[str]:
        """Process events that occur during this lap"""
        lap_commentary = []

        # Check for user events
        user_events = [e for e in self.pending_user_events if e.lap == self.current_lap]
        for event in user_events:
            commentary = self._execute_event(event)
            if commentary:
                lap_commentary.append(commentary)
            self.pending_user_events.remove(event)

        # Random events based on track characteristics
        if not user_events:  # Don't add random events on same lap as user events
            random_event = self._generate_random_event()
            if random_event:
                lap_commentary.append(random_event)

        # Position changes and overtakes
        if self.current_lap > 1 and not self.safety_car_active:
            overtake_commentary = self._simulate_overtakes()
            lap_commentary.extend(overtake_commentary)

        # Pit stops
        pit_commentary = self._simulate_pit_stops()
        lap_commentary.extend(pit_commentary)

        # Regular position updates
        if self.current_lap % 5 == 0:  # Every 5 laps
            leader = self.drivers[0].name
            gap = random.uniform(0.5, 3.0)
            position_update = self.commentary.generate_position_update(self.current_lap, leader, gap)
            lap_commentary.append(position_update)

        # Final laps commentary
        final_lap_commentary = self.commentary.generate_final_laps(
            self.current_lap, self.total_laps, self.drivers[0].name
        )
        if final_lap_commentary:
            lap_commentary.append(final_lap_commentary)

        return lap_commentary

    def _execute_event(self, event: RaceEvent) -> str:
        """Execute a specific race event"""
        if event.event_type == EventType.SAFETY_CAR:
            self.safety_car_active = True
            self.safety_car_laps_remaining = random.randint(3, 6)
            return self.commentary.generate_safety_car(self.current_lap, "Incident on track")

        elif event.event_type == EventType.BAD_PIT_STOP:
            if event.driver_affected:
                driver = next((d for d in self.drivers if d.name == event.driver_affected), None)
                if driver:
                    stop_time = random.uniform(6.0, 12.0)  # Bad stop
                    driver.pit_stops += 1
                    driver.tire_age = 0
                    # Move driver down positions due to slow stop
                    self._adjust_driver_position(driver, -random.randint(3, 8))
                    return self.commentary.generate_pit_stop(driver.name, self.current_lap, stop_time)

        elif event.event_type == EventType.WEATHER_CHANGE:
            if self.weather_condition == "DRY":
                self.weather_condition = "LIGHT_RAIN"
                weather_desc = "Light rain beginning to fall"
            elif self.weather_condition == "LIGHT_RAIN":
                self.weather_condition = "HEAVY_RAIN"
                weather_desc = "Rain intensifying - heavy downpour"
            else:
                self.weather_condition = "DRY"
                weather_desc = "Rain stopping, track beginning to dry"

            # Weather affects lap times
            self._apply_weather_effects()
            return self.commentary.generate_weather_change(self.current_lap, weather_desc)

        return ""

    def _generate_random_event(self) -> Optional[str]:
        """Generate random events based on track characteristics"""
        # Safety car probability
        if random.random() < (self.track_config["safety_car_probability"] / self.total_laps):
            if not self.safety_car_active:
                self.safety_car_active = True
                self.safety_car_laps_remaining = random.randint(3, 6)
                return self.commentary.generate_safety_car(self.current_lap)

        # Weather change probability
        if random.random() < (self.track_config["weather_change_probability"] / self.total_laps):
            return self._execute_event(RaceEvent(
                lap=self.current_lap,
                event_type=EventType.WEATHER_CHANGE
            ))

        return None

    def _simulate_overtakes(self) -> List[str]:
        """Simulate realistic overtakes based on track characteristics"""
        overtakes = []
        overtaking_difficulty = self.track_config["overtaking_difficulty"]

        # Lower difficulty = more overtakes
        overtake_probability = max(0.02, 0.15 - (overtaking_difficulty * 0.015))

        for i in range(1, len(self.drivers)):
            if random.random() < overtake_probability:
                # Driver behind overtakes driver ahead
                overtaker = self.drivers[i]
                overtaken = self.drivers[i-1]

                # Swap positions
                self.drivers[i-1], self.drivers[i] = self.drivers[i], self.drivers[i-1]
                overtaker.position, overtaken.position = overtaken.position, overtaker.position

                commentary = self.commentary.generate_overtake(
                    overtaker.name, overtaken.name, self.current_lap
                )
                overtakes.append(commentary)

        return overtakes

    def _simulate_pit_stops(self) -> List[str]:
        """Simulate strategic pit stops"""
        pit_stops = []

        for driver in self.drivers:
            # Pit stop strategy based on tire age and race progress
            should_pit = False

            # Mandatory pit stop around mid-race
            if (self.total_laps * 0.4 <= self.current_lap <= self.total_laps * 0.7 and
                driver.pit_stops == 0 and random.random() < 0.15):
                should_pit = True

            # Emergency pit for old tires
            elif driver.tire_age > 25 and random.random() < 0.3:
                should_pit = True

            if should_pit:
                stop_time = random.uniform(2.3, 3.5)  # Normal pit stop
                driver.pit_stops += 1
                driver.tire_age = 0
                driver.tire_compound = random.choice(["SOFT", "MEDIUM", "HARD"])

                # Pit stop loses positions
                self._adjust_driver_position(driver, -random.randint(2, 5))

                commentary = self.commentary.generate_pit_stop(
                    driver.name, self.current_lap, stop_time
                )
                pit_stops.append(commentary)

        return pit_stops

    def _adjust_driver_position(self, driver: DriverState, position_change: int):
        """Adjust driver position and reorder grid"""
        old_pos = driver.position
        new_pos = max(1, min(20, old_pos + position_change))

        if new_pos != old_pos:
            # Remove driver from current position
            self.drivers.remove(driver)
            driver.position = new_pos

            # Insert at new position
            self.drivers.insert(new_pos - 1, driver)

            # Update all positions
            for i, d in enumerate(self.drivers):
                d.position = i + 1

    def _apply_weather_effects(self):
        """Apply weather effects to lap times and strategy"""
        for driver in self.drivers:
            if self.weather_condition == "LIGHT_RAIN":
                driver.lap_time += random.uniform(0.5, 2.0)
            elif self.weather_condition == "HEAVY_RAIN":
                driver.lap_time += random.uniform(2.0, 5.0)
                # Some drivers are better in wet conditions
                if "Hamilton" in driver.name or "Verstappen" in driver.name:
                    driver.lap_time -= 1.0  # Wet weather specialists
            else:  # DRY
                # Reset to base lap time with small variation
                base_time = self.track_config["base_lap_time"]
                performance_mod = (driver.position - 1) * 0.05
                driver.lap_time = base_time + performance_mod + random.uniform(-0.1, 0.1)

    def _update_tire_age(self):
        """Update tire age for all drivers"""
        for driver in self.drivers:
            if not driver.dnf:
                driver.tire_age += 1

                # Tire degradation affects lap time
                degradation_factor = self.track_config["tire_degradation"] / 10
                driver.lap_time += (driver.tire_age * degradation_factor * 0.01)

    def get_race_state(self) -> Dict:
        """Get current race state for ML predictions"""
        return {
            "current_lap": self.current_lap,
            "total_laps": self.total_laps,
            "weather_condition": self.weather_condition,
            "safety_car_active": self.safety_car_active,
            "driver_positions": [
                {
                    "driver_number": i + 1,
                    "driver_name": driver.name,
                    "position": driver.position,
                    "last_lap_time": driver.lap_time,
                    "tire_compound": driver.tire_compound,
                    "tire_age": driver.tire_age,
                    "pit_stops": driver.pit_stops,
                    "sector_times": [
                        driver.lap_time / 3 + random.uniform(-0.1, 0.1),
                        driver.lap_time / 3 + random.uniform(-0.1, 0.1),
                        driver.lap_time / 3 + random.uniform(-0.1, 0.1)
                    ]
                }
                for i, driver in enumerate(self.drivers)
            ],
            "weather_data": {
                "air_temp": 22.5 + random.uniform(-2, 2),
                "track_temp": 28.0 + random.uniform(-3, 3),
                "humidity": 65.0 + random.uniform(-10, 10),
                "rain": self.weather_condition != "DRY",
                "wind_speed": random.uniform(5, 15),
                "wind_direction": random.randint(0, 360)
            },
            "tire_data": [
                {
                    "driver_number": i + 1,
                    "compound": driver.tire_compound,
                    "age": driver.tire_age,
                    "wear": min(100, driver.tire_age * 3 + random.randint(0, 10))
                }
                for i, driver in enumerate(self.drivers)
            ]
        }

    async def simulate_lap(self) -> Dict:
        """Simulate one lap of the race"""
        if self.race_finished:
            return {"status": "finished"}

        if not self.race_started:
            self.race_started = True
            self.current_lap = 1
            start_commentary = self.commentary.generate_race_start()
            self.commentary_feed.append({
                "lap": 0,
                "message": start_commentary,
                "timestamp": datetime.now().isoformat()
            })
        else:
            self.current_lap += 1

        # Process safety car
        if self.safety_car_active:
            self.safety_car_laps_remaining -= 1
            if self.safety_car_laps_remaining <= 0:
                self.safety_car_active = False
                self.commentary_feed.append({
                    "lap": self.current_lap,
                    "message": f"Lap {self.current_lap}: Safety Car returns to the pits - racing resumes!",
                    "timestamp": datetime.now().isoformat()
                })

        # Process lap events
        lap_commentary = self._process_lap_events()
        for comment in lap_commentary:
            self.commentary_feed.append({
                "lap": self.current_lap,
                "message": comment,
                "timestamp": datetime.now().isoformat()
            })

        # Update tire age and effects
        self._update_tire_age()

        # Update ML predictions
        ml_predictions = {}
        if not self.ml_updates_stopped:
            ml_predictions = self._update_ml_predictions()

        # Check if ML updates should stop (10 laps to go)
        if self.current_lap >= (self.total_laps - 10) and not self.ml_updates_stopped:
            self.ml_updates_stopped = True

            # Generate final prediction
            final_predictions = self._update_ml_predictions()

            self.commentary_feed.append({
                "lap": self.current_lap,
                "message": f"Lap {self.current_lap}: Final ML prediction generated - 10 laps remaining!",
                "timestamp": datetime.now().isoformat(),
                "type": "final_prediction"
            })

        # Check if race is finished
        if self.current_lap >= self.total_laps:
            self.race_finished = True
            winner = self.drivers[0].name
            self.commentary_feed.append({
                "lap": self.current_lap,
                "message": f"CHEQUERED FLAG! {winner} wins the {self.event_name}!",
                "timestamp": datetime.now().isoformat()
            })

        return {
            "status": "running" if not self.race_finished else "finished",
            "current_lap": self.current_lap,
            "total_laps": self.total_laps,
            "ml_updates_active": not self.ml_updates_stopped,
            "ml_predictions": ml_predictions,
            "ml_initialized": self.ml_initialized,
            "race_state": self.get_race_state(),
            "commentary": self.commentary_feed[-5:] if self.commentary_feed else [],  # Last 5 comments
            "can_add_events": self.can_add_user_event(),
            "events_used": self.user_events_used,
            "max_events": self.max_user_events
        }

    def get_final_results(self) -> List[Dict]:
        """Get final race results"""
        return [
            {
                "position": driver.position,
                "driver": driver.name,
                "pit_stops": driver.pit_stops,
                "penalties": driver.penalties,
                "dnf": driver.dnf
            }
            for driver in sorted(self.drivers, key=lambda d: d.position)
        ]

    def get_simulation_summary(self) -> Dict:
        """Get complete simulation summary"""
        return {
            "event_name": self.event_name,
            "simulation_speed": self.simulation_speed,
            "total_laps": self.total_laps,
            "final_results": self.get_final_results(),
            "commentary_feed": self.commentary_feed,
            "user_events_used": self.user_events_used,
            "track_config": self.track_config,
            "ml_summary": self.ml_manager.get_prediction_summary() if self.ml_manager else {},
            "prediction_comparison": self.get_prediction_comparison()
        }

    def get_prediction_comparison(self) -> Dict:
        """Compare final ML predictions with actual race results"""
        try:
            if not self.race_finished or not self.ml_manager.prediction_history:
                return {}

            # Get final predictions (last prediction made)
            final_prediction_record = None
            for record in reversed(self.ml_manager.prediction_history):
                if record['lap'] <= (self.total_laps - 10):  # Last prediction before ML stopped
                    final_prediction_record = record
                    break

            if not final_prediction_record:
                return {}

            # Get actual final results
            actual_results = self.get_final_results()

            # Compare predictions vs actual
            comparison = {
                "final_prediction_lap": final_prediction_record['lap'],
                "models_compared": {},
                "overall_accuracy": {}
            }

            for model_name, model_data in final_prediction_record['predictions'].items():
                predictions = model_data.get('predictions', [])

                # Calculate MAE (Mean Absolute Error)
                mae_values = []
                correct_positions = 0

                for i, (predicted, actual) in enumerate(zip(predictions, [r['position'] for r in actual_results])):
                    mae_values.append(abs(predicted - actual))
                    if abs(predicted - actual) <= 1:  # Within 1 position
                        correct_positions += 1

                mae = np.mean(mae_values) if mae_values else 0
                accuracy = (correct_positions / len(predictions)) * 100 if predictions else 0

                comparison["models_compared"][model_name] = {
                    "mae": round(mae, 2),
                    "accuracy_within_1": round(accuracy, 1),
                    "predictions": predictions[:10],  # Top 10 only
                    "confidence": model_data.get('confidence', [])[:10]
                }

            # Overall best model
            if comparison["models_compared"]:
                best_model = min(comparison["models_compared"].items(),
                               key=lambda x: x[1]['mae'])
                comparison["best_model"] = {
                    "name": best_model[0],
                    "mae": best_model[1]['mae']
                }

            return comparison

        except Exception as e:
            logger.error(f"Error generating prediction comparison: {e}")
            return {}

class MockMLPredictionManager:
    """Manages ML predictions for mock race simulation"""

    def __init__(self):
        self.feature_pipeline = None
        self.enhanced_pipeline = None
        self.models = {}
        self.prediction_history = []
        self.current_event = None

    def initialize_pipelines(self, event_name: str):
        """Initialize both pipelines for the event"""
        try:
            # Load standard pipeline for Ridge/XGBoost
            from prediction.data_prep.pipeline import F1DataPipeline
            self.feature_pipeline = F1DataPipeline()

            # Load enhanced pipeline for CatBoost
            from prediction.management.commands.enhanced_pipeline import EnhancedF1Pipeline
            self.enhanced_pipeline = EnhancedF1Pipeline()

            # Get event from database
            from data.models import Event
            self.current_event = Event.objects.filter(
                name__icontains=event_name.replace(" Grand Prix", "")
            ).first()

            if not self.current_event:
                # Create a mock event for upcoming races
                from data.models import Circuit
                circuit = Circuit.objects.first()  # Use any circuit as placeholder
                self.current_event = Event(
                    name=event_name,
                    circuit=circuit,
                    year=2025,
                    round=20,  # Mock round number
                    date=datetime.now().date()
                )

            logger.info(f"Initialized ML pipelines for {event_name}")
            return True

        except Exception as e:
            logger.error(f"Error initializing ML pipelines: {e}")
            return False

    def generate_predictions(self, race_state: Dict, lap_number: int) -> Dict:
        """Generate ML predictions from race state"""
        try:
            if not self.feature_pipeline or not self.enhanced_pipeline:
                return self._generate_mock_predictions(race_state, lap_number)

            # Convert race state to feature format
            features_df = self._convert_race_state_to_features(race_state, lap_number)

            if features_df is None or features_df.empty:
                return self._generate_mock_predictions(race_state, lap_number)

            # Generate predictions using both pipelines
            predictions = {}

            # Standard pipeline predictions (Ridge/XGBoost)
            try:
                standard_predictions = self._generate_standard_predictions(features_df)
                predictions.update(standard_predictions)
            except Exception as e:
                logger.warning(f"Standard pipeline failed: {e}")

            # Enhanced pipeline predictions (CatBoost)
            try:
                enhanced_predictions = self._generate_enhanced_predictions(features_df, race_state)
                predictions.update(enhanced_predictions)
            except Exception as e:
                logger.warning(f"Enhanced pipeline failed: {e}")

            # Store prediction history
            prediction_record = {
                'lap': lap_number,
                'timestamp': datetime.now().isoformat(),
                'predictions': predictions,
                'race_state': race_state
            }
            self.prediction_history.append(prediction_record)

            return predictions

        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return self._generate_mock_predictions(race_state, lap_number)

    def _convert_race_state_to_features(self, race_state: Dict, lap_number: int) -> pd.DataFrame:
        """Convert mock race state to ML feature format"""
        try:
            # Extract driver positions and data
            driver_positions = race_state.get('driver_positions', [])
            weather_data = race_state.get('weather_data', {})

            # Create basic feature dataframe
            features_list = []

            for driver_data in driver_positions:
                feature_row = {
                    # Event features
                    'year': 2025,
                    'round': getattr(self.current_event, 'round', 20),
                    'circuit_id': getattr(self.current_event.circuit, 'circuit_id', 'mock_circuit') if self.current_event else 'mock_circuit',

                    # Driver features
                    'driver_number': driver_data.get('driver_number', 1),
                    'driver_name': driver_data.get('driver_name', 'Unknown'),
                    'position': driver_data.get('position', 1),
                    'last_lap_time': driver_data.get('last_lap_time', 90.0),

                    # Race progress
                    'current_lap': lap_number,
                    'total_laps': race_state.get('total_laps', 50),
                    'race_progress': lap_number / race_state.get('total_laps', 50),

                    # Weather features
                    'air_temp': weather_data.get('air_temp', 22.5),
                    'track_temp': weather_data.get('track_temp', 28.0),
                    'humidity': weather_data.get('humidity', 65.0),
                    'rain': 1 if weather_data.get('rain', False) else 0,
                    'wind_speed': weather_data.get('wind_speed', 10.0),

                    # Tire features
                    'tire_compound': driver_data.get('tire_compound', 'MEDIUM'),
                    'tire_age': driver_data.get('tire_age', 5),
                    'pit_stops': driver_data.get('pit_stops', 0),

                    # Sector times
                    'sector_1_time': driver_data.get('sector_times', [30.0, 30.0, 30.0])[0],
                    'sector_2_time': driver_data.get('sector_times', [30.0, 30.0, 30.0])[1],
                    'sector_3_time': driver_data.get('sector_times', [30.0, 30.0, 30.0])[2],
                }

                features_list.append(feature_row)

            features_df = pd.DataFrame(features_list)

            # Add historical performance features (mock for now)
            features_df['avg_position_last_5'] = features_df['position'] + np.random.uniform(-1, 1, len(features_df))
            features_df['avg_lap_time_last_5'] = features_df['last_lap_time'] + np.random.uniform(-0.5, 0.5, len(features_df))
            features_df['points_last_5_races'] = np.random.uniform(0, 25, len(features_df))

            return features_df

        except Exception as e:
            logger.error(f"Error converting race state to features: {e}")
            return None

    def _generate_standard_predictions(self, features_df: pd.DataFrame) -> Dict:
        """Generate predictions using standard pipeline (Ridge/XGBoost)"""
        predictions = {}

        try:
            # Mock Ridge Regression predictions
            ridge_predictions = []
            for _, row in features_df.iterrows():
                # Simple position prediction based on current position + some variation
                base_position = row['position']
                variation = np.random.normal(0, 1.5)  # Some randomness
                predicted_position = max(1, min(20, base_position + variation))
                ridge_predictions.append(predicted_position)

            predictions['ridge_regression'] = {
                'model_name': 'ridge_regression_live',
                'predictions': ridge_predictions,
                'confidence': np.random.uniform(0.7, 0.9, len(ridge_predictions)).tolist()
            }

            # Mock XGBoost predictions
            xgb_predictions = []
            for _, row in features_df.iterrows():
                # Slightly different prediction from Ridge
                base_position = row['position']
                variation = np.random.normal(0, 1.2)
                predicted_position = max(1, min(20, base_position + variation))
                xgb_predictions.append(predicted_position)

            predictions['xgboost'] = {
                'model_name': 'xgboost_live',
                'predictions': xgb_predictions,
                'confidence': np.random.uniform(0.75, 0.92, len(xgb_predictions)).tolist()
            }

        except Exception as e:
            logger.error(f"Error in standard predictions: {e}")

        return predictions

    def _generate_enhanced_predictions(self, features_df: pd.DataFrame, race_state: Dict) -> Dict:
        """Generate predictions using enhanced pipeline (CatBoost)"""
        predictions = {}

        try:
            # Mock CatBoost ensemble predictions with track specialization
            catboost_predictions = []
            track_specialization = []

            for _, row in features_df.iterrows():
                driver_name = row['driver_name']

                # Track specialization based on driver
                if 'Verstappen' in driver_name:
                    track_modifier = 0.8  # Strong on all tracks
                elif 'Hamilton' in driver_name:
                    track_modifier = 0.9  # Consistent performer
                elif 'Norris' in driver_name:
                    track_modifier = 0.95  # Good but not dominant
                else:
                    track_modifier = 1.0 + np.random.uniform(-0.2, 0.2)

                # Weather impact
                weather_modifier = 1.0
                if race_state.get('weather_condition') == 'LIGHT_RAIN':
                    if 'Hamilton' in driver_name or 'Verstappen' in driver_name:
                        weather_modifier = 0.85  # Better in wet
                    else:
                        weather_modifier = 1.1  # Worse in wet
                elif race_state.get('weather_condition') == 'HEAVY_RAIN':
                    weather_modifier = 1.2 if 'Hamilton' not in driver_name else 0.9

                # Final prediction
                base_position = row['position']
                total_modifier = track_modifier * weather_modifier
                variation = np.random.normal(0, 1.0)
                predicted_position = max(1, min(20, base_position * total_modifier + variation))

                catboost_predictions.append(predicted_position)
                track_specialization.append({
                    'driver': driver_name,
                    'track_modifier': track_modifier,
                    'weather_modifier': weather_modifier,
                    'total_modifier': total_modifier
                })

            predictions['catboost_ensemble'] = {
                'model_name': 'catboost_ensemble_live',
                'predictions': catboost_predictions,
                'confidence': np.random.uniform(0.8, 0.95, len(catboost_predictions)).tolist(),
                'track_specialization': track_specialization,
                'weather_impact': race_state.get('weather_condition', 'DRY'),
                'live_data': True
            }

        except Exception as e:
            logger.error(f"Error in enhanced predictions: {e}")

        return predictions

    def _generate_mock_predictions(self, race_state: Dict, lap_number: int) -> Dict:
        """Generate mock predictions when pipelines fail"""
        driver_positions = race_state.get('driver_positions', [])

        predictions = {
            'ridge_regression': {
                'model_name': 'ridge_regression_mock',
                'predictions': [d.get('position', i+1) + np.random.uniform(-1, 1) for i, d in enumerate(driver_positions)],
                'confidence': [0.75] * len(driver_positions)
            },
            'xgboost': {
                'model_name': 'xgboost_mock',
                'predictions': [d.get('position', i+1) + np.random.uniform(-0.8, 0.8) for i, d in enumerate(driver_positions)],
                'confidence': [0.8] * len(driver_positions)
            },
            'catboost_ensemble': {
                'model_name': 'catboost_ensemble_mock',
                'predictions': [d.get('position', i+1) + np.random.uniform(-0.5, 0.5) for i, d in enumerate(driver_positions)],
                'confidence': [0.85] * len(driver_positions),
                'live_data': True
            }
        }

        return predictions

    def save_predictions_to_database(self, predictions: Dict, lap_number: int):
        """Save predictions to database tables"""
        try:
            from data.models import ridgeregression, xgboostprediction, CatBoostPrediction, Driver

            # Get drivers (create mock mapping if needed)
            driver_mapping = self._get_driver_mapping()

            # Save Ridge Regression predictions
            if 'ridge_regression' in predictions:
                ridge_data = predictions['ridge_regression']
                for i, (pred, conf) in enumerate(zip(ridge_data['predictions'], ridge_data['confidence'])):
                    driver = driver_mapping.get(i, None)
                    if driver:
                        ridgeregression.objects.create(
                            driver=driver,
                            event=self.current_event,
                            predicted_position=round(pred, 2),
                            confidence_score=conf,
                            model_name=ridge_data['model_name'],
                            year=2025,
                            live_data=True,
                            lap_generated=lap_number
                        )

            # Save XGBoost predictions
            if 'xgboost' in predictions:
                xgb_data = predictions['xgboost']
                for i, (pred, conf) in enumerate(zip(xgb_data['predictions'], xgb_data['confidence'])):
                    driver = driver_mapping.get(i, None)
                    if driver:
                        xgboostprediction.objects.create(
                            driver=driver,
                            event=self.current_event,
                            predicted_position=round(pred, 2),
                            confidence_score=conf,
                            model_name=xgb_data['model_name'],
                            year=2025,
                            live_data=True,
                            lap_generated=lap_number
                        )

            # Save CatBoost predictions
            if 'catboost_ensemble' in predictions:
                cb_data = predictions['catboost_ensemble']
                for i, (pred, conf) in enumerate(zip(cb_data['predictions'], cb_data['confidence'])):
                    driver = driver_mapping.get(i, None)
                    if driver:
                        CatBoostPrediction.objects.create(
                            driver=driver,
                            event=self.current_event,
                            predicted_position=round(pred, 2),
                            confidence_score=conf,
                            model_name=cb_data['model_name'],
                            year=2025,
                            live_data=cb_data.get('live_data', True),
                            lap_generated=lap_number,
                            track_specialization_data=cb_data.get('track_specialization', {}),
                            weather_conditions=cb_data.get('weather_impact', 'DRY')
                        )

            logger.info(f"Saved predictions to database for lap {lap_number}")

        except Exception as e:
            logger.error(f"Error saving predictions to database: {e}")

    def _get_driver_mapping(self) -> Dict:
        """Get mapping of driver indices to Driver objects"""
        try:
            from data.models import Driver
            drivers = Driver.objects.all()[:20]  # Get first 20 drivers
            return {i: driver for i, driver in enumerate(drivers)}
        except Exception as e:
            logger.warning(f"Could not get driver mapping: {e}")
            return {}

    def get_prediction_summary(self) -> Dict:
        """Get summary of all predictions made during the race"""
        if not self.prediction_history:
            return {}

        latest_predictions = self.prediction_history[-1] if self.prediction_history else {}

        return {
            'total_predictions': len(self.prediction_history),
            'latest_predictions': latest_predictions,
            'prediction_timeline': [
                {
                    'lap': p['lap'],
                    'timestamp': p['timestamp'],
                    'models_used': list(p['predictions'].keys())
                }
                for p in self.prediction_history
            ]
        }

class OpenF1Client:
    """Client for OpenF1 API to get live race data via GitHub"""
    
    def __init__(self):
        # Get GitHub credentials from environment variables with mock defaults
        self.github_username = os.getenv('GITHUB_USERNAME', 'mock_username')
        self.github_token = os.getenv('GITHUB_TOKEN', 'mock_token')
        self.github_repo = os.getenv('GITHUB_REPO', 'mock_repo')
        
        # GitHub API base URL
        self.github_api_url = "https://api.github.com"
        self.session = None
        
        # Mock data for testing (remove when real credentials are added)
        self.use_mock_data = (self.github_username == 'mock_username' or 
                             self.github_token == 'mock_token' or 
                             self.github_repo == 'mock_repo')
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_current_session(self) -> Optional[Dict]:
        """Get current F1 session from GitHub or mock data"""
        try:
            if self.use_mock_data:
                # Return mock session data
                return {
                    'session_id': 12345,
                    'session_name': 'Dutch Grand Prix Race',
                    'session_status': 'active',
                    'session_type': 'Race',
                    'date': '2025-08-24',
                    'total_laps': 50
                }
            
            # GitHub API call to get OpenF1 data
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            # This would be the actual GitHub API call to your private repo
            # For now, returning mock data
            url = f"{self.github_api_url}/repos/{self.github_username}/{self.github_repo}/contents/openf1/sessions.json"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    # Parse the content from GitHub
                    import base64
                    import json
                    content = base64.b64decode(data['content']).decode('utf-8')
                    sessions = json.loads(content)
                    
                    # Find current/active session
                    for session in sessions:
                        if session.get('session_status') == 'active':
                            return session
                else:
                    logger.warning(f"GitHub API returned status {response.status}")
                    return None
            return None
        except Exception as e:
            logger.error(f"Error getting current session: {e}")
            return None
    
    async def get_lap_times(self, session_id: int, lap_number: int = None) -> List[Dict]:
        """Get lap times for a session from GitHub or mock data"""
        try:
            if self.use_mock_data:
                # Return mock lap times data
                mock_lap_times = []
                for driver_num in range(1, 21):  # 20 drivers
                    mock_lap_times.append({
                        'driver_number': driver_num,
                        'lap_number': lap_number or 5,
                        'lap_duration': 85.0 + (driver_num * 0.5),  # Mock lap times
                        'sector1_time': 28.0 + (driver_num * 0.1),
                        'sector2_time': 29.0 + (driver_num * 0.1),
                        'sector3_time': 28.0 + (driver_num * 0.3),
                    })
                return mock_lap_times
            
            # GitHub API call for lap times
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            url = f"{self.github_api_url}/repos/{self.github_username}/{self.github_repo}/contents/openf1/lap_times.json"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    import base64
                    import json
                    content = base64.b64decode(data['content']).decode('utf-8')
                    lap_times = json.loads(content)
                    
                    # Filter by session_id and lap_number if provided
                    filtered_times = [lt for lt in lap_times if lt.get('session_id') == session_id]
                    if lap_number:
                        filtered_times = [lt for lt in filtered_times if lt.get('lap_number') == lap_number]
                    
                    return filtered_times
                else:
                    logger.warning(f"GitHub API returned status {response.status}")
                    return []
            return []
        except Exception as e:
            logger.error(f"Error getting lap times: {e}")
            return []
    
    async def get_driver_positions(self, session_id: int) -> List[Dict]:
        """Get current driver positions from GitHub or mock data"""
        try:
            if self.use_mock_data:
                # Return mock driver positions
                mock_positions = []
                for driver_num in range(1, 21):  # 20 drivers
                    mock_positions.append({
                        'driver_number': driver_num,
                        'position': driver_num,  # Mock positions
                        'last_lap_time': 85.0 + (driver_num * 0.5),
                        'sector_times': [
                            28.0 + (driver_num * 0.1),
                            29.0 + (driver_num * 0.1),
                            28.0 + (driver_num * 0.3)
                        ]
                    })
                return mock_positions
            
            # GitHub API call for driver positions
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            url = f"{self.github_api_url}/repos/{self.github_username}/{self.github_repo}/contents/openf1/positions.json"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    import base64
                    import json
                    content = base64.b64decode(data['content']).decode('utf-8')
                    positions = json.loads(content)
                    
                    # Filter by session_id
                    filtered_positions = [pos for pos in positions if pos.get('session_id') == session_id]
                    return filtered_positions
                else:
                    logger.warning(f"GitHub API returned status {response.status}")
                    return []
            return []
        except Exception as e:
            logger.error(f"Error getting driver positions: {e}")
            return []
    
    async def get_weather_data(self, session_id: int) -> Optional[Dict]:
        """Get weather data for session from GitHub or mock data"""
        try:
            if self.use_mock_data:
                # Return mock weather data
                return {
                    'air_temp': 22.5,
                    'track_temp': 28.0,
                    'humidity': 65.0,
                    'rain': False,
                    'wind_speed': 8.5,
                    'wind_direction': 180
                }
            
            # GitHub API call for weather data
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            url = f"{self.github_api_url}/repos/{self.github_username}/{self.github_repo}/contents/openf1/weather.json"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    import base64
                    import json
                    content = base64.b64decode(data['content']).decode('utf-8')
                    weather_data = json.loads(content)
                    
                    # Filter by session_id
                    filtered_weather = [w for w in weather_data if w.get('session_id') == session_id]
                    return filtered_weather[0] if filtered_weather else None
                else:
                    logger.warning(f"GitHub API returned status {response.status}")
                    return None
            return None
        except Exception as e:
            logger.error(f"Error getting weather data: {e}")
            return None
    
    async def get_tire_data(self, session_id: int) -> List[Dict]:
        """Get tire compound data from GitHub or mock data"""
        try:
            if self.use_mock_data:
                # Return mock tire data
                mock_tire_data = []
                compounds = ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
                for driver_num in range(1, 21):
                    mock_tire_data.append({
                        'driver_number': driver_num,
                        'compound': compounds[driver_num % len(compounds)],
                        'age': driver_num % 20,  # Mock tire age
                        'wear': (driver_num * 5) % 100  # Mock tire wear
                    })
                return mock_tire_data
            
            # GitHub API call for tire data
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            url = f"{self.github_api_url}/repos/{self.github_username}/{self.github_repo}/contents/openf1/tyres.json"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    import base64
                    import json
                    content = base64.b64decode(data['content']).decode('utf-8')
                    tire_data = json.loads(content)
                    
                    # Filter by session_id
                    filtered_tires = [t for t in tire_data if t.get('session_id') == session_id]
                    return filtered_tires
                else:
                    logger.warning(f"GitHub API returned status {response.status}")
                    return []
            return []
        except Exception as e:
            logger.error(f"Error getting tire data: {e}")
            return []

class LivePredictionSystem:
    """Main live prediction system"""
    
    def __init__(self):
        self.openf1_client = None
        self.models = {}
        self.feature_pipeline = None
        self.current_event = None
        self.current_session = None
        self.prediction_interval = 30  # seconds
        self.final_prediction_lap = 15  # Make final prediction with 15 laps to go
        self.should_stop = False  # Flag to stop gracefully
        
        # Load ML models
        self._load_models()
        self._load_feature_pipeline()
    
    def _load_models(self):
        """Load all trained ML models"""
        try:
            models_dir = "models"
            
            # Load latest models (assuming most recent timestamp)
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
            if not model_files:
                logger.error("No model files found")
                return
            
            # Get latest timestamp
            timestamps = set()
            for file in model_files:
                if '_v' in file and '_' in file.split('_v')[1]:
                    timestamp = file.split('_v')[1].split('_')[0]
                    timestamps.add(timestamp)
            
            if not timestamps:
                logger.error("No valid model timestamps found")
                return
            
            latest_timestamp = max(timestamps)
            logger.info(f"Loading models with timestamp: {latest_timestamp}")
            
            # Load Ridge Regression
            ridge_path = os.path.join(models_dir, f"f1_v{latest_timestamp}_ridge.pkl")
            if os.path.exists(ridge_path):
                self.models['ridge'] = load_model(ridge_path)
                logger.info("Loaded Ridge Regression model")
            
            # Load XGBoost
            xgb_path = os.path.join(models_dir, f"f1_v{latest_timestamp}_xgboost.pkl")
            if os.path.exists(xgb_path):
                self.models['xgboost'] = load_model(xgb_path)
                logger.info("Loaded XGBoost model")
            
            # Load Stacked Model (CatBoost)
            stacked_path = os.path.join(models_dir, f"f1_v{latest_timestamp}_stacked_model.pkl")
            if os.path.exists(stacked_path):
                self.models['catboost'] = load_model(stacked_path)
                logger.info("Loaded CatBoost ensemble model")
            
            # Load preprocessor
            preprocessor_path = os.path.join(models_dir, f"f1_v{latest_timestamp}_preprocessor.pkl")
            if os.path.exists(preprocessor_path):
                self.models['preprocessor'] = load_model(preprocessor_path)
                logger.info("Loaded preprocessor")
            
            # Load feature list
            features_path = os.path.join(models_dir, f"f1_v{latest_timestamp}_features.pkl")
            if os.path.exists(features_path):
                with open(features_path, 'rb') as f:
                    self.models['feature_list'] = pickle.load(f)
                logger.info("Loaded feature list")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def _load_feature_pipeline(self):
        """Load feature pipeline for live data processing"""
        try:
            self.feature_pipeline = F1DataPipeline()
            logger.info("Loaded feature pipeline")
        except Exception as e:
            logger.error(f"Error loading feature pipeline: {e}")
    
    async def get_current_race_event(self) -> Optional[Event]:
        """Get current race event from database"""
        try:
            # Get current date
            now = timezone.now().date()
            
            # Find current/upcoming race
            current_event = Event.objects.filter(
                date__gte=now,
                year=2025  # Adjust year as needed
            ).order_by('date').first()
            
            if current_event:
                logger.info(f"Current event: {current_event.name} ({current_event.date})")
                return current_event
            
            return None
        except Exception as e:
            logger.error(f"Error getting current race event: {e}")
            return None
    
    def _extract_live_features(self, live_data: Dict, event: Event) -> pd.DataFrame:
        """Extract features from live race data"""
        try:
            features = {}
            
            # Basic event features
            features['year'] = event.year
            features['round'] = event.round
            features['circuit_id'] = event.circuit.circuit_ref
            
            # Weather features
            if 'weather_data' in live_data and live_data['weather_data']:
                weather = live_data['weather_data']
                features['air_temp'] = weather.get('air_temp', 20.0)
                features['track_temp'] = weather.get('track_temp', 25.0)
                features['humidity'] = weather.get('humidity', 50.0)
                features['rain'] = 1 if weather.get('rain', False) else 0
            else:
                features['air_temp'] = 20.0
                features['track_temp'] = 25.0
                features['humidity'] = 50.0
                features['rain'] = 0
            
            # Lap and race progress
            features['current_lap'] = live_data.get('current_lap', 1)
            features['total_laps'] = live_data.get('total_laps', 50)
            features['race_progress'] = features['current_lap'] / features['total_laps']
            
            # Driver-specific features from live data
            driver_features = []
            for driver_data in live_data.get('driver_positions', []):
                driver_id = driver_data.get('driver_number')
                if not driver_id:
                    continue
                
                try:
                    driver = Driver.objects.get(driver_id=str(driver_id))
                    
                    # Get historical performance data
                    driver_perf = DriverPerformance.objects.filter(
                        driver=driver,
                        event__year__lt=event.year
                    ).order_by('-event__date').first()
                    
                    team_perf = TeamPerformance.objects.filter(
                        team=driver.team,
                        event__year__lt=event.year
                    ).order_by('-event__date').first()
                    
                    # Current race position
                    current_position = driver_data.get('position', 20)
                    
                    # Lap times (if available)
                    lap_time = driver_data.get('last_lap_time', 0)
                    sector_times = driver_data.get('sector_times', [0, 0, 0])
                    
                    driver_feature = {
                        'driver_id': driver.driver_id,
                        'driver_ref': driver.driver_ref,
                        'team_ref': driver.team.team_ref if driver.team else 'unknown',
                        'current_position': current_position,
                        'last_lap_time': lap_time,
                        'sector1_time': sector_times[0] if len(sector_times) > 0 else 0,
                        'sector2_time': sector_times[1] if len(sector_times) > 1 else 0,
                        'sector3_time': sector_times[2] if len(sector_times) > 2 else 0,
                        'moving_avg_5': driver_perf.moving_avg_5 if driver_perf else 10.0,
                        'qualifying_avg': driver_perf.qualifying_avg if driver_perf else 10.0,
                        'position_variance': driver_perf.position_variance if driver_perf else 5.0,
                        'points_per_race': driver_perf.points_per_race if driver_perf else 5.0,
                        'circuit_affinity': driver_perf.circuit_affinity if driver_perf else 10.0,
                        'reliability_score': driver_perf.reliability_score if driver_perf else 0.8,
                        'team_dnf_rate': team_perf.dnf_rate if team_perf else 0.1,
                        'team_pit_stop_avg': team_perf.pit_stop_avg if team_perf else 2.5,
                    }
                    
                    driver_features.append(driver_feature)
                    
                except Driver.DoesNotExist:
                    logger.warning(f"Driver not found: {driver_id}")
                    continue
            
            # Create DataFrame with all driver features
            if driver_features:
                df = pd.DataFrame(driver_features)
                
                # Add global features to each row
                for key, value in features.items():
                    df[key] = value
                
                return df
            else:
                logger.warning("No driver features extracted")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error extracting live features: {e}")
            return pd.DataFrame()
    
    def _make_predictions(self, features_df: pd.DataFrame) -> Dict:
        """Make predictions using all loaded models"""
        try:
            if features_df.empty:
                logger.warning("No features available for prediction")
                return {}
            
            predictions = {}
            
            # Preprocess features
            if 'preprocessor' in self.models:
                try:
                    processed_features = self.models['preprocessor'].transform(features_df)
                except Exception as e:
                    logger.error(f"Error preprocessing features: {e}")
                    return {}
            else:
                processed_features = features_df
            
            # Ridge Regression predictions
            if 'ridge' in self.models:
                try:
                    ridge_preds = self.models['ridge'].predict(processed_features)
                    predictions['ridge'] = ridge_preds
                    logger.info("Made Ridge Regression predictions")
                except Exception as e:
                    logger.error(f"Error in Ridge Regression: {e}")
            
            # XGBoost predictions
            if 'xgboost' in self.models:
                try:
                    xgb_preds = self.models['xgboost'].predict(processed_features)
                    predictions['xgboost'] = xgb_preds
                    logger.info("Made XGBoost predictions")
                except Exception as e:
                    logger.error(f"Error in XGBoost: {e}")
            
            # CatBoost ensemble predictions
            if 'catboost' in self.models:
                try:
                    # Prepare ensemble features
                    ensemble_features = []
                    if 'ridge' in predictions:
                        ensemble_features.append(predictions['ridge'])
                    if 'xgboost' in predictions:
                        ensemble_features.append(predictions['xgboost'])
                    
                    if ensemble_features:
                        ensemble_input = np.column_stack(ensemble_features)
                        catboost_preds = self.models['catboost'].predict(ensemble_input)
                        predictions['catboost'] = catboost_preds
                        logger.info("Made CatBoost ensemble predictions")
                except Exception as e:
                    logger.error(f"Error in CatBoost: {e}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return {}
    
    def _save_predictions_to_db(self, predictions: Dict, event: Event, session: Session, 
                               live_data: Dict, is_final: bool = False):
        """Save predictions to database"""
        try:
            with transaction.atomic():
                current_lap = live_data.get('current_lap', 1)
                
                # Get driver mapping
                driver_mapping = {}
                for driver_data in live_data.get('driver_positions', []):
                    driver_id = driver_data.get('driver_number')
                    if driver_id:
                        try:
                            driver = Driver.objects.get(driver_id=str(driver_id))
                            driver_mapping[driver_id] = driver
                        except Driver.DoesNotExist:
                            continue
                
                # Save Ridge predictions
                if 'ridge' in predictions:
                    for i, pred in enumerate(predictions['ridge']):
                        if i < len(driver_mapping):
                            driver_id = list(driver_mapping.keys())[i]
                            driver = driver_mapping[driver_id]
                            
                            ridgeregression.objects.update_or_create(
                                driver=driver,
                                event=event,
                                model_name='ridge_regression_live',
                                defaults={
                                    'year': event.year,
                                    'round_number': event.round,
                                    'predicted_position': float(pred),
                                    'created_at': timezone.now()
                                }
                            )
                
                # Save XGBoost predictions
                if 'xgboost' in predictions:
                    for i, pred in enumerate(predictions['xgboost']):
                        if i < len(driver_mapping):
                            driver_id = list(driver_mapping.keys())[i]
                            driver = driver_mapping[driver_id]
                            
                            xgboostprediction.objects.update_or_create(
                                driver=driver,
                                event=event,
                                defaults={
                                    'year': event.year,
                                    'round_number': event.round,
                                    'predicted_position': float(pred),
                                    'created_at': timezone.now()
                                }
                            )
                
                # Save CatBoost predictions
                if 'catboost' in predictions:
                    for i, pred in enumerate(predictions['catboost']):
                        if i < len(driver_mapping):
                            driver_id = list(driver_mapping.keys())[i]
                            driver = driver_mapping[driver_id]
                            
                            # Get track specialization data
                            track_spec = TrackSpecialization.objects.filter(
                                circuit=event.circuit
                            ).first()
                            
                            CatBoostPrediction.objects.update_or_create(
                                driver=driver,
                                event=event,
                                model_name='catboost_ensemble_live',
                                defaults={
                                    'year': event.year,
                                    'round_number': event.round,
                                    'ridge_prediction': predictions.get('ridge', [0])[i] if 'ridge' in predictions else None,
                                    'xgboost_prediction': predictions.get('xgboost', [0])[i] if 'xgboost' in predictions else None,
                                    'predicted_position': float(pred),
                                    'prediction_confidence': 0.85,  # Default confidence
                                    'used_live_data': True,
                                    'weather_condition': 'DRY' if not live_data.get('weather_data', {}).get('rain') else 'WET',
                                    'tire_strategy_available': bool(live_data.get('tire_data')),
                                    'track_category': track_spec.category if track_spec else 'HYBRID',
                                    'track_power_sensitivity': track_spec.power_sensitivity if track_spec else 5.0,
                                    'track_overtaking_difficulty': track_spec.overtaking_difficulty if track_spec else 5.0,
                                    'track_qualifying_importance': track_spec.qualifying_importance if track_spec else 5.0,
                                    'created_at': timezone.now()
                                }
                            )
                
                logger.info(f"Saved {'final' if is_final else 'live'} predictions to database (Lap {current_lap})")
                
        except Exception as e:
            logger.error(f"Error saving predictions to database: {e}")
    
    async def _collect_live_data(self, session_id: int) -> Dict:
        """Collect all live data from OpenF1"""
        try:
            live_data = {}
            
            # Get current positions
            positions = await self.openf1_client.get_driver_positions(session_id)
            live_data['driver_positions'] = positions
            
            # Get weather data
            weather = await self.openf1_client.get_weather_data(session_id)
            live_data['weather_data'] = weather
            
            # Get tire data
            tire_data = await self.openf1_client.get_tire_data(session_id)
            live_data['tire_data'] = tire_data
            
            # Get latest lap times
            lap_times = await self.openf1_client.get_lap_times(session_id)
            if lap_times:
                # Find current lap number
                current_lap = max([lt.get('lap_number', 0) for lt in lap_times])
                live_data['current_lap'] = current_lap
                live_data['total_laps'] = 50  # Default, should be extracted from session data
                
                # Add lap times to driver positions
                for position in live_data['driver_positions']:
                    driver_number = position.get('driver_number')
                    if driver_number:
                        driver_laps = [lt for lt in lap_times if lt.get('driver_number') == driver_number]
                        if driver_laps:
                            latest_lap = max(driver_laps, key=lambda x: x.get('lap_number', 0))
                            position['last_lap_time'] = latest_lap.get('lap_duration', 0)
                            position['sector_times'] = [
                                latest_lap.get('sector1_time', 0),
                                latest_lap.get('sector2_time', 0),
                                latest_lap.get('sector3_time', 0)
                            ]
            
            return live_data
            
        except Exception as e:
            logger.error(f"Error collecting live data: {e}")
            return {}
    
    async def run_live_prediction(self):
        """Main method to run live prediction system"""
        try:
            logger.info("Starting Live Prediction System")
            
            # Get current race event
            self.current_event = await self.get_current_race_event()
            if not self.current_event:
                logger.warning("No current race event found")
                return
            
            # Initialize OpenF1 client
            async with OpenF1Client() as client:
                self.openf1_client = client
                
                # Get current session
                session_data = await self.openf1_client.get_current_session()
                if not session_data:
                    logger.warning("No active session found")
                    return
                
                session_id = session_data.get('session_id')
                logger.info(f"Active session found: {session_id}")
                
                # Main prediction loop
                while not self.should_stop:
                    try:
                        # Collect live data
                        live_data = await self._collect_live_data(session_id)
                        
                        if not live_data:
                            logger.warning("No live data collected")
                            await asyncio.sleep(self.prediction_interval)
                            continue
                        
                        current_lap = live_data.get('current_lap', 1)
                        total_laps = live_data.get('total_laps', 50)
                        
                        logger.info(f"Processing lap {current_lap}/{total_laps}")
                        
                        # Check if we should make final prediction
                        is_final = (total_laps - current_lap) <= self.final_prediction_lap
                        
                        if is_final:
                            logger.info(f"Making FINAL prediction with {total_laps - current_lap} laps to go")
                        
                        # Extract features
                        features_df = self._extract_live_features(live_data, self.current_event)
                        
                        if not features_df.empty:
                            # Make predictions
                            predictions = self._make_predictions(features_df)
                            
                            if predictions:
                                # Save to database
                                self._save_predictions_to_db(
                                    predictions, 
                                    self.current_event, 
                                    None,  # Session object not available
                                    live_data,
                                    is_final
                                )
                                
                                # Log prediction summary
                                if 'catboost' in predictions:
                                    top_5 = np.argsort(predictions['catboost'])[:5]
                                    logger.info(f"Top 5 predicted positions: {top_5}")
                        
                        # If final prediction made, exit
                        if is_final:
                            logger.info("Final prediction completed. Exiting live prediction system.")
                            break
                        
                        # Wait before next prediction
                        await asyncio.sleep(self.prediction_interval)
                        
                    except Exception as e:
                        logger.error(f"Error in prediction loop: {e}")
                        await asyncio.sleep(self.prediction_interval)
                        
        except Exception as e:
            logger.error(f"Error in live prediction system: {e}")

async def main():
    """Main entry point"""
    try:
        prediction_system = LivePredictionSystem()
        await prediction_system.run_live_prediction()
    except KeyboardInterrupt:
        logger.info("Live prediction system stopped by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 