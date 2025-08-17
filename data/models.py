from django.db import models
from pytz import timezone
from django.utils import timezone
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver
import logging
from django.db.models import Sum

logger = logging.getLogger(__name__)

class Circuit(models.Model):
    name = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    country = models.CharField(max_length=100)
    circuit_ref = models.CharField(max_length=50, unique=True, default='unknown')
    
    # FastF1 specific metadata
    circuit_id = models.CharField(max_length=50, unique=True, blank=True, null=True)
    circuit_type = models.CharField(
        max_length=20,
        choices=[
            ('HIGH_SPEED', 'High Speed'),
            ('TECHNICAL', 'Technical'),
            ('STREET', 'Street Circuit'),
            ('HYBRID', 'Hybrid')
        ],
        default='HYBRID'
    )
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    altitude = models.FloatField(null=True, blank=True)
    
    def __str__(self):
        return self.name

class Team(models.Model):
    name = models.CharField(max_length=100)
    team_ref = models.CharField(max_length=50, unique=True, default='unknown')
    
    # FastF1 specific metadata
    team_id = models.CharField(max_length=50, unique=True, blank=True, null=True)
    country = models.CharField(max_length=50, blank=True, null=True)
    
    # Performance metrics
    season_dnf_rate = models.FloatField(
        null=True,
        blank=True,
        help_text="Percentage of DNFs in current season"
    )
    pit_stop_avg = models.FloatField(
        null=True,
        blank=True,
        help_text="Average pit stop time in milliseconds"
    )
    
    def __str__(self):
        return self.name

class Driver(models.Model):
    driver_id = models.CharField(max_length=20, unique=True)
    given_name = models.CharField(max_length=50)
    family_name = models.CharField(max_length=50)
    nationality = models.CharField(max_length=50)
    driver_ref = models.CharField(max_length=50, unique=True, default='unknown')
    code = models.CharField(max_length=3, blank=True, null=True)
    permanent_number = models.IntegerField(null=True, blank=True)
    
    # FastF1 specific metadata
    full_name = models.CharField(max_length=100, blank=True, null=True)
    date_of_birth = models.DateField(null=True, blank=True)
    
    # Performance metrics
    recent_form = models.FloatField(
        null=True,
        blank=True,
        help_text="Average finish position in last 5 races"
    )
    consistency_score = models.FloatField(
        null=True,
        blank=True,
        help_text="Position variance over season"
    )
    
    def __str__(self):
        return f"{self.given_name} {self.family_name}"

class Event(models.Model):
    """Renamed from Race to better match FastF1 terminology"""
    year = models.IntegerField()
    round = models.IntegerField()
    name = models.CharField(max_length=100)
    date = models.DateField()
    circuit = models.ForeignKey(Circuit, on_delete=models.CASCADE)
    official_name = models.CharField(max_length=200, blank=True, null=True)
    
    # FastF1 specific metadata
    event_id = models.CharField(max_length=50, unique=True, blank=True, null=True)
    event_format = models.CharField(
        max_length=20,
        choices=[
            ('CONVENTIONAL', 'Conventional'),
            ('SPRINT', 'Sprint Weekend')
        ],
        default='CONVENTIONAL'
    )
    
    # Weather data
    weather_data = models.JSONField(
        default=dict,
        blank=True,
        null=True,
        help_text="Temperature, precipitation, humidity, etc."
    )
    weather_impact = models.FloatField(
        null=True,
        blank=True,
        help_text="Calculated weather impact score"
    )
    
    class Meta:
        unique_together = ('year', 'round')
        indexes = [
            models.Index(fields=['year', 'circuit']),
        ]

    def __str__(self):
        return f"{self.year} {self.name} Round {self.round}"

class SessionType(models.Model):
    """Represents different session types (FP1, FP2, Qualifying, Race, etc.)"""
    name = models.CharField(max_length=50)
    session_type = models.CharField(max_length=20, unique=True)

    def __str__(self):
        return self.name

class Session(models.Model):
    """Represents a specific session (e.g., 2023 Bahrain Grand Prix Qualifying)"""
    event = models.ForeignKey(Event, on_delete=models.CASCADE, related_name='sessions')
    session_type = models.ForeignKey(SessionType, on_delete=models.CASCADE)
    date = models.DateTimeField()
    session_id = models.CharField(max_length=50, unique=True, blank=True, null=True)
    
    # Session-specific weather
    air_temp = models.FloatField(null=True, blank=True)
    track_temp = models.FloatField(null=True, blank=True)
    humidity = models.FloatField(null=True, blank=True)
    wind_speed = models.FloatField(null=True, blank=True)
    wind_direction = models.FloatField(null=True, blank=True)
    rain = models.BooleanField(default=False)
    
    class Meta:
        unique_together = ('event', 'session_type')
        
    def __str__(self):
        return f"{self.event} - {self.session_type}"

class SessionResult(models.Model):
    """Base model for session results (qualifying/race)"""
    session = models.ForeignKey(Session, on_delete=models.CASCADE)
    driver = models.ForeignKey(Driver, on_delete=models.CASCADE)
    team = models.ForeignKey(Team, on_delete=models.CASCADE)
    position = models.IntegerField(null=True)
    position_text = models.CharField(max_length=10, blank=True, null=True)
    time = models.DurationField(null=True, blank=True)
    time_millis = models.IntegerField(null=True, blank=True)  # For faster calculations
    points = models.FloatField(null=True, blank=True)
    
    # FastF1 specific
    status = models.CharField(max_length=50, null=True, blank=True)
    laps = models.IntegerField(null=True, blank=True)
    
    class Meta:
        abstract = True

class QualifyingResult(SessionResult):
    session = models.ForeignKey(  # <-- Add or override this field
        Session,
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    q1 = models.DurationField(null=True, blank=True)
    q2 = models.DurationField(null=True, blank=True)
    q3 = models.DurationField(null=True, blank=True)
    
    # Add qualifying delta (critical feature from Bell 2021)
    pole_delta = models.FloatField(
        null=True,
        blank=True,
        help_text="Time delta to pole position in seconds"
    )
    
    # FastF1 specific
    q1_millis = models.IntegerField(null=True, blank=True)
    q2_millis = models.IntegerField(null=True, blank=True)
    q3_millis = models.IntegerField(null=True, blank=True)

    class Meta:
        unique_together = ('session', 'driver')
        ordering = ['position']

    def __str__(self):
        return f"{self.session} - {self.driver} - Q{self.position}"

class RaceResult(SessionResult):
    session = models.ForeignKey(  # <-- Add or override this field
        Session,
        on_delete=models.CASCADE,
        null=True,
        blank=True
    )
    # Add position gain/loss
    grid_position = models.IntegerField(null=True, blank=True)
    position_gain = models.IntegerField(
        null=True,
        blank=True,
        help_text="Position change from grid to finish"
    )
    
    # FastF1 specific
    fastest_lap_rank = models.IntegerField(null=True, blank=True)
    fastest_lap_time = models.DurationField(null=True, blank=True)
    fastest_lap_speed = models.FloatField(null=True, blank=True)
    pit_stops = models.IntegerField(null=True, blank=True)
    tyre_stints = models.JSONField(null=True, blank=True)  # Store tyre strategy
    
    class Meta:
        unique_together = ('session', 'driver')
        ordering = ['position']
        indexes = [
            models.Index(fields=['session', 'team']),
            models.Index(fields=['driver', 'session']),
        ]

    def __str__(self):
        return f"{self.session} - {self.driver} - Pos {self.position}"

# ===== FEATURE ENGINEERING MODELS =====
class DriverPerformance(models.Model):
    """Rolling performance metrics for drivers"""
    driver = models.ForeignKey(Driver, on_delete=models.CASCADE)
    event = models.ForeignKey(Event, on_delete=models.CASCADE)
    moving_avg_5 = models.FloatField(
        help_text="5-race moving average position"
    )
    qualifying_avg = models.FloatField(
        help_text="Average qualifying position last 5 races"
    )
    position_variance = models.FloatField(
        help_text="Standard deviation of finish positions"
    )
    points_per_race = models.FloatField(
        help_text="Average points per race in current season"
    )
    
    # Existing fields
    circuit_affinity = models.FloatField(
        null=True, blank=True,
        help_text="Average finish position on the same circuit"
    )
    quali_improvement = models.FloatField(
        null=True, blank=True,
        help_text="Improvement in qualifying position over recent races"
    )
    teammate_battle = models.FloatField(
        null=True, blank=True,
        help_text="Teammate head-to-head win ratio"
    )
    wet_weather_perf = models.FloatField(
        null=True, blank=True,
        help_text="Performance metric in wet weather conditions"
    )
    reliability_score = models.FloatField(
        null=True, 
        blank=True,
        help_text="Driver's reliability score"
    )
    
    # New fields
    rivalry_performance = models.FloatField(
        null=True, blank=True,
        help_text="Performance against frequent rivals"
    )
    quali_race_delta = models.FloatField(
        null=True, blank=True,
        help_text="Average positions gained from qualifying to race"
    )
    position_momentum = models.FloatField(
        null=True, blank=True,
        help_text="Trend of position improvements (negative = improving)"
    )
    dnf_rate = models.FloatField(default=0.0, help_text="Driver's DNF rate")
    pit_stop_avg = models.FloatField(default=0.0, help_text="Average pit stop time in seconds")
    
    class Meta:
        unique_together = ('driver', 'event')
        indexes = [
            models.Index(fields=['driver', 'event']),
        ]

class TeamPerformance(models.Model):
    team = models.ForeignKey(Team, on_delete=models.CASCADE)
    event = models.ForeignKey(Event, on_delete=models.CASCADE)
    dnf_rate = models.FloatField(default=0.0, help_text="Team's DNF rate")
    pit_stop_avg = models.FloatField(default=0.0, help_text="Team's average pit stop time")
    moving_avg_5 = models.FloatField(default=0.0, help_text="5-race moving average position")
    position_variance = models.FloatField(default=0.0, help_text="Variance in finishing positions")
    qualifying_avg = models.FloatField(default=0.0, help_text="Average qualifying position")
    reliability_score = models.FloatField(null=True, blank=True)
    development_slope = models.FloatField(
        null=True, blank=True,
        help_text="Trend slope of team position over last 10 races"
    )
    qualifying_consistency = models.FloatField(
        null=True, blank=True,
        help_text="Standard deviation of qualifying positions"
    )
    
    # New field
    pit_stop_std = models.FloatField(
        null=True, blank=True,
        help_text="Standard deviation of pit stop times"
    )
    
    class Meta:
        unique_together = ('team', 'event')


class TrackCharacteristics(models.Model):
    """Precomputed metrics for circuits"""
    circuit = models.ForeignKey(Circuit, on_delete=models.CASCADE)
    overtaking_index = models.FloatField(
        help_text="Average position changes per race"
    )
    safety_car_probability = models.FloatField(
        help_text="Probability of safety car appearance"
    )
    rain_impact = models.FloatField(
        help_text="Performance impact of rain (0-1)"
    )
    avg_pit_loss = models.FloatField(
        help_text="Average time loss per pit stop (seconds)"
    )
    
    class Meta:
        unique_together = ('circuit',)

# ===== PREDICTION MODELS =====
class PredictionModel(models.Model):
    """Stores trained ML model metadata"""
    MODEL_TYPES = [
        ('GENERAL', 'General Prediction'),
        ('TRACK', 'Track-Specific'),
        ('QUALIFYING', 'Qualifying Predictor')
    ]
    name = models.CharField(max_length=100)
    version = models.CharField(max_length=20)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    circuit = models.ForeignKey(
        Circuit, 
        null=True, 
        blank=True, 
        on_delete=models.SET_NULL
    )
    created_at = models.DateTimeField(auto_now_add=True)
    metrics = models.JSONField(
        default=dict,
        help_text="MAE, RMSE, accuracy@3, accuracy@5"
    )
    feature_list = models.JSONField(
        default=list,
        help_text="List of features used in training"
    )
    is_active = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.name} v{self.version}"

class RacePrediction(models.Model):
    """Stores prediction outputs for races"""
    event = models.ForeignKey(Event, on_delete=models.CASCADE)
    session = models.ForeignKey(Session, on_delete=models.CASCADE)
    model = models.ForeignKey(PredictionModel, on_delete=models.CASCADE)
    predicted_at = models.DateTimeField(auto_now_add=True)
    predictions = models.JSONField(
        help_text="Driver-position predictions with confidence scores"
    )
    top_3_accuracy = models.FloatField(
        null=True,
        blank=True,
        help_text="Actual accuracy after session completion"
    )
    
    class Meta:
        unique_together = ('session', 'model')

class PitStop(models.Model):
    """Represents a pit stop during a session"""
    team = models.ForeignKey(Team, on_delete=models.CASCADE)
    driver = models.ForeignKey(Driver, on_delete=models.CASCADE)
    session = models.ForeignKey(Session, on_delete=models.CASCADE)
    lap = models.IntegerField()
    duration = models.DurationField()
    time_millis = models.IntegerField(help_text="Duration in milliseconds")
    tyre_compound = models.CharField(max_length=10, blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('session', 'driver', 'lap')
        indexes = [
            models.Index(fields=['team', 'session']),
        ]

    def __str__(self):
        return f"{self.session} - {self.driver} - Lap {self.lap}"
    

class ridgeregression(models.Model):
    driver = models.ForeignKey(Driver, on_delete=models.CASCADE)
    event = models.ForeignKey(Event, on_delete=models.CASCADE)
    
    year = models.IntegerField()
    round_number = models.IntegerField()

    predicted_position = models.FloatField()
    actual_position = models.IntegerField(null=True, blank=True)

    model_name = models.CharField(max_length=100, default='ridge_regression')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('driver', 'event', 'model_name')
        indexes = [
            models.Index(fields=['year', 'round_number', 'model_name']),
        ]
        ordering = ['predicted_position']

    def __str__(self):
        return f"{self.model_name.upper()} | {self.driver} | {self.event} → Predicted: {self.predicted_position:.2f}"

class xgboostprediction(models.Model):
    driver = models.ForeignKey(Driver, on_delete=models.CASCADE)
    event = models.ForeignKey(Event, on_delete=models.CASCADE)
    
    year = models.IntegerField()  # Temporary default
    round_number = models.IntegerField()  # Temporary default

    predicted_position = models.FloatField()
    actual_position = models.IntegerField(null=True, blank=True)

    model_name = models.CharField(max_length=100, default='xgboost_regression')
    class Meta:
        unique_together = ('event', 'driver')
        verbose_name = "XGBoost Prediction"
        verbose_name_plural = "XGBoost Predictions"

    def __str__(self):
        actual = f" (Actual: {self.actual_position})" if self.actual_position else ""
        return f"{self.driver} - {self.event} - Pred: {self.predicted_position}{actual}"
    


# Add these classes to your existing models.py file

class TrackSpecialization(models.Model):
    """Track categorization for specialized predictions"""
    
    TRACK_CATEGORIES = [
        ('POWER', 'Power Circuits'),
        ('TECHNICAL', 'Technical Circuits'),
        ('STREET', 'Street Circuits'),
        ('HYBRID', 'Hybrid Circuits'),
        ('HIGH_SPEED', 'High Speed Circuits'),
    ]
    
    circuit = models.OneToOneField(Circuit, on_delete=models.CASCADE)
    category = models.CharField(max_length=20, choices=TRACK_CATEGORIES)
    
    overtaking_difficulty = models.FloatField(
        default=5.0,
        help_text="Scale 1-10, where 10 is very difficult to overtake"
    )
    tire_degradation_rate = models.FloatField(
        default=5.0,
        help_text="Scale 1-10, where 10 is very high degradation"
    )
    qualifying_importance = models.FloatField(
        default=5.0,
        help_text="Scale 1-10, where 10 means qualifying position is crucial"
    )
    power_sensitivity = models.FloatField(
        default=5.0,
        help_text="Scale 1-10, where 10 means engine power is very important"
    )
    aero_sensitivity = models.FloatField(
        default=5.0,
        help_text="Scale 1-10, where 10 means aerodynamics are crucial"
    )
    weather_impact = models.FloatField(
        default=5.0,
        help_text="Scale 1-10, where 10 means weather greatly affects results"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Track Specialization"
        verbose_name_plural = "Track Specializations"
    
    def __str__(self):
        return f"{self.circuit.name} - {self.get_category_display()}"
    

    # [Previous models.py content up to initialize_track_data]
    @classmethod
    def initialize_track_data(cls):
        """Initialize track specializations with realistic F1 circuit data"""
        track_data = {
            'Monza': {
                'search_terms': ['Monza', 'Italian', 'Italy'],
                'category': 'POWER',
                'overtaking_difficulty': 3.0,
                'tire_degradation_rate': 4.0,
                'qualifying_importance': 7.0,
                'power_sensitivity': 9.0,
                'aero_sensitivity': 3.0,
                'weather_impact': 6.0
            },
            'Spa-Francorchamps': {
                'search_terms': ['Spa', 'Belgian', 'Belgium', 'Francorchamps'],
                'category': 'POWER',
                'overtaking_difficulty': 4.0,
                'tire_degradation_rate': 6.0,
                'qualifying_importance': 6.0,
                'power_sensitivity': 8.0,
                'aero_sensitivity': 6.0,
                'weather_impact': 9.0
            },
            'Silverstone': {
                'search_terms': ['Silverstone', 'British', 'Britain', 'UK', 'Great Britain'],
                'category': 'HIGH_SPEED',
                'overtaking_difficulty': 4.0,
                'tire_degradation_rate': 8.0,
                'qualifying_importance': 6.0,
                'power_sensitivity': 7.0,
                'aero_sensitivity': 8.0,
                'weather_impact': 8.0
            },
            'Monaco': {
                'search_terms': ['Monaco', 'Monte Carlo', 'Monte-Carlo'],
                'category': 'TECHNICAL',
                'overtaking_difficulty': 10.0,
                'tire_degradation_rate': 2.0,
                'qualifying_importance': 10.0,
                'power_sensitivity': 2.0,
                'aero_sensitivity': 6.0,
                'weather_impact': 9.0
            },
            'Hungary': {
                'search_terms': ['Hungary', 'Hungarian', 'Hungaroring', 'Budapest'],
                'category': 'TECHNICAL',
                'overtaking_difficulty': 8.0,
                'tire_degradation_rate': 6.0,
                'qualifying_importance': 8.0,
                'power_sensitivity': 4.0,
                'aero_sensitivity': 7.0,
                'weather_impact': 7.0
            },
            'Singapore': {
                'search_terms': ['Singapore', 'Marina Bay'],
                'category': 'STREET',
                'overtaking_difficulty': 8.0,
                'tire_degradation_rate': 5.0,
                'qualifying_importance': 8.0,
                'power_sensitivity': 4.0,
                'aero_sensitivity': 6.0,
                'weather_impact': 8.0
            },
            'Baku': {
                'search_terms': ['Baku', 'Azerbaijan', 'Azerbaijan Grand Prix'],
                'category': 'STREET',
                'overtaking_difficulty': 4.0,
                'tire_degradation_rate': 5.0,
                'qualifying_importance': 7.0,
                'power_sensitivity': 8.0,
                'aero_sensitivity': 5.0,
                'weather_impact': 6.0
            },
            'Jeddah': {
                'search_terms': ['Jeddah', 'Saudi', 'Saudi Arabia', 'Arabian'],
                'category': 'STREET',
                'overtaking_difficulty': 5.0,
                'tire_degradation_rate': 6.0,
                'qualifying_importance': 7.0,
                'power_sensitivity': 7.0,
                'aero_sensitivity': 7.0,
                'weather_impact': 4.0
            },
            'Suzuka': {
                'search_terms': ['Suzuka', 'Japanese', 'Japan'],
                'category': 'HIGH_SPEED',
                'overtaking_difficulty': 6.0,
                'tire_degradation_rate': 7.0,
                'qualifying_importance': 7.0,
                'power_sensitivity': 6.0,
                'aero_sensitivity': 9.0,
                'weather_impact': 8.0
            },
            'Interlagos': {
                'search_terms': ['Interlagos', 'Brazilian', 'Brazil', 'São Paulo', 'Sao Paulo'],
                'category': 'HIGH_SPEED',
                'overtaking_difficulty': 5.0,
                'tire_degradation_rate': 6.0,
                'qualifying_importance': 6.0,
                'power_sensitivity': 6.0,
                'aero_sensitivity': 7.0,
                'weather_impact': 9.0
            },
            'Bahrain': {
                'search_terms': ['Bahrain', 'Sakhir'],
                'category': 'HYBRID',
                'overtaking_difficulty': 4.0,
                'tire_degradation_rate': 7.0,
                'qualifying_importance': 6.0,
                'power_sensitivity': 6.0,
                'aero_sensitivity': 6.0,
                'weather_impact': 3.0
            },
            'Barcelona': {
                'search_terms': ['Barcelona', 'Spanish', 'Spain', 'Catalunya', 'Catalonia'],
                'category': 'HYBRID',
                'overtaking_difficulty': 7.0,
                'tire_degradation_rate': 8.0,
                'qualifying_importance': 7.0,
                'power_sensitivity': 5.0,
                'aero_sensitivity': 8.0,
                'weather_impact': 5.0
            },
            'Austria': {
                'search_terms': ['Austria', 'Austrian', 'Red Bull Ring', 'Spielberg'],
                'category': 'POWER',
                'overtaking_difficulty': 3.0,
                'tire_degradation_rate': 6.0,
                'qualifying_importance': 5.0,
                'power_sensitivity': 8.0,
                'aero_sensitivity': 6.0,
                'weather_impact': 7.0
            },
            'Imola': {
                'search_terms': ['Imola', 'Enzo', 'Ferrari', 'San Marino', 'Emilia'],
                'category': 'TECHNICAL',
                'overtaking_difficulty': 8.0,
                'tire_degradation_rate': 5.0,
                'qualifying_importance': 8.0,
                'power_sensitivity': 4.0,
                'aero_sensitivity': 7.0,
                'weather_impact': 6.0
            },
            'Miami': {
                'search_terms': ['Miami', 'Florida', 'United States'],
                'category': 'STREET',
                'overtaking_difficulty': 6.0,
                'tire_degradation_rate': 7.0,
                'qualifying_importance': 7.0,
                'power_sensitivity': 6.0,
                'aero_sensitivity': 6.0,
                'weather_impact': 7.0
            },
            'Canada': {
                'search_terms': ['Canada', 'Canadian', 'Montreal', 'Gilles Villeneuve', 'Notre Dame'],
                'category': 'POWER',
                'overtaking_difficulty': 4.0,
                'tire_degradation_rate': 5.0,
                'qualifying_importance': 6.0,
                'power_sensitivity': 8.0,
                'aero_sensitivity': 5.0,
                'weather_impact': 7.0
            },
            'France': {
                'search_terms': ['France', 'French', 'Paul Ricard', 'Le Castellet'],
                'category': 'HYBRID',
                'overtaking_difficulty': 5.0,
                'tire_degradation_rate': 8.0,
                'qualifying_importance': 6.0,
                'power_sensitivity': 6.0,
                'aero_sensitivity': 6.0,
                'weather_impact': 5.0
            },
            'Netherlands': {
                'search_terms': ['Netherlands', 'Dutch', 'Zandvoort'],
                'category': 'HIGH_SPEED',
                'overtaking_difficulty': 6.0,
                'tire_degradation_rate': 6.0,
                'qualifying_importance': 7.0,
                'power_sensitivity': 6.0,
                'aero_sensitivity': 8.0,
                'weather_impact': 8.0
            },
            'Mexico': {
                'search_terms': ['Mexico', 'Mexican', 'Rodriguez', 'Mexico City'],
                'category': 'HYBRID',
                'overtaking_difficulty': 4.0,
                'tire_degradation_rate': 6.0,
                'qualifying_importance': 6.0,
                'power_sensitivity': 7.0,
                'aero_sensitivity': 7.0,
                'weather_impact': 5.0
            },
            'Las Vegas': {
                'search_terms': ['Las Vegas', 'Vegas', 'Nevada'],
                'category': 'POWER',
                'overtaking_difficulty': 4.0,
                'tire_degradation_rate': 4.0,
                'qualifying_importance': 6.0,
                'power_sensitivity': 9.0,
                'aero_sensitivity': 4.0,
                'weather_impact': 3.0
            },
            'Qatar': {
                'search_terms': ['Qatar', 'Losail', 'Doha'],
                'category': 'HYBRID',
                'overtaking_difficulty': 5.0,
                'tire_degradation_rate': 7.0,
                'qualifying_importance': 6.0,
                'power_sensitivity': 6.0,
                'aero_sensitivity': 7.0,
                'weather_impact': 4.0
            },
            'Abu Dhabi': {
                'search_terms': ['Abu Dhabi', 'UAE', 'Emirates', 'Yas Marina', 'United Arab Emirates'],
                'category': 'HYBRID',
                'overtaking_difficulty': 6.0,
                'tire_degradation_rate': 5.0,
                'qualifying_importance': 7.0,
                'power_sensitivity': 6.0,
                'aero_sensitivity': 6.0,
                'weather_impact': 3.0
            },
            'Melbourne': {
                'search_terms': ['Melbourne', 'Australian', 'Australia', 'Albert Park'],
                'category': 'HIGH_SPEED',
                'overtaking_difficulty': 5.5,
                'tire_degradation_rate': 7.0,
                'qualifying_importance': 6.5,
                'power_sensitivity': 7.5,
                'aero_sensitivity': 6.5,
                'weather_impact': 4.0
            },
            'Austin': {
                'search_terms': ['Austin', 'United States', 'COTA', 'Circuit of the Americas'],
                'category': 'TECHNICAL',
                'overtaking_difficulty': 6.0,
                'tire_degradation_rate': 6.0,
                'qualifying_importance': 7.0,
                'power_sensitivity': 6.5,
                'aero_sensitivity': 7.5,
                'weather_impact': 5.5
            },
            'Shanghai': {
                'search_terms': ['Shanghai', 'Chinese', 'China'],
                'category': 'TECHNICAL',
                'overtaking_difficulty': 5.5,
                'tire_degradation_rate': 7.0,
                'qualifying_importance': 6.5,
                'power_sensitivity': 6.5,
                'aero_sensitivity': 6.5,
                'weather_impact': 4.5
            }
        }
        # [Rest of initialize_track_data implementation unchanged]
# [Rest of models.py unchanged]
        
        def find_circuit(search_terms):
            """Find circuit using multiple search strategies"""
            for term in search_terms:
                circuit = Circuit.objects.filter(name__icontains=term).first()
                if circuit:
                    logger.info(f"Found circuit for {term}: {circuit.name}")
                    return circuit
                circuit = Circuit.objects.filter(location__icontains=term).first()
                if circuit:
                    logger.info(f"Found circuit for {term}: {circuit.name}")
                    return circuit
                circuit = Circuit.objects.filter(country__icontains=term).first()
                if circuit:
                    logger.info(f"Found circuit for {term}: {circuit.name}")
                    return circuit
            logger.warning(f"No circuit found for search terms: {search_terms}")
            return None
        
        created_count = 0
        updated_count = 0
        not_found = []
        
        for circuit_name, config in track_data.items():
            try:
                circuit = find_circuit(config['search_terms'])
                
                if not circuit:
                    not_found.append(f"{circuit_name} (searched: {config['search_terms']})")
                    continue
                
                data = {k: v for k, v in config.items() if k != 'search_terms'}
                
                specialization, created = cls.objects.get_or_create(
                    circuit=circuit,
                    defaults=data
                )
                
                if created:
                    created_count += 1
                    logger.info(f"Created track specialization for {circuit_name} -> {circuit.name}")
                else:
                    for key, value in data.items():
                        setattr(specialization, key, value)
                    specialization.save()
                    updated_count += 1
                    logger.info(f"Updated track specialization for {circuit_name} -> {circuit.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {circuit_name}: {str(e)}", exc_info=True)
        
        if not_found:
            logger.warning(f"Could not find circuits for: {not_found}")
        
        logger.info(f"Track specialization initialization complete: {created_count} created, {updated_count} updated")
        return created_count, updated_count


class DriverSpecialization(models.Model):
    """Driver specialization characteristics"""
    
    SPECIALIZATION_TYPES = [
        ('OVERTAKING', 'Overtaking Specialist'),
        ('QUALIFYING', 'Qualifying Specialist'), 
        ('CONSISTENCY', 'Consistency Specialist'),
        ('WET_WEATHER', 'Wet Weather Specialist'),
        ('TIRE_MANAGEMENT', 'Tire Management Specialist'),
        ('TECHNICAL', 'Technical Circuit Specialist'),
        ('POWER', 'Power Circuit Specialist'),
    ]
    
    driver = models.ForeignKey(Driver, on_delete=models.CASCADE)
    specialization_type = models.CharField(max_length=20, choices=SPECIALIZATION_TYPES)
    strength_score = models.FloatField(
        help_text="Strength in this specialization (1-10 scale)"
    )
    
    # Track type performance modifiers
    power_circuit_modifier = models.FloatField(default=1.0)
    technical_circuit_modifier = models.FloatField(default=1.0)
    street_circuit_modifier = models.FloatField(default=1.0)
    wet_weather_modifier = models.FloatField(default=1.0)
    
    year = models.IntegerField(help_text="Year this specialization applies to")
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('driver', 'specialization_type', 'year')
        indexes = [
            models.Index(fields=['driver', 'year']),
        ]
    
    def __str__(self):
        return f"{self.driver} - {self.get_specialization_type_display()} ({self.year})"


# Enhanced prediction model for the new pipeline
class CatBoostPrediction(models.Model):
    """CatBoost predictions with track specialization"""
    driver = models.ForeignKey(Driver, on_delete=models.CASCADE)
    event = models.ForeignKey(Event, on_delete=models.CASCADE)
    
    year = models.IntegerField()
    round_number = models.IntegerField()
    
    # Base model predictions (inputs to CatBoost)
    ridge_prediction = models.FloatField(null=True, blank=True)
    xgboost_prediction = models.FloatField(null=True, blank=True)
    ensemble_prediction = models.FloatField(null=True, blank=True)
    
    # Track specialization features
    track_category = models.CharField(max_length=20, null=True, blank=True)
    track_power_sensitivity = models.FloatField(null=True, blank=True)
    track_overtaking_difficulty = models.FloatField(null=True, blank=True)
    track_qualifying_importance = models.FloatField(null=True, blank=True)
    
    # Final CatBoost prediction
    predicted_position = models.FloatField()
    prediction_confidence = models.FloatField(null=True, blank=True)
    
    # Actual results for comparison
    actual_position = models.IntegerField(null=True, blank=True)
    
    # OpenF1 integration flags
    used_live_data = models.BooleanField(default=False)
    weather_condition = models.CharField(max_length=20, null=True, blank=True)
    tire_strategy_available = models.BooleanField(default=False)
    
    model_name = models.CharField(max_length=100, default='catboost_ensemble')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        unique_together = ('driver', 'event', 'model_name')
        indexes = [
            models.Index(fields=['year', 'round_number']),
            models.Index(fields=['event', 'track_category']),
        ]
        ordering = ['predicted_position']
    
    def __str__(self):
        actual = f" (Actual: {self.actual_position})" if self.actual_position else ""
        return f"CatBoost | {self.driver} | {self.event} → {self.predicted_position:.2f}{actual}"


# Virtual Credits System Models
class UserProfile(models.Model):
    """Extended user profile with credits and betting information"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    credits = models.IntegerField(default=5000, help_text="Virtual credits for betting")
    total_bets_placed = models.IntegerField(default=0, help_text="Total number of bets placed")
    total_credits_won = models.IntegerField(default=0, help_text="Total credits won from bets")
    total_credits_lost = models.IntegerField(default=0, help_text="Total credits lost from bets")
    favorite_circuit = models.ForeignKey('Circuit', on_delete=models.SET_NULL, null=True, blank=True)
    join_date = models.DateTimeField(auto_now_add=True)
    last_active = models.DateTimeField(auto_now=True)
    
    # Achievement tracking
    circuits_visited = models.ManyToManyField('Circuit', blank=True, related_name='visitors')
    achievements_unlocked = models.ManyToManyField('Achievement', blank=True, related_name='unlocked_by')
    
    # Risk management and betting limits
    max_bet_amount = models.IntegerField(default=1000, help_text="Maximum bet amount allowed")
    daily_bet_limit = models.IntegerField(default=5000, help_text="Daily betting limit")
    daily_bets_placed = models.IntegerField(default=0, help_text="Bets placed today")
    daily_bet_amount = models.IntegerField(default=0, help_text="Total amount bet today")
    last_bet_date = models.DateField(null=True, blank=True, help_text="Date of last bet")
    
    # Risk tolerance settings
    risk_tolerance = models.CharField(
        max_length=20,
        choices=[
            ('CONSERVATIVE', 'Conservative'),
            ('MODERATE', 'Moderate'),
            ('AGGRESSIVE', 'Aggressive'),
        ],
        default='MODERATE'
    )
    
    # Subscription tier system
    SUBSCRIPTION_TIERS = [
        ('BASIC', 'Basic (Free)'),
        ('PREMIUM', 'Premium'),
        ('PRO', 'Pro'),
    ]
    
    subscription_tier = models.CharField(
        max_length=20,
        choices=SUBSCRIPTION_TIERS,
        default='BASIC',
        help_text="User's subscription tier for ML model access"
    )
    subscription_start_date = models.DateTimeField(null=True, blank=True)
    subscription_end_date = models.DateTimeField(null=True, blank=True)
    is_subscription_active = models.BooleanField(default=True)
    
    # Betting restrictions
    is_suspended = models.BooleanField(default=False, help_text="Account suspended from betting")
    suspension_reason = models.TextField(blank=True, help_text="Reason for suspension")
    suspension_until = models.DateTimeField(null=True, blank=True, help_text="Suspension end date")
    
    class Meta:
        verbose_name = "User Profile"
        verbose_name_plural = "User Profiles"
    
    def __str__(self):
        return f"{self.user.username}'s Profile"
    
    @property
    def win_rate(self):
        """Calculate win rate percentage"""
        total_bets = self.total_bets_placed
        if total_bets == 0:
            return 0.0
        return round((self.total_credits_won / (self.total_credits_won + self.total_credits_lost)) * 100, 1)
    
    @property
    def net_profit(self):
        """Calculate net profit/loss"""
        return self.total_credits_won - self.total_credits_lost
    
    def can_place_bet(self, amount):
        """Check if user can place a bet of given amount"""
        # Check if account is suspended
        if self.is_suspended:
            if self.suspension_until and timezone.now() < self.suspension_until:
                return False, "Account is suspended"
            else:
                # Suspension expired, reactivate account
                self.is_suspended = False
                self.suspension_reason = ""
                self.suspension_until = None
                self.save()
        
        # Check if user has enough credits
        if self.credits < amount:
            return False, "Insufficient credits"
        
        # Check maximum bet amount
        if amount > self.max_bet_amount:
            return False, f"Bet amount exceeds maximum allowed ({self.max_bet_amount} credits)"
        
        # Check daily limits
        today = timezone.now().date()
        if self.last_bet_date != today:
            # Reset daily counters
            self.daily_bets_placed = 0
            self.daily_bet_amount = 0
            self.last_bet_date = today
            self.save()
        
        if self.daily_bet_amount + amount > self.daily_bet_limit:
            return False, f"Daily betting limit exceeded ({self.daily_bet_limit} credits)"
        
        return True, "OK"
    
    def update_betting_stats(self, bet_amount, won=False, payout=0):
        """Update betting statistics after a bet"""
        self.total_bets_placed += 1
        self.daily_bets_placed += 1
        self.daily_bet_amount += bet_amount
        
        if won:
            self.total_credits_won += payout
        else:
            self.total_credits_lost += bet_amount
        
        self.save()
    
    def get_risk_adjusted_limits(self):
        """Get risk-adjusted betting limits based on user's risk tolerance"""
        base_limits = {
            'CONSERVATIVE': {
                'max_bet_percent': 0.05,  # 5% of total credits
                'daily_limit_percent': 0.20,  # 20% of total credits
                'max_concurrent_bets': 3
            },
            'MODERATE': {
                'max_bet_percent': 0.10,  # 10% of total credits
                'daily_limit_percent': 0.40,  # 40% of total credits
                'max_concurrent_bets': 5
            },
            'AGGRESSIVE': {
                'max_bet_percent': 0.20,  # 20% of total credits
                'daily_limit_percent': 0.60,  # 60% of total credits
                'max_concurrent_bets': 8
            }
        }
        
        limits = base_limits.get(self.risk_tolerance, base_limits['MODERATE'])
        return {
            'max_bet_amount': int(self.credits * limits['max_bet_percent']),
            'daily_bet_limit': int(self.credits * limits['daily_limit_percent']),
            'max_concurrent_bets': limits['max_concurrent_bets']
        }
    
    def get_available_models(self):
        """Get list of ML models available to user based on subscription tier"""
        model_access = {
            'BASIC': ['ridge_regression'],
            'PREMIUM': ['ridge_regression', 'xgboost'],
            'PRO': ['ridge_regression', 'xgboost', 'catboost']
        }
        return model_access.get(self.subscription_tier, ['ridge_regression'])
    
    def can_access_model(self, model_name):
        """Check if user can access a specific ML model"""
        # Basic model is always available to authenticated users
        if model_name == 'ridge_regression':
            return True
            
        # For premium/pro models, check subscription status
        if not self.is_subscription_active:
            return False
        
        # Check if subscription has expired
        if self.subscription_end_date and timezone.now() > self.subscription_end_date:
            return False
            
        available_models = self.get_available_models()
        return model_name in available_models
    
    def get_subscription_display_info(self):
        """Get subscription tier display information"""
        tier_info = {
            'BASIC': {
                'name': 'Basic (Free)',
                'color': 'secondary',
                'features': ['Ridge Regression Model', 'Basic Predictions'],
                'price': 'Free'
            },
            'PREMIUM': {
                'name': 'Premium',
                'color': 'primary',
                'features': ['Ridge Regression', 'XGBoost Model', 'Enhanced Accuracy'],
                'price': '$9.99/month'
            },
            'PRO': {
                'name': 'Pro',
                'color': 'success',
                'features': ['All Models', 'CatBoost Ensemble', 'Maximum Accuracy', 'Priority Support'],
                'price': '$19.99/month'
            }
        }
        return tier_info.get(self.subscription_tier, tier_info['BASIC'])


class CreditTransaction(models.Model):
    """Track all credit transactions for audit trail"""
    TRANSACTION_TYPES = [
        ('SIGNUP_BONUS', 'Signup Bonus'),
        ('BET_PLACED', 'Bet Placed'),
        ('BET_WON', 'Bet Won'),
        ('BET_LOST', 'Bet Lost'),
        ('ACHIEVEMENT_BONUS', 'Achievement Bonus'),
        ('CIRCUIT_VISIT', 'Circuit Visit Bonus'),
        ('ADMIN_ADJUSTMENT', 'Admin Adjustment'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='credit_transactions')
    transaction_type = models.CharField(max_length=20, choices=TRANSACTION_TYPES)
    amount = models.IntegerField(help_text="Positive for credits gained, negative for credits spent")
    description = models.CharField(max_length=200)
    balance_after = models.IntegerField(help_text="User's credit balance after this transaction")
    timestamp = models.DateTimeField(auto_now_add=True)
    
    # Optional references
    bet = models.ForeignKey('Bet', on_delete=models.SET_NULL, null=True, blank=True)
    achievement = models.ForeignKey('Achievement', on_delete=models.SET_NULL, null=True, blank=True)
    circuit = models.ForeignKey('Circuit', on_delete=models.SET_NULL, null=True, blank=True)
    
    class Meta:
        ordering = ['-timestamp']
        verbose_name = "Credit Transaction"
        verbose_name_plural = "Credit Transactions"
    
    def __str__(self):
        return f"{self.user.username} - {self.get_transaction_type_display()} ({self.amount:+d})"


class Bet(models.Model):
    """Enhanced user betting model for prediction market"""
    BET_TYPES = [
        ('PODIUM_FINISH', 'Podium Finish'),
        ('EXACT_POSITION', 'Exact Position'),
        ('DNF_PREDICTION', 'DNF Prediction'),
        ('QUALIFYING_POSITION', 'Qualifying Position'),
        ('FASTEST_LAP', 'Fastest Lap'),
        ('HEAD_TO_HEAD', 'Head-to-Head'),
        ('TEAM_BATTLE', 'Team Battle'),
        ('SAFETY_CAR', 'Safety Car'),
        ('WEATHER_BET', 'Weather Bet'),
    ]
    
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('WON', 'Won'),
        ('LOST', 'Lost'),
        ('CANCELLED', 'Cancelled'),
        ('VOID', 'Void'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='bets')
    event = models.ForeignKey('Event', on_delete=models.CASCADE)
    bet_type = models.CharField(max_length=20, choices=BET_TYPES)
    
    # Primary selection
    driver = models.ForeignKey('Driver', on_delete=models.CASCADE, null=True, blank=True)
    team = models.ForeignKey('Team', on_delete=models.CASCADE, null=True, blank=True)
    
    # For head-to-head and team battles
    opponent_driver = models.ForeignKey('Driver', on_delete=models.CASCADE, null=True, blank=True, related_name='opponent_bets')
    opponent_team = models.ForeignKey('Team', on_delete=models.CASCADE, null=True, blank=True, related_name='opponent_bets')
    
    # Bet details
    predicted_position = models.IntegerField(null=True, blank=True)
    credits_staked = models.IntegerField()
    odds = models.FloatField(help_text="Decimal odds (e.g., 2.5 means 2.5x return)")
    potential_payout = models.IntegerField(help_text="Credits to be won if bet succeeds")
    
    # Market tracking
    market_volume_at_time = models.IntegerField(default=0, help_text="Total market volume when bet was placed")
    odds_movement = models.JSONField(default=list, help_text="Odds movement history")
    
    # Result tracking
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    actual_result = models.CharField(max_length=50, null=True, blank=True)
    payout_received = models.IntegerField(default=0)
    
    # ML prediction integration
    ml_prediction_used = models.BooleanField(default=False)
    ml_predicted_position = models.FloatField(null=True, blank=True)
    ml_confidence = models.FloatField(null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    settled_at = models.DateTimeField(null=True, blank=True)
    notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Bet"
        verbose_name_plural = "Bets"
        indexes = [
            models.Index(fields=['event', 'bet_type']),
            models.Index(fields=['user', 'status']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        bet_description = f"{self.get_bet_type_display()}"
        if self.driver:
            bet_description += f" - {self.driver.given_name} {self.driver.family_name}"
        elif self.team:
            bet_description += f" - {self.team.name}"
        return f"{self.user.username}: {bet_description} ({self.event.name})"
    
    def save(self, *args, **kwargs):
        # Calculate potential payout if not set
        if not self.potential_payout:
            self.potential_payout = int(self.credits_staked * self.odds)
        
        # Track market volume at time of bet
        if not self.market_volume_at_time:
            self.market_volume_at_time = Bet.objects.filter(
                event=self.event,
                bet_type=self.bet_type,
                driver=self.driver,
                team=self.team
            ).aggregate(total=Sum('credits_staked'))['total'] or 0
        
        super().save(*args, **kwargs)
    
    @property
    def is_winning_bet(self):
        """Determine if bet is winning based on actual results"""
        if self.status != 'PENDING':
            return self.status == 'WON'
        
        # This would need to be implemented based on actual race results
        # For now, return None for pending bets
        return None
    
    @property
    def market_performance(self):
        """Calculate how this bet performed relative to market"""
        if not self.market_volume_at_time:
            return 0
        
        current_volume = Bet.objects.filter(
            event=self.event,
            bet_type=self.bet_type,
            driver=self.driver,
            team=self.team
        ).aggregate(total=Sum('credits_staked'))['total'] or 0
        
        return ((current_volume - self.market_volume_at_time) / self.market_volume_at_time * 100) if self.market_volume_at_time > 0 else 0


class Achievement(models.Model):
    """Achievement system for user engagement"""
    ACHIEVEMENT_TYPES = [
        ('ACCURACY', 'Prediction Accuracy'),
        ('CIRCUIT_MASTERY', 'Circuit Mastery'),
        ('BETTING_VOLUME', 'Betting Volume'),
        ('PROFIT_MAKING', 'Profit Making'),
        ('EXPLORER', 'Circuit Explorer'),
        ('SPECIAL', 'Special Achievement'),
    ]
    
    name = models.CharField(max_length=100)
    description = models.TextField()
    achievement_type = models.CharField(max_length=20, choices=ACHIEVEMENT_TYPES)
    icon = models.CharField(max_length=50, help_text="FontAwesome icon class")
    
    # Unlock conditions
    required_accuracy = models.FloatField(null=True, blank=True, help_text="Required prediction accuracy %")
    required_circuits = models.IntegerField(null=True, blank=True, help_text="Required circuits visited")
    required_bets = models.IntegerField(null=True, blank=True, help_text="Required number of bets")
    required_profit = models.IntegerField(null=True, blank=True, help_text="Required profit in credits")
    
    # Rewards
    bonus_credits = models.IntegerField(default=0)
    special_perks = models.JSONField(default=dict, blank=True)
    
    # Display
    rarity = models.CharField(max_length=20, choices=[
        ('COMMON', 'Common'),
        ('RARE', 'Rare'),
        ('EPIC', 'Epic'),
        ('LEGENDARY', 'Legendary'),
    ], default='COMMON')
    
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Achievement"
        verbose_name_plural = "Achievements"
    
    def __str__(self):
        return self.name


# Signal to create UserProfile when User is created
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    if hasattr(instance, 'profile'):
        instance.profile.save()

class MarketMaker(models.Model):
    """Market maker model for managing liquidity and odds adjustment"""
    event = models.ForeignKey('Event', on_delete=models.CASCADE)
    bet_type = models.CharField(max_length=20, choices=Bet.BET_TYPES)
    driver = models.ForeignKey('Driver', on_delete=models.CASCADE, null=True, blank=True)
    team = models.ForeignKey('Team', on_delete=models.CASCADE, null=True, blank=True)
    
    # Market state
    total_volume = models.IntegerField(default=0, help_text="Total betting volume")
    total_bets = models.IntegerField(default=0, help_text="Total number of bets")
    current_odds = models.FloatField(help_text="Current market odds")
    base_odds = models.FloatField(help_text="Base odds without market adjustment")
    
    # Liquidity management
    available_liquidity = models.IntegerField(default=10000, help_text="Available credits for market making")
    max_exposure = models.IntegerField(default=5000, help_text="Maximum exposure per bet")
    
    # Market maker settings
    spread_percentage = models.FloatField(default=0.05, help_text="Bid-ask spread as percentage")
    adjustment_sensitivity = models.FloatField(default=0.1, help_text="How quickly odds adjust to volume")
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ('event', 'bet_type', 'driver', 'team')
        indexes = [
            models.Index(fields=['event', 'bet_type']),
            models.Index(fields=['total_volume']),
        ]
    
    def __str__(self):
        bet_description = f"{self.get_bet_type_display()}"
        if self.driver:
            bet_description += f" - {self.driver.given_name} {self.driver.family_name}"
        elif self.team:
            bet_description += f" - {self.team.name}"
        return f"{self.event.name}: {bet_description}"
    
    def calculate_adjusted_odds(self, bet_amount):
        """Calculate adjusted odds based on market dynamics"""
        # Base odds from historical data and ML predictions
        base_odds = self.base_odds
        
        # Volume impact adjustment
        volume_ratio = self.total_volume / max(self.available_liquidity, 1)
        volume_adjustment = 1 + (volume_ratio * self.adjustment_sensitivity)
        
        # Exposure adjustment
        exposure_ratio = (self.total_volume + bet_amount) / max(self.max_exposure, 1)
        if exposure_ratio > 1:
            exposure_adjustment = 1 + (exposure_ratio - 1) * 0.2
        else:
            exposure_adjustment = 1.0
        
        # Calculate final odds
        adjusted_odds = base_odds * volume_adjustment * exposure_adjustment
        
        # Apply spread
        bid_odds = adjusted_odds * (1 - self.spread_percentage)
        ask_odds = adjusted_odds * (1 + self.spread_percentage)
        
        return {
            'bid': round(bid_odds, 2),
            'ask': round(ask_odds, 2),
            'mid': round(adjusted_odds, 2)
        }
    
    def update_market_state(self, bet_amount):
        """Update market state after a bet is placed"""
        self.total_volume += bet_amount
        self.total_bets += 1
        
        # Recalculate current odds
        new_odds = self.calculate_adjusted_odds(0)
        self.current_odds = new_odds['mid']
        
        self.save()
    
    def get_market_depth(self):
        """Get market depth information"""
        recent_bets = Bet.objects.filter(
            event=self.event,
            bet_type=self.bet_type,
            driver=self.driver,
            team=self.team
        ).order_by('-created_at')[:10]
        
        depth_data = {
            'total_volume': self.total_volume,
            'total_bets': self.total_bets,
            'current_odds': self.current_odds,
            'available_liquidity': self.available_liquidity,
            'recent_activity': []
        }
        
        for bet in recent_bets:
            depth_data['recent_activity'].append({
                'amount': bet.credits_staked,
                'odds': bet.odds,
                'timestamp': bet.created_at.isoformat()
            })
        
        return depth_data


class MarketOrder(models.Model):
    """Market order model for advanced trading functionality"""
    ORDER_TYPES = [
        ('MARKET', 'Market Order'),
        ('LIMIT', 'Limit Order'),
        ('STOP', 'Stop Order'),
    ]
    
    SIDE_CHOICES = [
        ('BUY', 'Buy'),
        ('SELL', 'Sell'),
    ]
    
    STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('FILLED', 'Filled'),
        ('CANCELLED', 'Cancelled'),
        ('REJECTED', 'Rejected'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='market_orders')
    market_maker = models.ForeignKey(MarketMaker, on_delete=models.CASCADE)
    
    order_type = models.CharField(max_length=10, choices=ORDER_TYPES)
    side = models.CharField(max_length=4, choices=SIDE_CHOICES)
    amount = models.IntegerField(help_text="Amount in credits")
    limit_price = models.FloatField(null=True, blank=True, help_text="Limit price for limit orders")
    stop_price = models.FloatField(null=True, blank=True, help_text="Stop price for stop orders")
    
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='PENDING')
    filled_amount = models.IntegerField(default=0)
    average_price = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    filled_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'status']),
            models.Index(fields=['market_maker', 'side']),
        ]
    
    def __str__(self):
        return f"{self.user.username}: {self.side} {self.amount} @ {self.market_maker}"
    
    def execute_market_order(self):
        """Execute a market order"""
        if self.order_type != 'MARKET':
            return False
        
        # Get current market odds
        odds = self.market_maker.calculate_adjusted_odds(self.amount)
        
        if self.side == 'BUY':
            execution_price = odds['ask']
        else:  # SELL
            execution_price = odds['bid']
        
        # Check if we have enough liquidity
        if self.amount > self.market_maker.available_liquidity:
            self.status = 'REJECTED'
            self.save()
            return False
        
        # Execute the order
        self.filled_amount = self.amount
        self.average_price = execution_price
        self.status = 'FILLED'
        self.filled_at = timezone.now()
        self.save()
        
        # Update market maker state
        self.market_maker.update_market_state(self.amount)
        
        return True