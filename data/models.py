from django.db import models

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
