from django.contrib import admin
from .models import (
    Event, Circuit, Team, Driver,
    QualifyingResult, RaceResult,
    Session, SessionType,
    DriverPerformance, TeamPerformance, TrackCharacteristics,
    PredictionModel, RacePrediction,ridgeregression
)

@admin.register(Event)
class EventAdmin(admin.ModelAdmin):
    list_display = ('year', 'round', 'name', 'circuit', 'event_format')
    search_fields = ('name',)
    list_filter = ('year', 'event_format', 'circuit')

@admin.register(Circuit)
class CircuitAdmin(admin.ModelAdmin):
    list_display = ('name', 'country', 'circuit_type', 'circuit_ref')
    search_fields = ('name', 'country')

@admin.register(Team)
class TeamAdmin(admin.ModelAdmin):
    list_display = ('name', 'team_ref', 'season_dnf_rate', 'pit_stop_avg')

@admin.register(Driver)
class DriverAdmin(admin.ModelAdmin):
    list_display = ('given_name', 'family_name', 'code', 'nationality', 'recent_form', 'consistency_score')
    search_fields = ('given_name', 'family_name', 'code')

@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    list_display = ('event', 'session_type', 'date')
    list_filter = ('session_type',)

@admin.register(SessionType)
class SessionTypeAdmin(admin.ModelAdmin):
    list_display = ('name', 'session_type')

@admin.register(QualifyingResult)
class QualifyingResultAdmin(admin.ModelAdmin):
    list_display = ('session', 'driver', 'team', 'position', 'pole_delta')
    list_filter = ('session__event__year',)

@admin.register(RaceResult)
class RaceResultAdmin(admin.ModelAdmin):
    list_display = ('session', 'driver', 'team', 'position', 'grid_position', 'position_gain', 'points')

@admin.register(DriverPerformance)
class DriverPerformanceAdmin(admin.ModelAdmin):
    list_display = ('driver', 'event', 'moving_avg_5', 'qualifying_avg', 'points_per_race')

@admin.register(TeamPerformance)
class TeamPerformanceAdmin(admin.ModelAdmin):
    list_display = ('team', 'event', 'dnf_rate', 'pit_stop_avg', 'reliability_score')

@admin.register(TrackCharacteristics)
class TrackCharacteristicsAdmin(admin.ModelAdmin):
    list_display = ('circuit', 'overtaking_index', 'safety_car_probability', 'rain_impact')

@admin.register(PredictionModel)
class PredictionModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'version', 'model_type', 'circuit', 'is_active', 'created_at')

@admin.register(RacePrediction)
class RacePredictionAdmin(admin.ModelAdmin):
    list_display = ('event', 'session', 'model', 'predicted_at', 'top_3_accuracy')


admin.site.register(ridgeregression)