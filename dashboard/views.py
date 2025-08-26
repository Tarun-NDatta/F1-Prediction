from django.shortcuts import get_object_or_404, render, get_list_or_404, redirect
from django.contrib.auth.decorators import login_required
from dashboard.decorators import subscription_required
from prediction.analysis.chaos import analyze_prediction_errors_by_events, CounterfactualAnalyzer, analyze_event
import numpy as np
import math
from django.http import JsonResponse


from django.db.models import Count, Avg, Q, Max, Sum, CharField, Value
from django.db.models.functions import Concat

from data import models
from data.models import (
    Event, RaceResult, QualifyingResult, Circuit, Team, Driver,
    Session, SessionType, ridgeregression, xgboostprediction, CatBoostPrediction,
    UserProfile, CreditTransaction, Bet, Achievement, TrackSpecialization, MarketMaker, MarketOrder
)
from django.core.paginator import Paginator
from django.http import JsonResponse
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib import messages
from django.contrib.auth import login as auth_login, authenticate, logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django import forms
from django.contrib.auth.models import User
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.conf import settings
from django.utils import timezone

import random
import string
from datetime import datetime, timedelta
import json
import logging
from .decorators import subscription_required, get_user_model_context
from .models import RaceIncident

# Set up logging to help debug
logger = logging.getLogger(__name__)

class F1UserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True, label='Email')

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
        return user

class F1AuthenticationForm(AuthenticationForm):
    username = forms.CharField(label="Username or Email")

    def clean(self):
        username_or_email = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')
        if username_or_email and password:
            # Try to authenticate by username
            user = authenticate(self.request, username=username_or_email, password=password)
            if not user:
                # Try to authenticate by email
                try:
                    user_obj = User.objects.get(email=username_or_email)
                    user = authenticate(self.request, username=user_obj.username, password=password)
                except User.DoesNotExist:
                    pass
            if not user:
                raise forms.ValidationError('Invalid username/email or password.')
            self.confirm_login_allowed(user)
            self.user_cache = user
        return self.cleaned_data

token_generator = PasswordResetTokenGenerator()

def home(request):
    return render(request, 'home.html')


def results(request):
    # Get filter parameters
    selected_year = request.GET.get('year', '2025')
    selected_circuit = request.GET.get('circuit', 'all')
    selected_session = request.GET.get('session', 'race')
    page_number = request.GET.get('page', 1)

    # Convert year to integer with validation
    try:
        year_int = int(selected_year)
        if year_int not in [2022, 2023, 2024, 2025]:  # Now includes 2025
            year_int = 2025
            selected_year = '2025'
    except (ValueError, TypeError):
        year_int = 2025
        selected_year = '2025'

    # Get available years from database (dynamic)
    available_years = sorted(Event.objects.values_list('year', flat=True).distinct(), reverse=True)
    available_circuits = Circuit.objects.all().order_by('name')

    # Build base queryset for events
    events_query = Event.objects.filter(year=year_int).select_related('circuit')

    # Filter by circuit if specified
    if selected_circuit != 'all':
        try:
            circuit_id = int(selected_circuit)
            events_query = events_query.filter(circuit_id=circuit_id)
        except (ValueError, TypeError):
            pass

    # Order events by round
    events = events_query.order_by('round')

    # Prepare results data
    results_data = []

    for event in events:
        try:
            # Get session type based on selection
            SESSION_TYPE_MAP = {
                'race': 'RACE',
                'qualifying': 'QUALIFYING',
                # add others if needed
            }
            session_type_code = SESSION_TYPE_MAP.get(selected_session, 'RACE')

            # Get the session with error handling
            try:
                session = Session.objects.get(
                    event=event,
                    session_type__session_type=session_type_code
                )
            except Session.DoesNotExist:
                continue  # Skip events without this session type

            # Get results based on session type
            if selected_session == 'qualifying':
                session_results = QualifyingResult.objects.filter(
                    session=session
                ).select_related('driver', 'team').order_by('position')
                total_laps = 0
            else:
                session_results = RaceResult.objects.filter(
                    session=session
                ).select_related('driver', 'team').order_by('position')
                total_laps = session_results.aggregate(Max('laps'))['laps__max'] or 0

            # Process results
            processed_results = []
            for result in session_results:
                result_data = {
                    'position': result.position,
                    'driver_name': f"{result.driver.given_name} {result.driver.family_name}",
                    'driver_number': result.driver.permanent_number or '',
                    'team_name': result.team.name,
                    'team_class': f"team-{result.team.name.lower().replace(' ', '')}",
                    'time_or_status': format_result_time(result, selected_session),
                    'points': result.points or 0,
                    'is_podium': result.position <= 3 if result.position else False,
                    'is_dnf': result.status and ('DNF' in result.status or 'Retired' in result.status),
                    'fastest_lap': getattr(result, 'fastest_lap_rank', None) == 1,
                    'grid_position': getattr(result, 'grid_position', None),
                    'laps': result.laps if result.laps is not None else total_laps
                }
                processed_results.append(result_data)

            # Create race data structure
            race_data = {
                'event': event,
                'session_type': selected_session,
                'results': processed_results,
                'total_laps': total_laps,
                'weather': {
                    'air_temp': session.air_temp,
                    'rain': session.rain
                }
            }
            results_data.append(race_data)

        except Exception as e:
            print(f"Error processing event {event}: {str(e)}")
            continue

    # Pagination
    paginator = Paginator(results_data, 5)  # 5 races per page
    try:
        page_obj = paginator.page(page_number)
    except:
        page_obj = paginator.page(1)

    context = {
        'page_obj': page_obj,
        'available_years': available_years,
        'available_circuits': available_circuits,
        'selected_year': selected_year,
        'selected_circuit': selected_circuit,
        'selected_session': selected_session,
        'error_message': None if results_data else f"No {selected_session} data found for {selected_year}",
        'is_authenticated': request.user.is_authenticated,
    }

    return render(request, 'results.html', context)

# Helper functions
def format_result_time(result, session_type):
    """Format time or status for display"""
    if session_type == 'qualifying':
        if result.q3:
            return format_duration(result.q3)
        elif result.q2:
            return format_duration(result.q2)
        elif result.q1:
            return format_duration(result.q1)
    else:  # race
        # Always show status text when available (Retired, Lapped, Fuel pressure, etc.)
        if result.status:
            return result.status
        elif result.time:
            return format_duration(result.time)

    return result.status or 'No Time'

def format_duration(duration):
    """Format duration object to readable string"""
    if not duration:
        return 'No Time'

    try:
        total_seconds = duration.total_seconds()
        minutes = int(total_seconds // 60)
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:06.3f}"
    except (AttributeError, TypeError):
        return str(duration)

def get_team_class(team_name):
    """Map team names to CSS classes"""
    if not team_name:
        return 'team-default'

    team_mapping = {
        'Mercedes': 'team-mercedes',
        'Ferrari': 'team-ferrari',
        'Red Bull': 'team-redbull',
        'Red Bull Racing': 'team-redbull',
        'McLaren': 'team-mclaren',
        'Alpine': 'team-alpine',
        'Aston Martin': 'team-astonmartin',
        'Alfa Romeo': 'team-alfaromeo',
        'Haas': 'team-haas',
        'AlphaTauri': 'team-alphatauri',
        'Williams': 'team-williams',
    }

    for team_key, css_class in team_mapping.items():
        if team_key.lower() in team_name.lower():
            return css_class

    return 'team-default'


def prediction(request):
    predictions_locked = not request.user.is_authenticated

    # Get user's subscription context
    model_context = get_user_model_context(request.user)
    user_available_models = model_context.get('available_models', ['ridge_regression'])

    # Available models configuration - UPDATED to include CatBoost
    AVAILABLE_MODELS = {
        'ridge_regression': {
            'name': 'Ridge Regression',
            'model_class': ridgeregression,
            'available_rounds': list(range(1, 15)),  # Rounds 1-14 (completed) + upcoming
            'status': 'active',
            'model_name_filter': 'ridge_regression',
            'tier_required': 'BASIC'
        },
        'xgboost': {
            'name': 'XGBoost',
            'model_class': xgboostprediction,
            'available_rounds': list(range(1, 15)),
            'status': 'active',
            'model_name_filter': 'xgboost_regression',
            'tier_required': 'PREMIUM'
        },
        'catboost': {
            'name': 'CatBoost Ensemble',
            'model_class': CatBoostPrediction,
            'available_rounds': list(range(1, 15)),
            'status': 'active',
            'model_name_filter': 'catboost_ensemble',
            'tier_required': 'PRO'
        }
    }

    # All models available in system
    available_models_for_template = AVAILABLE_MODELS

    # Show all models to everyone (no subscription gating in UI)
    subscription_tier = model_context.get('subscription_tier', 'BASIC')
    visible_models_map = available_models_for_template

    # Get selected model from request (default to ridge_regression)
    selected_model_key = request.GET.get('model', 'ridge_regression')
    if selected_model_key not in AVAILABLE_MODELS:
        selected_model_key = 'ridge_regression'

    # Ensure selected model key is valid; otherwise default to ridge_regression
    if selected_model_key not in AVAILABLE_MODELS:
        selected_model_key = 'ridge_regression'

    selected_model = AVAILABLE_MODELS[selected_model_key]

    # Get all events for 2025 season, ordered by round
    all_events = Event.objects.filter(year=2025).select_related('circuit').order_by('round')

    # Get available rounds for selected model dynamically from DB
    available_rounds_qs = []
    try:
        model_cls = selected_model['model_class']
        if model_cls is not None:
            qs = model_cls.objects.filter(year=2025)
            if hasattr(model_cls, 'model_name'):
                qs = qs.filter(model_name__startswith=selected_model['model_name_filter'])
            available_rounds_qs = list(qs.values_list('round_number', flat=True).distinct())
    except Exception:
        available_rounds_qs = []

    available_rounds = sorted(set(available_rounds_qs))

    # Create display string for available rounds
    if available_rounds:
        available_rounds_display = f"{min(available_rounds)}-{max(available_rounds)}"
    else:
        available_rounds_display = "Coming Soon"

    # Get all drivers and teams that participate in the season for dynamic updates
    season_drivers = Driver.objects.filter(
        raceresult__session__event__year=2025
    ).distinct().select_related()

    season_teams = Team.objects.filter(
        raceresult__session__event__year=2025
    ).distinct()

    # Calculate cumulative points for drivers and teams up to each race
    def get_cumulative_points(event_round):
        # Get all events up to and including the current round
        completed_events = Event.objects.filter(
            year=2025,
            round__lte=event_round
        ).values_list('id', flat=True)

        # Driver points
        driver_points = {}
        for driver in season_drivers:
            total_points = RaceResult.objects.filter(
                session__event_id__in=completed_events,
                driver=driver
            ).aggregate(total=Sum('points'))['total'] or 0
            driver_points[driver.id] = total_points

        # Team points
        team_points = {}
        for team in season_teams:
            total_points = RaceResult.objects.filter(
                session__event_id__in=completed_events,
                team=team
            ).aggregate(total=Sum('points'))['total'] or 0
            team_points[team.id] = total_points

        return driver_points, team_points

    # Prepare data for all races - RENAMED from races_data to results
    results = []

    # Consistency metrics containers
    per_race_labels = []
    per_race_mae = []
    global_top10_hits = 0
    global_top10_misses = 0

    # Simple incident notes for races (static annotations to explain anomalies)
    incident_notes_map = {
        # Use event.name as key (ensure it matches your DB names)
        'British Grand Prix': [
            'Penalty for Oscar Piastri impacted finishing position',
            'Max Verstappen had a poor restart impacting race pace'
        ],
        'Monaco Grand Prix': [
            'Track position crucial; limited overtaking skewed results vs. pace'
        ],
        'Canadian Grand Prix': [
            'Safety car timing reshuffled the order significantly'
        ],
    }

    # Merge DB incidents if present
    try:
        for evt in all_events:
            db_incidents = RaceIncident.objects.filter(year=evt.year, round=evt.round, event_name=evt.name)
            if db_incidents.exists():
                incident_notes_map[evt.name] = [inc.description for inc in db_incidents.order_by('lap', 'driver_name')]
    except Exception:
        pass

    for event in all_events:
        # Get cumulative points up to this race
        driver_cumulative_points, team_cumulative_points = get_cumulative_points(event.round)

        # Get predictions and actual results for this event
        if selected_model['model_class'] is not None:
            # Updated query to use model_name_filter for better filtering
            predictions_qs = selected_model['model_class'].objects.filter(
                event=event,
                year=2025,
                round_number=event.round
            )
            # Additional filtering by model_name if the field exists
            if hasattr(selected_model['model_class'], 'model_name'):
                predictions_qs = predictions_qs.filter(model_name__startswith=selected_model['model_name_filter'])

            predictions = list(predictions_qs)
        else:
            predictions = []

        # Rank-by-score normalization: compute predicted ranks per event
        predicted_rank_map = {}
        try:
            scores = sorted(
                [(p.driver_id, float(p.predicted_position)) for p in predictions],
                key=lambda t: t[1]
            )
            for idx, (driver_id, _) in enumerate(scores, start=1):
                predicted_rank_map[driver_id] = idx
        except Exception:
            predicted_rank_map = {}

        # Get actual results for this event (RaceResult)
        actuals_qs = RaceResult.objects.filter(session__event=event).select_related('driver', 'team')
        actuals = list(actuals_qs) if actuals_qs else []

        # Build a set of all drivers who have either a prediction or an actual result
        driver_ids = set()
        for pred in predictions:
            driver_ids.add(pred.driver_id)
        for act in actuals:
            driver_ids.add(act.driver_id)

        # Bias correction for CatBoost/XGBoost using track specialization (if available)
        try:
            if selected_model_key in ('catboost', 'xgboost'):
                track_spec = TrackSpecialization.objects.filter(circuit=event.circuit).first()
                if track_spec:
                    # Higher qualifying importance tends to reduce race-order variance; nudge predictions slightly
                    q_importance = float(track_spec.qualifying_importance or 0.0)  # 0..10
                    q_bias = (5.0 - q_importance) / 50.0  # small adjustment in ranks (roughly -0.1..+0.1)

                    # Power sensitivity and overtaking difficulty increase spread; apply tiny offsets
                    power_bias = float(track_spec.power_sensitivity or 0.0) / 200.0
                    overtake_bias = float(track_spec.overtaking_difficulty or 0.0) / 200.0
                    total_bias = q_bias + power_bias + overtake_bias

                    # Apply to predicted_rank_map then re-normalize to integer ranks
                    if predicted_rank_map:
                        # convert to float ranks, add bias, then re-sort back to 1..N
                        adjusted = []
                        for d_id, rnk in predicted_rank_map.items():
                            adjusted.append((d_id, float(rnk) + total_bias))
                        adjusted.sort(key=lambda t: t[1])
                        predicted_rank_map = {d_id: idx for idx, (d_id, _) in enumerate(adjusted, start=1)}
        except Exception:
            pass

        # Build a mapping for quick lookup
        pred_map = {p.driver_id: p for p in predictions}
        act_map = {a.driver_id: a for a in actuals}

        comparison = []
        show_coming_soon = len(predictions) == 0

        for driver_id in driver_ids:
            pred = pred_map.get(driver_id)
            act = act_map.get(driver_id)

            if pred and act:
                # Both prediction and actual result available
                is_correct = pred.predicted_position == act.position
                difference = pred.predicted_position - act.position

                # Get team color for display
                team_color = act.team.color if hasattr(act.team, 'color') and act.team.color else '#666666'

                # Determine confidence if available
                conf_val = None
                if hasattr(pred, 'prediction_confidence') and pred.prediction_confidence is not None:
                    conf_val = pred.prediction_confidence
                elif hasattr(pred, 'confidence') and pred.confidence is not None:
                    conf_val = pred.confidence
                else:
                    # Fallback heuristic confidence (for CatBoost) using base model agreement if available
                    try:
                        if hasattr(pred, 'ridge_prediction') and hasattr(pred, 'xgboost_prediction') \
                           and pred.ridge_prediction is not None and pred.xgboost_prediction is not None:
                            dispersion = abs(float(pred.ridge_prediction) - float(pred.xgboost_prediction))
                            conf_val = max(0.1, min(0.99, 1.0 - (dispersion / 10.0)))
                    except Exception:
                        conf_val = None

                comparison_item = {
                    'driver': f"{act.driver.given_name} {act.driver.family_name}",
                    'team': act.team.name,
                    'team_color': team_color,
                    'predicted': predicted_rank_map.get(driver_id, int(round(pred.predicted_position))),
                    'actual': act.position,
                    'difference': f"{int(difference):+d}" if difference != 0 else "0",
                    'confidence': f"{conf_val:.1%}" if conf_val is not None else "N/A",
                    'is_correct': is_correct,
                    'points': act.points or 0,
                }

                # Add CatBoost-specific features if available
                if selected_model_key == 'catboost' and hasattr(pred, 'track_category'):
                    comparison_item['catboost_features'] = {
                        'track_category': getattr(pred, 'track_category', 'N/A'),
                        'power_sensitivity': getattr(pred, 'track_power_sensitivity', 'N/A'),
                        'overtaking_difficulty': getattr(pred, 'track_overtaking_difficulty', 'N/A'),
                    }
                else:
                    comparison_item['catboost_features'] = {
                        'track_category': 'N/A',
                        'power_sensitivity': 'N/A',
                        'overtaking_difficulty': 'N/A',
                    }

                comparison.append(comparison_item)
            elif pred:
                # Only prediction available (future race)
                conf_val = None
                if hasattr(pred, 'prediction_confidence') and pred.prediction_confidence is not None:
                    conf_val = pred.prediction_confidence
                elif hasattr(pred, 'confidence') and pred.confidence is not None:
                    conf_val = pred.confidence
                else:
                    try:
                        if hasattr(pred, 'ridge_prediction') and hasattr(pred, 'xgboost_prediction') \
                           and pred.ridge_prediction is not None and pred.xgboost_prediction is not None:
                            dispersion = abs(float(pred.ridge_prediction) - float(pred.xgboost_prediction))
                            conf_val = max(0.1, min(0.99, 1.0 - (dispersion / 10.0)))
                    except Exception:
                        conf_val = None
                comparison_item = {
                    'driver': f"{pred.driver.given_name} {pred.driver.family_name}",
                    'team': pred.team.name if hasattr(pred, 'team') and pred.team else "N/A",
                    'team_color': '#666666',
                    'predicted': predicted_rank_map.get(pred.driver_id, int(round(pred.predicted_position))),
                    'actual': 'N/A',
                    'difference': 'N/A',
                    'confidence': f"{conf_val:.1%}" if conf_val is not None else "N/A",
                    'is_correct': None,
                    'points': 0,
                }

                if selected_model_key == 'catboost' and hasattr(pred, 'track_category'):
                    comparison_item['catboost_features'] = {
                        'track_category': getattr(pred, 'track_category', 'N/A'),
                        'power_sensitivity': getattr(pred, 'track_power_sensitivity', 'N/A'),
                        'overtaking_difficulty': getattr(pred, 'track_overtaking_difficulty', 'N/A'),
                    }
                else:
                    comparison_item['catboost_features'] = {
                        'track_category': 'N/A',
                        'power_sensitivity': 'N/A',
                        'overtaking_difficulty': 'N/A',
                    }

                comparison.append(comparison_item)

        # Sort comparison by actual position (if available), otherwise by predicted position
        comparison.sort(key=lambda x: (x['actual'] if x['actual'] != 'N/A' else float('inf'), x['predicted']))

        # Consistency metrics: per-race MAE and global Top-10 hit/miss
        try:
            if not show_coming_soon:
                # per-race MAE for selected model
                diffs = [abs(int(item['predicted']) - int(item['actual'])) for item in comparison if item['actual'] != 'N/A' and isinstance(item['predicted'], (int, float, str))]
                if diffs:
                    per_race_labels.append(event.name)
                    per_race_mae.append(round(sum(diffs) / len(diffs), 2))
                # top-10 vs out-of-top-10
                for item in comparison:
                    if item['actual'] != 'N/A':
                        if int(item['predicted']) <= 10 and int(item['actual']) <= 10:
                            global_top10_hits += 1
                        elif int(item['predicted']) <= 10 and int(item['actual']) > 10:
                            global_top10_misses += 1
        except Exception:
            pass

        # Track specialization once per race (for CatBoost summary)
        track_info = None
        try:
            track_spec = TrackSpecialization.objects.filter(circuit=event.circuit).first()
            if track_spec:
                track_info = {
                    'track_category': str(track_spec.category),
                    'track_power_sensitivity': float(track_spec.power_sensitivity or 0),
                    'track_overtaking_difficulty': float(track_spec.overtaking_difficulty or 0),
                    'track_qualifying_importance': float(track_spec.qualifying_importance or 0),
                }
        except Exception:
            track_info = None

        race_data = {
            'event': event,
            'comparison': comparison,
            'show_coming_soon': show_coming_soon,
            'track_info': track_info,
            'incident_notes': incident_notes_map.get(event.name, []),
        }
        results.append(race_data)

    # Prepare chart data for visualization
    chart_data = []
    for race_data in results:
        for item in race_data['comparison']:
            if item['actual'] != 'N/A':  # Only include races with actual results
                chart_data.append({
                    'predicted': item['predicted'],
                    'actual': item['actual'],
                    'driver': item['driver'],
                    'race': race_data['event'].name,
                    'isCorrect': item['is_correct']
                })

    # Calculate model accuracy statistics
    model_stats = calculate_model_accuracy_stats(selected_model['model_class'], selected_model['model_name_filter'])

    # Debug: Print chart data to see what's being passed
    print(f"Chart data length: {len(chart_data)}")
    print(f"Chart data sample: {chart_data[:2] if chart_data else 'Empty'}")

    # Add subscription context
    # Build model comparison stats (Ridge, XGBoost, CatBoost) for charts
    model_defs = [
        ('Ridge Regression', ridgeregression, 'ridge_regression'),
        ('XGBoost', xgboostprediction, 'xgboost_regression'),
        ('CatBoost Ensemble', CatBoostPrediction, 'catboost_ensemble')
    ]
    comparison_labels = []
    comparison_mae = []
    comparison_top3 = []
    comparison_top10 = []
    for label, cls, name_filter in model_defs:
        stats = calculate_model_accuracy_stats(cls, name_filter)
        comparison_labels.append(label)
        if stats:
            comparison_mae.append(round(stats.get('mae', 0), 3))
            comparison_top3.append(round(stats.get('top_3_accuracy', 0), 1))
            comparison_top10.append(round(stats.get('top_10_accuracy', 0), 1))
        else:
            comparison_mae.append(0)
            comparison_top3.append(0)
            comparison_top10.append(0)

    context = {
        'results': results,
        'available_rounds': available_rounds,
        'available_rounds_display': available_rounds_display,
        'total_races': len(all_events),
        'races_with_predictions': len(available_rounds),
        'model_name': selected_model['name'],
        'selected_model_key': selected_model_key,
        'available_models_map': available_models_for_template,
        'visible_models_map': visible_models_map,
        'model_status': selected_model['status'],
        'error': None,
        'predictions_locked': predictions_locked,
        'is_authenticated': request.user.is_authenticated,
        'model_stats': model_stats,
        'chart_data_json': json.dumps(chart_data),
        'chart_data': chart_data,  # Add raw chart data for debugging
        'comparison_labels': json.dumps(comparison_labels),
        'comparison_mae': json.dumps(comparison_mae),
        'comparison_top3': json.dumps(comparison_top3),
        'comparison_top10': json.dumps(comparison_top10),
        'per_race_labels': json.dumps(per_race_labels),
        'per_race_mae': json.dumps(per_race_mae),
        'top10_hits': global_top10_hits,
        'top10_misses': global_top10_misses,
        **model_context,  # Add subscription context
    }

    return render(request, 'prediction.html', context)

def calculate_model_accuracy_stats(model_class, model_name_filter):
    """Calculate accuracy statistics for a given model"""
    if not model_class:
        return None

    try:
        # Get all predictions with actual results
        predictions = model_class.objects.filter(
            actual_position__isnull=False,
            year=2025
        )

        if hasattr(model_class, 'model_name'):
            predictions = predictions.filter(model_name=model_name_filter)

        if not predictions.exists():
            return None

        # Calculate metrics
        total_predictions = predictions.count()
        correct_predictions = predictions.filter(predicted_position=models.F('actual_position')).count()
        overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

        # Calculate MAE
        mae = 0
        top_3_correct = 0
        top_10_correct = 0
        top_3_total = 0
        top_10_total = 0

        for pred in predictions:
            mae += abs(pred.predicted_position - pred.actual_position)

            # Top 3 accuracy (predictions within 3 positions)
            if abs(pred.predicted_position - pred.actual_position) <= 3:
                top_3_correct += 1
            top_3_total += 1

            # Top 10 accuracy (predictions within 10 positions)
            if abs(pred.predicted_position - pred.actual_position) <= 10:
                top_10_correct += 1
            top_10_total += 1

        mae = mae / total_predictions if total_predictions > 0 else 0
        top_3_accuracy = top_3_correct / top_3_total if top_3_total > 0 else 0
        top_10_accuracy = top_10_correct / top_10_total if top_10_total > 0 else 0

        return {
            'total_predictions': total_predictions,
            'overall_accuracy': overall_accuracy * 100,  # Convert to percentage
            'mae': mae,
            'top_3_accuracy': top_3_accuracy * 100,  # Convert to percentage
            'top_10_accuracy': top_10_accuracy * 100,  # Convert to percentage
        }

    except Exception as e:
        print(f"Error calculating model stats: {e}")
        return None


def driver_analytics(request):
    """Driver analytics page with charts and comparisons"""
    # Get current year (default to 2025)
    current_year = request.GET.get('year', '2025')
    try:
        year_int = int(current_year)
    except (ValueError, TypeError):
        year_int = 2025

    # Get events for the selected year
    events = Event.objects.filter(year=year_int).order_by('round')
    event_ids = events.values_list('id', flat=True)

    # Get all drivers with their results
    drivers = Driver.objects.filter(
        raceresult__session__event_id__in=event_ids
    ).distinct()

    # Get driver standings data for charts
    driver_standings_data = []
    for driver in drivers:
        driver_results = RaceResult.objects.filter(
            session__event_id__in=event_ids,
            driver=driver
        ).order_by('session__event__round')

        if driver_results.exists():
            points_progression = []
            cumulative_points = 0

            for result in driver_results:
                cumulative_points += result.points or 0
                points_progression.append({
                    'round': result.session.event.round,
                    'race_name': result.session.event.name,
                    'points': result.points or 0,
                    'cumulative_points': cumulative_points,
                    'position': result.position,
                })

            driver_standings_data.append({
                'driver': driver,
                'points_progression': points_progression,
                'total_points': cumulative_points,
                'team': driver_results.first().team,
            })

    # Sort by total points
    driver_standings_data.sort(key=lambda x: x['total_points'], reverse=True)

    # Get teammate comparisons
    teammate_comparisons = []
    teams = Team.objects.filter(
        raceresult__session__event_id__in=event_ids
    ).distinct()

    for team in teams:
        team_drivers = Driver.objects.filter(
            raceresult__session__event_id__in=event_ids,
            raceresult__team=team
        ).distinct()

        if team_drivers.count() >= 2:
            driver1, driver2 = team_drivers[:2]

            driver1_points = RaceResult.objects.filter(
                session__event_id__in=event_ids,
                driver=driver1
            ).aggregate(total_points=models.Sum('points'))['total_points'] or 0

            driver2_points = RaceResult.objects.filter(
                session__event_id__in=event_ids,
                driver=driver2
            ).aggregate(total_points=models.Sum('points'))['total_points'] or 0

            teammate_comparisons.append({
                'team': team,
                'driver1': {
                    'driver': driver1,
                    'points': driver1_points,
                },
                'driver2': {
                    'driver': driver2,
                    'points': driver2_points,
                },
                'difference': abs(driver1_points - driver2_points),
            })

    context = {
        'driver_standings_data': driver_standings_data,
        'teammate_comparisons': teammate_comparisons,
        'events': events,
        'selected_year': year_int,
    }

    return render(request, 'driver_analytics.html', context)


def team_analytics(request):
    """Team analytics page with race-by-race progression"""
    # Get current year (default to 2025)
    current_year = request.GET.get('year', '2025')
    try:
        year_int = int(current_year)
    except (ValueError, TypeError):
        year_int = 2025

    # Get events for the selected year
    events = Event.objects.filter(year=year_int).order_by('round')
    event_ids = events.values_list('id', flat=True)

    # Get all teams with their results
    teams = Team.objects.filter(
        raceresult__session__event_id__in=event_ids
    ).distinct()

    # Get team progression data for charts
    team_progression_data = []
    for team in teams:
        team_results = RaceResult.objects.filter(
            session__event_id__in=event_ids,
            team=team
        ).order_by('session__event__round')

        if team_results.exists():
            points_progression = []
            cumulative_points = 0

            # Group by event to get team points per race
            for event in events:
                event_results = team_results.filter(session__event=event)
                event_points = sum(result.points or 0 for result in event_results)
                cumulative_points += event_points

                points_progression.append({
                    'round': event.round,
                    'race_name': event.name,
                    'points': event_points,
                    'cumulative_points': cumulative_points,
                })

            team_progression_data.append({
                'team': team,
                'points_progression': points_progression,
                'total_points': cumulative_points,
            })

    # Sort by total points
    team_progression_data.sort(key=lambda x: x['total_points'], reverse=True)

    context = {
        'team_progression_data': team_progression_data,
        'events': events,
        'selected_year': year_int,
    }

    return render(request, 'team_analytics.html', context)


def standings(request):
    # Get current year (default to 2025)
    current_year = request.GET.get('year', '2025')
    try:
        year_int = int(current_year)
    except (ValueError, TypeError):
        year_int = 2025

    # Get selected round (default to all completed races)
    selected_round = request.GET.get('round', 'all')

    # Get all events for the selected year
    events_query = Event.objects.filter(year=year_int).order_by('round')

    # Filter by round if specified
    if selected_round != 'all':
        try:
            round_int = int(selected_round)
            events_query = events_query.filter(round__lte=round_int)
        except (ValueError, TypeError):
            pass

    event_ids = events_query.values_list('id', flat=True)
    all_events = events_query

    # Driver standings: sum points for each driver (official cumulative)
    driver_points = (
        RaceResult.objects
        .filter(session__event_id__in=event_ids)
        .values('driver', 'driver__given_name', 'driver__family_name', 'driver__permanent_number', 'team__name')
        .annotate(points=Sum('points'))
        .order_by('-points', 'driver__family_name', 'driver__given_name')
    )
    driver_standings = []
    for idx, d in enumerate(driver_points, 1):
        driver_standings.append({
            'position': idx,
            'number': d['driver__permanent_number'],
            'name': f"{d['driver__given_name']} {d['driver__family_name']}",
            'team': d['team__name'],
            'points': d['points'] or 0,
        })

    # Team standings: sum points for each team (constructors)
    team_points = (
        RaceResult.objects
        .filter(session__event_id__in=event_ids)
        .values('team', 'team__name')
        .annotate(points=Sum('points'))
        .order_by('-points', 'team__name')
    )
    team_standings = []
    for idx, t in enumerate(team_points, 1):
        team_standings.append({
            'position': idx,
            'name': t['team__name'],
            'points': t['points'] or 0,
        })

    # Get recent events for dynamic updates
    recent_events = Event.objects.filter(year=year_int).order_by('-date')[:5]

    # Get available rounds for dropdown
    available_rounds = Event.objects.filter(year=year_int).order_by('round').values_list('round', 'name')

    context = {
        'driver_standings': driver_standings,
        'team_standings': team_standings,
        'selected_year': year_int,
        'selected_round': selected_round,
        'recent_events': recent_events,
        'available_rounds': available_rounds,
    }
    return render(request, 'standings.html', context)

def register(request):
    next_url = request.GET.get('next') or request.POST.get('next')
    if request.method == 'POST':
        form = F1UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.is_active = False
            user.save()
            # Send verification email
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = token_generator.make_token(user)
            activation_link = request.build_absolute_uri(
                f"/activate/{uid}/{token}/?next={next_url or ''}"
            )
            subject = 'Activate your F1 Dashboard account'
            message = render_to_string('activation_email.txt', {
                'user': user,
                'activation_link': activation_link,
            })
            send_mail(subject, message, settings.DEFAULT_FROM_EMAIL, [user.email])
            return render(request, 'registration_pending.html', {'email': user.email})
    else:
        form = F1UserCreationForm()
    return render(request, 'registration.html', {'form': form, 'next': next_url})

def activate(request, uidb64, token):
    next_url = request.GET.get('next')
    try:
        uid = force_str(urlsafe_base64_decode(uidb64))
        user = User.objects.get(pk=uid)
    except (TypeError, ValueError, OverflowError, User.DoesNotExist):
        user = None
    if user is not None and token_generator.check_token(user, token):
        user.is_active = True
        user.save()
        auth_login(request, user)
        if next_url:
            return redirect(next_url)
        return redirect('home')
    else:
        return render(request, 'activation_invalid.html')

def login_view(request):
    error = None
    next_url = request.GET.get('next') or request.POST.get('next')
    if request.method == 'POST':
        form = F1AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            auth_login(request, user)
            if next_url:
                return redirect(next_url)
            return redirect('home')
        else:
            error = 'Invalid username or password.'
    else:
        form = F1AuthenticationForm()
    return render(request, 'login.html', {'form': form, 'error': error, 'next': next_url})

def logout_view(request):
    auth_logout(request)
    return redirect('home')

@login_required
def subscription_management(request):
    """Subscription management page"""
    try:
        profile = request.user.profile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'upgrade':
            new_tier = request.POST.get('tier')
            if new_tier in ['PREMIUM', 'PRO']:
                profile.subscription_tier = new_tier
                profile.subscription_start_date = timezone.now()
                # Set end date to 30 days from now (for demo purposes)
                profile.subscription_end_date = timezone.now() + timedelta(days=30)
                profile.is_subscription_active = True
                profile.save()

                messages.success(request, f'Successfully upgraded to {new_tier} tier!')

                # Create credit transaction for upgrade
                CreditTransaction.objects.create(
                    user=request.user,
                    transaction_type='ADMIN_ADJUSTMENT',
                    amount=0,  # No credits involved in subscription
                    description=f'Subscription upgraded to {new_tier}',
                    balance_after=profile.credits
                )

        elif action == 'cancel':
            profile.subscription_tier = 'BASIC'
            profile.is_subscription_active = True
            profile.subscription_end_date = None
            profile.save()

            messages.info(request, 'Subscription cancelled. You now have Basic access.')

        return redirect('subscription_management')

    # Get all tier information
    tier_info = {
        'BASIC': profile.get_subscription_display_info() if profile.subscription_tier == 'BASIC' else {
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

    context = {
        'profile': profile,
        'tier_info': tier_info,
        'current_tier_info': profile.get_subscription_display_info(),
        'available_models': profile.get_available_models(),
        'subscription_active': profile.is_subscription_active,
        'subscription_end_date': profile.subscription_end_date,
    }

    return render(request, 'subscription_management.html', context)

def statistics(request):
    """Statistics page showing driver and team analytics"""
    # Get current year (default to 2025)
    current_year = request.GET.get('year', '2025')
    try:
        year_int = int(current_year)
    except (ValueError, TypeError):
        year_int = 2025

    # Get events for the selected year
    events = Event.objects.filter(year=year_int)
    event_ids = events.values_list('id', flat=True)

    # Driver Statistics
    driver_stats = (
        RaceResult.objects
        .filter(session__event_id__in=event_ids)
        .values('driver', 'driver__given_name', 'driver__family_name', 'driver__permanent_number', 'team__name')
        .annotate(
            total_races=Count('id'),
            total_points=Sum('points'),
            avg_points=Avg('points'),
            wins=Count('id', filter=Q(position=1)),
            podiums=Count('id', filter=Q(position__lte=3)),
            dnf_count=Count('id', filter=Q(status__icontains='DNF')),
            best_position=Max('position'),
        )
        .order_by('-total_points', '-wins')
    )

    # Team Statistics
    team_stats = (
        RaceResult.objects
        .filter(session__event_id__in=event_ids)
        .values('team', 'team__name')
        .annotate(
            total_races=Count('id'),
            total_points=Sum('points'),
            avg_points=Avg('points'),
            wins=Count('id', filter=Q(position=1)),
            podiums=Count('id', filter=Q(position__lte=3)),
            drivers_count=Count('driver', distinct=True),
        )
        .order_by('-total_points', '-wins')
    )

    # Circuit Statistics - ordered by event round (chronological)
    circuit_stats = (
        RaceResult.objects
        .filter(session__event_id__in=event_ids)
        .values('session__event__circuit__name', 'session__event__circuit__country', 'session__event__round', 'session__event__circuit__id')
        .annotate(
            races_held=Count('session__event', distinct=True),
            total_drivers=Count('driver', distinct=True),
            avg_points=Avg('points'),
        )
        .order_by('session__event__round')  # Order by round number (chronological)
    )

    # Season Overview
    season_overview = {
        'total_races': events.count(),
        'total_drivers': Driver.objects.count(),
        'total_teams': Team.objects.count(),
        'total_circuits': Circuit.objects.count(),
        'year': year_int,
    }

    # Available years for filter
    available_years = sorted(Event.objects.values_list('year', flat=True).distinct(), reverse=True)

    context = {
        'driver_stats': driver_stats,
        'team_stats': team_stats,
        'circuit_stats': circuit_stats,
        'season_overview': season_overview,
        'available_years': available_years,
        'selected_year': year_int,
    }

    return render(request, 'statistics.html', context)


def credits(request):
    """Credits history and transaction page"""
    if not request.user.is_authenticated:
        return redirect('login')

    # Get user profile
    try:
        profile = request.user.profile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)

    # Get recent transactions
    transactions = CreditTransaction.objects.filter(user=request.user).order_by('-timestamp')[:20]

    # Calculate summary stats
    total_earned = sum(t.amount for t in transactions if t.amount > 0)
    total_spent = abs(sum(t.amount for t in transactions if t.amount < 0))

    context = {
        'profile': profile,
        'transactions': transactions,
        'total_earned': total_earned,
        'total_spent': total_spent,
        'is_authenticated': request.user.is_authenticated,
    }

    return render(request, 'credits.html', context)


def portfolio(request):
    """User's betting portfolio and achievements"""
    if not request.user.is_authenticated:
        return redirect('login')

    # Get user profile
    try:
        profile = request.user.profile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)

    # Get user's bets
    bets = Bet.objects.filter(user=request.user).order_by('-created_at')[:10]

    # Get achievements
    achievements = Achievement.objects.filter(is_active=True)
    unlocked_achievements = profile.achievements_unlocked.all()

    # Calculate portfolio stats
    total_bets = profile.total_bets_placed
    win_rate = profile.win_rate
    net_profit = profile.net_profit
    circuits_visited = profile.circuits_visited.count()

    context = {
        'profile': profile,
        'bets': bets,
        'achievements': achievements,
        'unlocked_achievements': unlocked_achievements,
        'total_bets': total_bets,
        'win_rate': win_rate,
        'net_profit': net_profit,
        'circuits_visited': circuits_visited,
        'is_authenticated': request.user.is_authenticated,
    }

    return render(request, 'portfolio.html', context)


def circuits(request):
    """Circuit discovery page showing all circuits with user progress"""
    # Get all circuits with their characteristics
    circuits_data = Circuit.objects.all().order_by('name')

    # Get user progress if authenticated
    user_progress = {}
    if request.user.is_authenticated:
        try:
            profile = request.user.profile
            visited_circuits = profile.circuits_visited.all()
            user_progress = {
                'visited_count': visited_circuits.count(),
                'total_circuits': circuits_data.count(),
                'visited_circuits': set(visited_circuits.values_list('id', flat=True)),
            }
        except UserProfile.DoesNotExist:
            profile = UserProfile.objects.create(user=request.user)
            user_progress = {
                'visited_count': 0,
                'total_circuits': circuits_data.count(),
                'visited_circuits': set(),
            }

    # Get circuit statistics
    circuit_stats = []
    for circuit in circuits_data:
        # Get recent race results for this circuit
        recent_races = RaceResult.objects.filter(
            session__event__circuit=circuit
        ).select_related('driver', 'team', 'session__event').order_by('-session__event__date')[:5]

        # Get track characteristics if available
        try:
            track_spec = TrackSpecialization.objects.get(circuit=circuit)
            track_data = {
                'category': track_spec.category,
                'overtaking_difficulty': track_spec.overtaking_difficulty,
                'power_sensitivity': track_spec.power_sensitivity,
                'weather_impact': track_spec.weather_impact,
            }
        except TrackSpecialization.DoesNotExist:
            track_data = None

        circuit_stats.append({
            'circuit': circuit,
            'recent_races': recent_races,
            'track_data': track_data,
            'is_visited': circuit.id in user_progress.get('visited_circuits', set()),
        })

    context = {
        'circuits': circuit_stats,
        'user_progress': user_progress,
        'is_authenticated': request.user.is_authenticated,
    }

    return render(request, 'circuits.html', context)


def circuit_detail(request, circuit_id):
    """Detailed circuit page with track info and historical data"""
    circuit = get_object_or_404(Circuit, id=circuit_id)

    # Get track specialization data
    try:
        track_spec = TrackSpecialization.objects.get(circuit=circuit)
    except TrackSpecialization.DoesNotExist:
        track_spec = None

    # Get historical race results for this circuit
    historical_results = RaceResult.objects.filter(
        session__event__circuit=circuit
    ).select_related('driver', 'team', 'session__event').order_by('-session__event__date')[:20]

    # Get qualifying results
    qualifying_results = QualifyingResult.objects.filter(
        session__event__circuit=circuit
    ).select_related('driver', 'team', 'session__event').order_by('-session__event__date')[:10]

    # Get circuit statistics
    total_races = RaceResult.objects.filter(session__event__circuit=circuit).count()
    unique_drivers = RaceResult.objects.filter(session__event__circuit=circuit).values('driver').distinct().count()
    unique_teams = RaceResult.objects.filter(session__event__circuit=circuit).values('team').distinct().count()

    # Check if user has visited this circuit
    user_visited = False
    if request.user.is_authenticated:
        try:
            profile = request.user.profile
            user_visited = profile.circuits_visited.filter(id=circuit_id).exists()
        except UserProfile.DoesNotExist:
            pass

    # Get recent events at this circuit
    recent_events = Event.objects.filter(circuit=circuit).order_by('-date')[:5]

    context = {
        'circuit': circuit,
        'track_spec': track_spec,
        'historical_results': historical_results,
        'qualifying_results': qualifying_results,
        'total_races': total_races,
        'unique_drivers': unique_drivers,
        'unique_teams': unique_teams,
        'recent_events': recent_events,
        'user_visited': user_visited,
        'is_authenticated': request.user.is_authenticated,
    }

    return render(request, 'circuit_detail.html', context)


@login_required
@csrf_exempt
def mark_circuit_visited(request, circuit_id):
    """Mark a circuit as visited and award 100 credits to the user"""
    if request.method == 'POST':
        try:
            circuit = get_object_or_404(Circuit, id=circuit_id)
            profile, created = UserProfile.objects.get_or_create(user=request.user)

            # Check if circuit is already visited
            if profile.circuits_visited.filter(id=circuit_id).exists():
                return JsonResponse({'status': 'already_visited', 'message': 'Circuit already visited'})

            # Mark circuit as visited
            profile.circuits_visited.add(circuit)

            # Award 100 credits
            profile.credits += 100
            profile.save()

            # Create credit transaction record
            CreditTransaction.objects.create(
                user=request.user,
                amount=100,
                transaction_type='CIRCUIT_VISIT',
                description=f'Visited {circuit.name}',
                balance_after=profile.credits,
                circuit=circuit
            )

            return JsonResponse({
                'status': 'success',
                'message': 'marked',
                'credits_awarded': 100,
                'total_credits': profile.credits
            })

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})


def betting(request):
    """Enhanced prediction market betting interface with real-time odds"""
    if not request.user.is_authenticated:
        return redirect('login')

    # Get completed events (rounds 1-14)
    completed_events = Event.objects.filter(
        year=2025,
        round__lte=14
    ).select_related('circuit').order_by('round')

    # Get upcoming events (Netherlands GP and beyond)
    upcoming_events = Event.objects.filter(
        year=2025,
        round__gt=14
    ).select_related('circuit').order_by('round')

    # Get user's profile and current credits
    try:
        profile = request.user.profile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)

    # Get recent bets for context
    recent_bets = Bet.objects.filter(user=request.user).select_related(
        'event', 'driver', 'team'
    ).order_by('-created_at')[:5]

    # Get available drivers and teams for betting
    drivers = Driver.objects.all().order_by('given_name', 'family_name')
    teams = Team.objects.all().order_by('name')

    # Enhanced bet types with descriptions and base odds
    enhanced_bet_types = [
        {
            'type': 'PODIUM_FINISH',
            'name': 'Podium Finish',
            'description': 'Driver finishes in top 3 positions',
            'base_odds': 3.0,
            'icon': 'fas fa-trophy'
        },
        {
            'type': 'EXACT_POSITION',
            'name': 'Exact Position',
            'description': 'Driver finishes in specific position',
            'base_odds': 15.0,
            'icon': 'fas fa-crosshairs'
        },
        {
            'type': 'DNF_PREDICTION',
            'name': 'DNF Prediction',
            'description': 'Driver does not finish the race',
            'base_odds': 8.0,
            'icon': 'fas fa-times-circle'
        },
        {
            'type': 'QUALIFYING_POSITION',
            'name': 'Qualifying Position',
            'description': 'Driver qualifies in specific position',
            'base_odds': 10.0,
            'icon': 'fas fa-flag-checkered'
        },
        {
            'type': 'FASTEST_LAP',
            'name': 'Fastest Lap',
            'description': 'Driver sets fastest lap of the race',
            'base_odds': 12.0,
            'icon': 'fas fa-stopwatch'
        },
        {
            'type': 'HEAD_TO_HEAD',
            'name': 'Head-to-Head',
            'description': 'One driver beats another',
            'base_odds': 2.0,
            'icon': 'fas fa-users'
        },
        {
            'type': 'TEAM_BATTLE',
            'name': 'Team Battle',
            'description': 'Team finishes ahead of another team',
            'base_odds': 2.5,
            'icon': 'fas fa-flag'
        },
        {
            'type': 'SAFETY_CAR',
            'name': 'Safety Car',
            'description': 'Safety car appears during the race',
            'base_odds': 4.0,
            'icon': 'fas fa-car'
        }
    ]

    # Get market statistics for all events
    market_stats = {}
    all_events = list(completed_events) + list(upcoming_events)
    for event in all_events:
        event_bets = Bet.objects.filter(event=event)
        total_volume = event_bets.aggregate(total=Sum('credits_staked'))['total'] or 0
        bet_count = event_bets.count()

        # Get most popular bets for this event
        popular_bets = event_bets.values('bet_type', 'driver__given_name', 'driver__family_name').annotate(
            volume=Sum('credits_staked'),
            count=Count('id')
        ).order_by('-volume')[:3]

        market_stats[event.id] = {
            'total_volume': total_volume,
            'bet_count': bet_count,
            'popular_bets': popular_bets
        }

    context = {
        'completed_events': completed_events,
        'upcoming_events': upcoming_events,
        'drivers': drivers,
        'teams': teams,
        'user_credits': profile.credits,
        'recent_bets': recent_bets,
        'enhanced_bet_types': enhanced_bet_types,
        'market_stats': market_stats,
    }

    return render(request, 'betting.html', context)


@login_required
@subscription_required(allowed_models=['catboost_ensemble'])
def chaos_impact(request):
    # Section A: build 2022–2025 overview on the fly
    df = analyze_prediction_errors_by_events(season=None, model_name='catboost_ensemble')
    overall_mae = float(np.mean(np.abs(df['error']))) if not df.empty else float('nan')
    cfa = CounterfactualAnalyzer(df)
    clean_mae = cfa.analyze_clean_race_accuracy()
    # Perfect chaos knowledge: set affected drivers' error to 0
    df_cf = df.copy()
    if not df_cf.empty:
        df_cf.loc[df_cf['driver_affected'] == True, 'error'] = 0.0
        perfect_mae = float(np.mean(np.abs(df_cf['error'])))
    else:
        perfect_mae = float('nan')

    # Simple bootstrap CI for clean MAE
    clean_mask = (df['race_category'] == 'clean') & (~df['driver_affected'])
    clean_errors = df.loc[clean_mask, 'error'].to_numpy() if not df.empty else np.array([])
    def bootstrap_ci(arr, n=1000, alpha=0.05):
        if arr.size == 0:
            return None
        rng = np.random.default_rng(42)
        boots = []
        for _ in range(n):
            sample = rng.choice(arr, size=arr.size, replace=True)
            boots.append(np.mean(np.abs(sample)))
        lo = float(np.quantile(boots, alpha/2))
        hi = float(np.quantile(boots, 1 - alpha/2))
        return (lo, hi)
    ci = bootstrap_ci(clean_errors)

    overview = {
        'overall_mae': f"{overall_mae:.2f}" if not math.isnan(overall_mae) else '—',
        'clean_mae': f"{clean_mae:.2f}" if not math.isnan(clean_mae) else '—',
        'perfect_mae': f"{perfect_mae:.2f}" if not math.isnan(perfect_mae) else '—',
        'clean_ci': f"{ci[0]:.2f}-{ci[1]:.2f}" if ci else '—',
        'p_value': request.GET.get('p','~')  # optional: we can compute with scipy if installed
    }

    # Figures: try to use saved assets if available
    import os
    base = os.path.join(os.getcwd(), 'chaos_analysis')
    figures = {
        'box_by_category': None,
        'scatter_chaos_vs_error': None,
    }
    for name in ['box_mae_by_category_all_years.png', 'scatter_chaos_vs_error_all_years.png']:
        path = os.path.join(base, name)
        if os.path.exists(path):
            figures['box_by_category' if 'box' in name else 'scatter_chaos_vs_error'] = f"/static-cache/{name}"

    # Section B: 2025 events list
    season2025 = models.Event.objects.filter(year=2025).order_by('round').values('id','name','round')

    return render(request, 'chaos_impact.html', {
        'overview': overview,
        'figures': figures,
        'season2025': season2025,
    })


@login_required
@subscription_required(allowed_models=['catboost_ensemble'])
def api_chaos_event(request, event_id: int):
    data = analyze_event(event_id, model_name='catboost_ensemble')
    return JsonResponse(data or {}, safe=False)


def place_bet(request):
    """Enhanced bet placement with market maker integration"""
    if not request.user.is_authenticated:
        return JsonResponse({'success': False, 'error': 'Authentication required'})

    if request.method != 'POST':
        return JsonResponse({'success': False, 'error': 'POST method required'})

    try:
        # Get bet data from request
        event_id = request.POST.get('event_id')
        bet_type = request.POST.get('bet_type')
        driver_id = request.POST.get('driver_id')
        team_id = request.POST.get('team_id')
        position = request.POST.get('position')
        opponent_driver_id = request.POST.get('opponent_driver_id')
        opponent_team_id = request.POST.get('opponent_team_id')
        credits_staked = int(request.POST.get('credits_staked', 0))

        # Validate required fields
        if not all([event_id, bet_type, credits_staked]):
            return JsonResponse({'success': False, 'error': 'Missing required fields'})

        # Get user profile
        try:
            profile = request.user.profile
        except UserProfile.DoesNotExist:
            profile = UserProfile.objects.create(user=request.user)

        # Check betting limits and risk management
        can_bet, error_message = profile.can_place_bet(credits_staked)
        if not can_bet:
            return JsonResponse({'success': False, 'error': error_message})

        # Get risk-adjusted limits
        risk_limits = profile.get_risk_adjusted_limits()

        # Check if bet amount exceeds risk-adjusted limits
        if credits_staked > risk_limits['max_bet_amount']:
            return JsonResponse({
                'success': False,
                'error': f'Bet amount exceeds risk-adjusted limit ({risk_limits["max_bet_amount"]} credits)'
            })

        # Check daily limit
        if profile.daily_bet_amount + credits_staked > risk_limits['daily_bet_limit']:
            return JsonResponse({
                'success': False,
                'error': f'Daily betting limit exceeded ({risk_limits["daily_bet_limit"]} credits)'
            })

        # Check concurrent bets limit
        active_bets = Bet.objects.filter(
            user=request.user,
            status='PENDING'
        ).count()

        if active_bets >= risk_limits['max_concurrent_bets']:
            return JsonResponse({
                'success': False,
                'error': f'Maximum concurrent bets limit reached ({risk_limits["max_concurrent_bets"]} bets)'
            })

        # Get related objects
        event = get_object_or_404(Event, id=event_id)
        driver = get_object_or_404(Driver, id=driver_id) if driver_id else None
        team = get_object_or_404(Team, id=team_id) if team_id else None
        opponent_driver = get_object_or_404(Driver, id=opponent_driver_id) if opponent_driver_id else None
        opponent_team = get_object_or_404(Team, id=opponent_team_id) if opponent_team_id else None

        # Get or create market maker for this bet
        market_maker, created = MarketMaker.objects.get_or_create(
            event=event,
            bet_type=bet_type,
            driver=driver,
            team=team,
            defaults={
                'base_odds': calculate_bet_odds(bet_type, driver, team, event, position, opponent_driver, opponent_team),
                'current_odds': calculate_bet_odds(bet_type, driver, team, event, position, opponent_driver, opponent_team),
            }
        )

        # Get current market odds
        market_odds = market_maker.calculate_adjusted_odds(credits_staked)
        execution_odds = market_odds['ask']  # User pays the ask price

        # Check if market has enough liquidity
        if credits_staked > market_maker.available_liquidity:
            return JsonResponse({
                'success': False,
                'error': f'Insufficient market liquidity. Available: {market_maker.available_liquidity} credits'
            })

        # Get ML prediction if available
        ml_prediction = False
        ml_predicted_position = None
        ml_confidence = None
        if driver and event:
            prediction = CatBoostPrediction.objects.filter(
                driver=driver,
                event=event
            ).order_by('-created_at').first()

            if prediction:
                ml_prediction = True
                ml_predicted_position = prediction.predicted_position
                ml_confidence = prediction.prediction_confidence

        # Create the bet
        bet = Bet.objects.create(
            user=request.user,
            event=event,
            bet_type=bet_type,
            driver=driver,
            team=team,
            opponent_driver=opponent_driver,
            opponent_team=opponent_team,
            predicted_position=position,
            credits_staked=credits_staked,
            odds=execution_odds,
            potential_payout=int(credits_staked * execution_odds),
            ml_prediction_used=ml_prediction,
            ml_predicted_position=ml_predicted_position,
            ml_confidence=ml_confidence,
            status='PENDING'
        )

        # Update market maker state
        market_maker.update_market_state(credits_staked)

        # Deduct credits and update stats
        profile.credits -= credits_staked
        profile.update_betting_stats(credits_staked, won=False, payout=0)

        # Create credit transaction reflecting new balance
        CreditTransaction.objects.create(
            user=request.user,
            transaction_type='BET_PLACED',
            amount=-credits_staked,
            description=f'Bet placed on {event.name} - {bet.get_bet_type_display()}',
            balance_after=profile.credits,
            bet=bet
        )
        profile.save()

        # Get updated market statistics
        market_stats = get_market_statistics(event, bet_type, driver, team)

        return JsonResponse({
            'success': True,
            'bet_id': bet.id,
            'new_credits': profile.credits,
            'execution_odds': execution_odds,
            'potential_payout': bet.potential_payout,
            'market_stats': market_stats,
            'message': 'Bet placed successfully!'
        })

    except Exception as e:
        logger.error(f"Error placing bet: {e}")
        return JsonResponse({'success': False, 'error': str(e)})


def my_bets(request):
    """User's betting history and active bets"""
    if not request.user.is_authenticated:
        return redirect('login')

    # Get user's profile
    try:
        profile = request.user.profile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)

    # Get all user's bets
    bets = Bet.objects.filter(user=request.user).select_related(
        'event', 'driver', 'team'
    ).order_by('-created_at')

    # Separate active and completed bets (status choices are uppercase)
    active_bets = bets.filter(status='PENDING')
    completed_bets = bets.filter(status__in=['WON', 'LOST'])

    # Calculate betting statistics
    total_bets = bets.count()
    won_bets = completed_bets.filter(status='WON').count()
    win_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0

    total_wagered = bets.aggregate(total=Sum('credits_staked'))['total'] or 0
    total_won = completed_bets.filter(status='WON').aggregate(total=Sum('payout_received'))['total'] or 0
    net_profit = total_won - total_wagered

    context = {
        'active_bets': active_bets,
        'completed_bets': completed_bets,
        'total_bets': total_bets,
        'won_bets': won_bets,
        'win_rate': win_rate,
        'total_wagered': total_wagered,
        'total_won': total_won,
        'net_profit': net_profit,
        'user_credits': profile.credits,
    }

    return render(request, 'my_bets.html', context)


def calculate_bet_odds(bet_type, driver, team, event, position=None, opponent_driver=None, opponent_team=None):
    """Enhanced odds calculation with ML predictions and market dynamics"""

    # Base odds from historical data
    base_odds = get_historical_odds(bet_type, driver, team, event, position, opponent_driver, opponent_team)

    # Apply ML prediction adjustments
    ml_adjustment = get_ml_prediction_adjustment(bet_type, driver, team, event, position)

    # Apply market dynamics (betting volume impact)
    market_adjustment = get_market_dynamics_adjustment(bet_type, driver, team, event)

    # Apply track-specific adjustments
    track_adjustment = get_track_specific_adjustment(bet_type, driver, team, event)

    # Calculate final odds
    final_odds = base_odds * ml_adjustment * market_adjustment * track_adjustment

    # Ensure odds are within reasonable bounds
    final_odds = max(1.1, min(50.0, final_odds))

    return round(final_odds, 2)


def get_historical_odds(bet_type, driver, team, event, position=None, opponent_driver=None, opponent_team=None):
    """Get base odds from historical performance data"""
    base_odds = 2.0  # Default odds

    if bet_type == 'PODIUM_FINISH':
        if driver:
            recent_podiums = RaceResult.objects.filter(
                driver=driver,
                position__lte=3
            ).count()
            recent_races = RaceResult.objects.filter(driver=driver).count()
            if recent_races > 0:
                podium_rate = recent_podiums / recent_races
                base_odds = max(1.5, min(8.0, 1 / podium_rate))

    elif bet_type == 'EXACT_POSITION':
        if position and driver:
            position = int(position)
            # Historical performance at this position
            position_finishes = RaceResult.objects.filter(
                driver=driver,
                position=position
            ).count()
            total_races = RaceResult.objects.filter(driver=driver).count()
            if total_races > 0:
                position_rate = position_finishes / total_races
                base_odds = max(5.0, min(30.0, 1 / position_rate))
            else:
                base_odds = 15.0

    elif bet_type == 'DNF_PREDICTION':
        if driver:
            recent_dnfs = RaceResult.objects.filter(
                driver=driver,
                status__icontains='DNF'
            ).count()
            recent_races = RaceResult.objects.filter(driver=driver).count()
            if recent_races > 0:
                dnf_rate = recent_dnfs / recent_races
                if dnf_rate > 0:
                    base_odds = max(1.2, min(15.0, 1 / dnf_rate))
                else:
                    # No DNFs observed: set high base odds (unlikely to DNF)
                    base_odds = 15.0

    elif bet_type == 'QUALIFYING_POSITION':
        if driver:
            recent_qualifying = QualifyingResult.objects.filter(driver=driver).count()
            if recent_qualifying > 0:
                avg_qualifying_pos = QualifyingResult.objects.filter(
                    driver=driver
                ).aggregate(avg_pos=Avg('position'))['avg_pos']
                if avg_qualifying_pos and avg_qualifying_pos <= 5:
                    base_odds = 2.5
                elif avg_qualifying_pos and avg_qualifying_pos <= 10:
                    base_odds = 2.0
                else:
                    base_odds = 1.8

    elif bet_type == 'FASTEST_LAP':
        if driver:
            fastest_laps = RaceResult.objects.filter(
                driver=driver,
                fastest_lap_rank=1
            ).count()
            total_races = RaceResult.objects.filter(driver=driver).count()
            if total_races > 0:
                fastest_lap_rate = fastest_laps / total_races
                base_odds = max(3.0, min(20.0, 1 / fastest_lap_rate))

    elif bet_type == 'HEAD_TO_HEAD':
        if driver and opponent_driver:
            # Calculate head-to-head record
            h2h_wins = 0
            h2h_races = 0
            for race_result in RaceResult.objects.filter(driver=driver):
                opponent_result = RaceResult.objects.filter(
                    driver=opponent_driver,
                    session=race_result.session
                ).first()
                if opponent_result and race_result.position and opponent_result.position:
                    h2h_races += 1
                    if race_result.position < opponent_result.position:
                        h2h_wins += 1

            if h2h_races > 0:
                win_rate = h2h_wins / h2h_races
                base_odds = max(1.3, min(3.0, 1 / win_rate))

    elif bet_type == 'TEAM_BATTLE':
        if team and opponent_team:
            # Calculate team battle record
            team_wins = 0
            team_battles = 0
            for race_result in RaceResult.objects.filter(team=team):
                opponent_result = RaceResult.objects.filter(
                    team=opponent_team,
                    session=race_result.session
                ).first()
                if opponent_result and race_result.position and opponent_result.position:
                    team_battles += 1
                    if race_result.position < opponent_result.position:
                        team_wins += 1

            if team_battles > 0:
                win_rate = team_wins / team_battles
                base_odds = max(1.3, min(4.0, 1 / win_rate))

    elif bet_type == 'SAFETY_CAR':
        # Base safety car probability (varies by circuit)
        base_odds = 4.0

    return base_odds


def get_ml_prediction_adjustment(bet_type, driver, team, event, position=None):
    """Apply ML prediction adjustments to odds"""
    adjustment = 1.0

    try:
        # Get latest CatBoost prediction for this driver/event
        if driver and event:
            prediction = CatBoostPrediction.objects.filter(
                driver=driver,
                event=event
            ).order_by('-created_at').first()

            if prediction:
                predicted_position = prediction.predicted_position
                confidence = prediction.prediction_confidence or 0.5

                if bet_type == 'PODIUM_FINISH':
                    if predicted_position <= 3:
                        # ML predicts podium - lower odds
                        adjustment = 0.8 + (0.2 * confidence)
                    else:
                        # ML doesn't predict podium - higher odds
                        adjustment = 1.2 + (0.3 * (1 - confidence))

                elif bet_type == 'EXACT_POSITION' and position:
                    position = int(position)
                    if abs(predicted_position - position) <= 1:
                        # ML predicts close to this position
                        adjustment = 0.7 + (0.3 * confidence)
                    else:
                        # ML predicts far from this position
                        adjustment = 1.3 + (0.4 * (1 - confidence))

                elif bet_type == 'DNF_PREDICTION':
                    # Use driver's reliability score
                    if hasattr(driver, 'reliability_score') and driver.reliability_score:
                        reliability = driver.reliability_score
                        if reliability < 0.7:  # Low reliability
                            adjustment = 0.6 + (0.2 * (1 - reliability))
                        else:  # High reliability
                            adjustment = 1.4 + (0.3 * reliability)

    except Exception as e:
        logger.warning(f"Error applying ML adjustment: {e}")
        adjustment = 1.0

    return adjustment


def get_market_dynamics_adjustment(bet_type, driver, team, event):
    """Apply market dynamics adjustments based on betting volume"""
    adjustment = 1.0

    try:
        # Get current betting volume for this bet type
        current_volume = Bet.objects.filter(
            event=event,
            bet_type=bet_type,
            driver=driver,
            team=team
        ).aggregate(total=Sum('credits_staked'))['total'] or 0

        # Get total volume for this event
        total_event_volume = Bet.objects.filter(event=event).aggregate(
            total=Sum('credits_staked')
        )['total'] or 1

        # Calculate volume ratio
        volume_ratio = current_volume / total_event_volume

        # Apply adjustment based on volume
        if volume_ratio > 0.3:  # High volume on this bet
            adjustment = 0.8  # Lower odds due to high demand
        elif volume_ratio < 0.05:  # Low volume on this bet
            adjustment = 1.2  # Higher odds due to low demand

    except Exception as e:
        logger.warning(f"Error applying market dynamics: {e}")
        adjustment = 1.0

    return adjustment


def get_track_specific_adjustment(bet_type, driver, team, event):
    """Apply track-specific adjustments based on circuit characteristics"""
    adjustment = 1.0

    try:
        circuit = event.circuit

        # Get track specialization data
        track_spec = TrackSpecialization.objects.filter(circuit=circuit).first()
        if track_spec:
            if bet_type == 'QUALIFYING_POSITION':
                # Qualifying importance affects qualifying bet odds
                qualifying_importance = track_spec.qualifying_importance / 10.0
                adjustment = 1.0 + (0.3 * (1 - qualifying_importance))

            elif bet_type == 'DNF_PREDICTION':
                # Track difficulty affects DNF probability
                overtaking_difficulty = track_spec.overtaking_difficulty / 10.0
                adjustment = 1.0 + (0.2 * overtaking_difficulty)

            elif bet_type == 'SAFETY_CAR':
                # Track characteristics affect safety car probability
                # This would need historical safety car data per circuit
                adjustment = 1.0

    except Exception as e:
        logger.warning(f"Error applying track adjustment: {e}")
        adjustment = 1.0

    return adjustment


def get_real_time_odds(request):
    """AJAX endpoint for real-time odds calculation"""
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=401)

    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)

    try:
        # Get parameters from request
        event_id = request.POST.get('event_id')
        bet_type = request.POST.get('bet_type')
        driver_id = request.POST.get('driver_id')
        team_id = request.POST.get('team_id')
        position = request.POST.get('position')
        opponent_driver_id = request.POST.get('opponent_driver_id')
        opponent_team_id = request.POST.get('opponent_team_id')

        # Validate required fields
        if not all([event_id, bet_type]):
            return JsonResponse({'error': 'Missing required fields'}, status=400)

        # Get related objects
        event = get_object_or_404(Event, id=event_id)
        driver = get_object_or_404(Driver, id=driver_id) if driver_id else None
        team = get_object_or_404(Team, id=team_id) if team_id else None
        opponent_driver = get_object_or_404(Driver, id=opponent_driver_id) if opponent_driver_id else None
        opponent_team = get_object_or_404(Team, id=opponent_team_id) if opponent_team_id else None

        # Calculate odds
        odds = calculate_bet_odds(
            bet_type, driver, team, event, position,
            opponent_driver, opponent_team
        )

        # Get market statistics
        market_stats = get_market_statistics(event, bet_type, driver, team)

        # Get ML prediction if available
        ml_prediction = None
        if driver and event:
            prediction = CatBoostPrediction.objects.filter(
                driver=driver,
                event=event
            ).order_by('-created_at').first()

            if prediction:
                ml_prediction = {
                    'predicted_position': round(prediction.predicted_position, 1),
                    'confidence': round(prediction.prediction_confidence or 0.5, 2),
                    'model_name': prediction.model_name
                }

        return JsonResponse({
            'success': True,
            'odds': odds,
            'market_stats': market_stats,
            'ml_prediction': ml_prediction
        })

    except Exception as e:
        logger.error(f"Error calculating real-time odds: {e}")
        return JsonResponse({'error': str(e)}, status=500)


def get_market_statistics(event, bet_type, driver, team):
    """Get market statistics for a specific bet"""
    try:
        # Get current betting volume for this bet
        current_volume = Bet.objects.filter(
            event=event,
            bet_type=bet_type,
            driver=driver,
            team=team
        ).aggregate(total=Sum('credits_staked'))['total'] or 0

        # Get total volume for this event
        total_event_volume = Bet.objects.filter(event=event).aggregate(
            total=Sum('credits_staked')
        )['total'] or 0

        # Get bet count for this specific bet
        bet_count = Bet.objects.filter(
            event=event,
            bet_type=bet_type,
            driver=driver,
            team=team
        ).count()

        # Get recent odds movement (last 10 bets)
        recent_odds = list(Bet.objects.filter(
            event=event,
            bet_type=bet_type,
            driver=driver,
            team=team
        ).order_by('-created_at').values_list('odds', flat=True)[:10])

        odds_trend = 'stable'
        if len(recent_odds) >= 2:
            first_odds = recent_odds[-1]
            last_odds = recent_odds[0]
            try:
                ratio = last_odds / first_odds if first_odds else 1.0
            except Exception:
                ratio = 1.0
            if ratio > 1.1:
                odds_trend = 'increasing'
            elif ratio < 0.9:
                odds_trend = 'decreasing'

        return {
            'current_volume': current_volume,
            'total_event_volume': total_event_volume,
            'volume_percentage': round((current_volume / total_event_volume * 100) if total_event_volume > 0 else 0, 1),
            'bet_count': bet_count,
            'odds_trend': odds_trend,
            'recent_odds': list(recent_odds)
        }

    except Exception as e:
        logger.error(f"Error getting market statistics: {e}")
        return {
            'current_volume': 0,
            'total_event_volume': 0,
            'volume_percentage': 0,
            'bet_count': 0,
            'odds_trend': 'stable',
            'recent_odds': []
        }


def market_depth(request, event_id):
    """Market depth view showing order book and trading interface"""
    if not request.user.is_authenticated:
        return redirect('login')

    event = get_object_or_404(Event, id=event_id)

    # Get all market makers for this event
    market_makers = MarketMaker.objects.filter(event=event).select_related(
        'driver', 'team'
    ).order_by('bet_type', 'driver__given_name', 'team__name')

    # Get user's profile
    try:
        profile = request.user.profile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)

    # Get user's active orders
    user_orders = MarketOrder.objects.filter(
        user=request.user,
        status='PENDING'
    ).select_related('market_maker').order_by('-created_at')

    context = {
        'event': event,
        'market_makers': market_makers,
        'user_credits': profile.credits,
        'user_orders': user_orders,
    }

    return render(request, 'market_depth.html', context)


def get_market_depth_data(request, market_maker_id):
    """AJAX endpoint for market depth data"""
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=401)

    try:
        market_maker = get_object_or_404(MarketMaker, id=market_maker_id)
        depth_data = market_maker.get_market_depth()

        # Add current odds for different bet amounts
        odds_ladder = []
        for amount in [100, 500, 1000, 2500, 5000]:
            odds = market_maker.calculate_adjusted_odds(amount)
            odds_ladder.append({
                'amount': amount,
                'bid': odds['bid'],
                'ask': odds['ask'],
                'spread': round(odds['ask'] - odds['bid'], 2)
            })

        depth_data['odds_ladder'] = odds_ladder

        return JsonResponse({
            'success': True,
            'depth_data': depth_data
        })

    except Exception as e:
        logger.error(f"Error getting market depth: {e}")
        return JsonResponse({'error': str(e)}, status=500)


def place_market_order(request):
    """Place a market order"""
    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=401)

    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)

    try:
        market_maker_id = request.POST.get('market_maker_id')
        order_type = request.POST.get('order_type')
        side = request.POST.get('side')
        amount = int(request.POST.get('amount', 0))
        limit_price = float(request.POST.get('limit_price', 0)) if request.POST.get('limit_price') else None

        # Validate required fields
        if not all([market_maker_id, order_type, side, amount]):
            return JsonResponse({'error': 'Missing required fields'}, status=400)

        # Get user profile
        try:
            profile = request.user.profile
        except UserProfile.DoesNotExist:
            profile = UserProfile.objects.create(user=request.user)

        # Check if user has enough credits
        if profile.credits < amount:
            return JsonResponse({'error': 'Insufficient credits'}, status=400)

        # Get market maker
        market_maker = get_object_or_404(MarketMaker, id=market_maker_id)

        # Create market order
        order = MarketOrder.objects.create(
            user=request.user,
            market_maker=market_maker,
            order_type=order_type,
            side=side,
            amount=amount,
            limit_price=limit_price
        )

        # Execute market order immediately
        if order_type == 'MARKET':
            success = order.execute_market_order()
            if success:
                # Deduct credits from user
                profile.credits -= amount
                profile.save()

                # Create credit transaction
                CreditTransaction.objects.create(
                    user=request.user,
                    transaction_type='ADMIN_ADJUSTMENT',
                    amount=-amount,
                    description=f'Market order: {side} {amount} credits',
                    balance_after=profile.credits
                )

        return JsonResponse({
            'success': True,
            'order_id': order.id,
            'status': order.status,
            'new_credits': profile.credits
        })

    except Exception as e:
        logger.error(f"Error placing market order: {e}")
        return JsonResponse({'error': str(e)}, status=500)


def risk_settings(request):
    """Risk management settings page"""
    if not request.user.is_authenticated:
        return redirect('login')

    try:
        profile = request.user.profile
    except UserProfile.DoesNotExist:
        profile = UserProfile.objects.create(user=request.user)

    if request.method == 'POST':
        # Update risk tolerance
        risk_tolerance = request.POST.get('risk_tolerance')
        if risk_tolerance in ['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE']:
            profile.risk_tolerance = risk_tolerance

        # Update manual limits
        max_bet_amount = request.POST.get('max_bet_amount')
        if max_bet_amount:
            try:
                profile.max_bet_amount = int(max_bet_amount)
            except ValueError:
                pass

        daily_bet_limit = request.POST.get('daily_bet_limit')
        if daily_bet_limit:
            try:
                profile.daily_bet_limit = int(daily_bet_limit)
            except ValueError:
                pass

        profile.save()

        return JsonResponse({'success': True, 'message': 'Settings updated successfully'})

    # Get current risk-adjusted limits
    risk_limits = profile.get_risk_adjusted_limits()

    # Get betting statistics
    active_bets = Bet.objects.filter(user=request.user, status='PENDING').count()
    today_bets = Bet.objects.filter(
        user=request.user,
        created_at__date=timezone.now().date()
    ).count()

    context = {
        'profile': profile,
        'risk_limits': risk_limits,
        'active_bets': active_bets,
        'today_bets': today_bets,
        'risk_tolerance_choices': [
            ('CONSERVATIVE', 'Conservative - Lower risk, lower rewards'),
            ('MODERATE', 'Moderate - Balanced risk and rewards'),
            ('AGGRESSIVE', 'Aggressive - Higher risk, higher rewards'),
        ]
    }

    return render(request, 'risk_settings.html', context)


def live_updates(request):
    """Live race updates and commentary page"""
    if not request.user.is_authenticated:
        return redirect('login')

    # Get current/upcoming race
    current_event = Event.objects.filter(
        year=2025,
        round__gt=14  # Netherlands GP and beyond
    ).order_by('date').first()

    # Get user's active bets for the current event
    active_bets = []
    if current_event:
        active_bets = Bet.objects.filter(
            user=request.user,
            event=current_event,
            status='PENDING'
        ).select_related('driver', 'team')[:5]

    # Available tracks for mock simulation
    available_tracks = [
        {'name': 'Australian Grand Prix', 'type': 'historical', 'mae': 3.8},
        {'name': 'Chinese Grand Prix', 'type': 'historical', 'mae': 3.5},
        {'name': 'Dutch Grand Prix', 'type': 'upcoming', 'mae': None},
        {'name': 'Italian Grand Prix', 'type': 'upcoming', 'mae': None},
        {'name': 'Azerbaijan Grand Prix', 'type': 'upcoming', 'mae': None},
        {'name': 'Singapore Grand Prix', 'type': 'upcoming', 'mae': None},
    ]

    context = {
        'current_event': current_event,
        'active_bets': active_bets,
        'user_credits': request.user.profile.credits if hasattr(request.user, 'profile') else 0,
        'available_tracks': available_tracks,
    }

    return render(request, 'live_updates.html', context)

# Mock Race Simulation Views
import asyncio
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import sys
import os

# Import mock race simulator
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

try:
    from live_prediction_system import MockRaceSimulator, EventType
except ImportError as e:
    MockRaceSimulator = None
    EventType = None

# Global simulator instance (in production, use Redis/database)
current_simulator = None

from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

@csrf_exempt
@require_http_methods(["POST"])
def start_mock_race(request):
    """Start a new mock race simulation"""
    global current_simulator

    # Temporarily disable auth check for testing
    # if not request.user.is_authenticated:
    #     return JsonResponse({'error': 'Authentication required'}, status=401)

    if MockRaceSimulator is None:
        return JsonResponse({'error': 'Mock race simulator not available'}, status=500)

    try:
        data = json.loads(request.body)

        event_name = data.get('event_name', 'Dutch Grand Prix')
        simulation_speed = data.get('simulation_speed', '10min')

        # Create new simulator
        current_simulator = MockRaceSimulator(
            event_name=event_name,
            simulation_speed=simulation_speed
        )

        # Convert track_config to JSON-serializable format
        track_config_serializable = current_simulator.track_config.copy()
        if 'track_type' in track_config_serializable:
            track_config_serializable['track_type'] = track_config_serializable['track_type'].value

        response_data = {
            'success': True,
            'message': f'Mock race started: {event_name}',
            'total_laps': current_simulator.total_laps,
            'lap_interval': current_simulator.lap_interval,
            'track_config': track_config_serializable
        }

        return JsonResponse(response_data)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def get_race_status(request):
    """Get current race status and updates"""
    global current_simulator

    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=401)

    if current_simulator is None:
        return JsonResponse({'error': 'No active race simulation'}, status=404)

    try:
        # Run the simulation step synchronously for web interface
        import asyncio
        import nest_asyncio

        # Allow nested event loops (needed for Django + asyncio)
        try:
            nest_asyncio.apply()
        except:
            pass

        # Create new event loop for this request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # Simulate one lap
            lap_result = loop.run_until_complete(current_simulator.simulate_lap())

            return JsonResponse({
                'success': True,
                'race_data': lap_result
            })
        finally:
            loop.close()

    except Exception as e:
        import traceback
        print(f"Error in get_race_status: {e}")
        print(traceback.format_exc())
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def trigger_race_event(request):
    """Trigger a user event during the race"""
    global current_simulator

    # Temporarily disable auth check for testing
    # if not request.user.is_authenticated:
    #     return JsonResponse({'error': 'Authentication required'}, status=401)

    if current_simulator is None:
        return JsonResponse({'error': 'No active race simulation'}, status=404)

    try:
        data = json.loads(request.body)

        event_type_str = data.get('event_type')
        target_lap = data.get('target_lap')
        if target_lap is None:
            target_lap = current_simulator.current_lap + 5
        driver_name = data.get('driver_name')

        # Convert string to EventType enum
        event_type_map = {
            'safety_car': EventType.SAFETY_CAR,
            'bad_pit_stop': EventType.BAD_PIT_STOP,
            'weather_change': EventType.WEATHER_CHANGE
        }

        event_type = event_type_map.get(event_type_str)
        if not event_type:
            return JsonResponse({'error': 'Invalid event type'}, status=400)

        success = current_simulator.add_user_event(event_type, target_lap, driver_name)

        if success:
            return JsonResponse({
                'success': True,
                'message': f'Event scheduled: {event_type_str} on lap {target_lap}',
                'events_remaining': current_simulator.max_user_events - current_simulator.user_events_used
            })
        else:
            return JsonResponse({
                'error': 'Cannot add event (max events reached or invalid lap)',
                'events_remaining': current_simulator.max_user_events - current_simulator.user_events_used
            }, status=400)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def get_final_results(request):
    """Get final race results"""
    global current_simulator

    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=401)

    if current_simulator is None:
        return JsonResponse({'error': 'No active race simulation'}, status=404)

    try:
        if current_simulator.race_finished:
            summary = current_simulator.get_simulation_summary()
            return JsonResponse({
                'success': True,
                'final_results': summary['final_results'],
                'commentary_feed': summary['commentary_feed'],
                'ml_summary': summary.get('ml_summary', {}),
                'prediction_comparison': summary.get('prediction_comparison', {}),
                'race_summary': {
                    'event_name': summary['event_name'],
                    'total_laps': summary['total_laps'],
                    'user_events_used': summary['user_events_used'],
                    'track_type': summary['track_config']['track_type'].value,
                    'ml_initialized': getattr(current_simulator, 'ml_initialized', False)
                }
            })
        else:
            return JsonResponse({'error': 'Race not finished yet'}, status=400)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def stop_mock_race(request):
    """Stop the current mock race simulation"""
    global current_simulator

    if not request.user.is_authenticated:
        return JsonResponse({'error': 'Authentication required'}, status=401)

    current_simulator = None
    return JsonResponse({'success': True, 'message': 'Mock race stopped'})

def forgot_password(request):
    """Forgot password page - step 1: enter email/username"""
    # Clear any existing session data to prevent conflicts
    if 'temp_password' in request.session:
        del request.session['temp_password']
    if 'user_id' in request.session:
        del request.session['user_id']
    if 'password_sent_time' in request.session:
        del request.session['password_sent_time']

    if request.method == 'POST':
        email_or_username = request.POST.get('email_or_username', '').strip()

        if not email_or_username:
            messages.error(request, 'Please enter your email or username.')
            return render(request, 'forgot_password.html', {'step': 1})

        # Try to find user by email or username
        try:
            from django.contrib.auth.models import User
            user = User.objects.get(
                Q(username=email_or_username) | Q(email=email_or_username)
            )

            # Generate temporary password
            temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

            # Store temporary password in session
            request.session['temp_password'] = temp_password
            request.session['user_id'] = user.id
            request.session['password_sent_time'] = timezone.now().isoformat()

            # Send email with temporary password
            try:
                send_mail(
                    'F1 Dashboard - Password Reset',
                    f'''Hello {user.username},

You requested a password reset for your F1 Dashboard account.

Your temporary password is: {temp_password}

Please enter this password on the next page to reset your password.

If you didn't request this reset, please ignore this email.

Best regards,
F1 Dashboard Team''',
                    'dattatarun86@gmail.com',
                    [user.email],
                    fail_silently=False,
                )

                messages.success(request, f'Password sent to {user.email}')
                return render(request, 'forgot_password.html', {'step': 2, 'email': user.email})

            except Exception as e:
                messages.error(request, 'Failed to send email. Please try again.')
                return render(request, 'forgot_password.html', {'step': 1})

        except User.DoesNotExist:
            messages.error(request, 'No account found with that email or username.')
            return render(request, 'forgot_password.html', {'step': 1})

    return render(request, 'forgot_password.html', {'step': 1})

def verify_temp_password(request):
    """Forgot password page - step 2: verify temporary password"""
    if request.method == 'POST':
        temp_password = request.POST.get('temp_password', '').strip()
        stored_password = request.session.get('temp_password')
        user_id = request.session.get('user_id')

        if not temp_password or not stored_password or not user_id:
            messages.error(request, 'Invalid session. Please try again.')
            return redirect('forgot_password')

        # Check if password is correct
        if temp_password == stored_password:
            # Check if password is not expired (15 minutes)
            sent_time = datetime.fromisoformat(request.session.get('password_sent_time', ''))
            if timezone.now() - sent_time > timedelta(minutes=15):
                messages.error(request, 'Temporary password has expired. Please request a new one.')
                return redirect('forgot_password')

            # Move to step 3: set new password
            return render(request, 'forgot_password.html', {'step': 3})
        else:
            messages.error(request, 'Incorrect temporary password.')
            return render(request, 'forgot_password.html', {'step': 2})

    return redirect('forgot_password')

def reset_password(request):
    """Forgot password page - step 3: set new password"""
    if request.method == 'POST':
        new_password = request.POST.get('new_password', '').strip()
        confirm_password = request.POST.get('confirm_password', '').strip()
        user_id = request.session.get('user_id')

        if not new_password or not confirm_password or not user_id:
            messages.error(request, 'Invalid session. Please try again.')
            return redirect('forgot_password')

        # Validate passwords
        if len(new_password) < 8:
            messages.error(request, 'Password must be at least 8 characters long.')
            return render(request, 'forgot_password.html', {'step': 3})

        if new_password != confirm_password:
            messages.error(request, 'Passwords do not match.')
            return render(request, 'forgot_password.html', {'step': 3})

        # Update user password
        try:
            from django.contrib.auth.models import User
            user = User.objects.get(id=user_id)
            user.set_password(new_password)
            user.save()

            # Clear session data
            if 'temp_password' in request.session:
                del request.session['temp_password']
            if 'user_id' in request.session:
                del request.session['user_id']
            if 'password_sent_time' in request.session:
                del request.session['password_sent_time']

            # Log user in
            auth_login(request, user)

            messages.success(request, 'Password successfully reset! You are now logged in.')
            return redirect('home')

        except User.DoesNotExist:
            messages.error(request, 'User not found. Please try again.')
            return redirect('forgot_password')

    return redirect('forgot_password')

@csrf_exempt
def resend_temp_password(request):
    """Resend temporary password"""
    user_id = request.session.get('user_id')

    if not user_id:
        messages.error(request, 'Invalid session. Please try again.')
        return redirect('forgot_password')

    try:
        from django.contrib.auth.models import User
        user = User.objects.get(id=user_id)

        # Generate new temporary password
        temp_password = ''.join(random.choices(string.ascii_letters + string.digits, k=8))

        # Update session
        request.session['temp_password'] = temp_password
        request.session['password_sent_time'] = timezone.now().isoformat()

        # Send new email
        try:
            send_mail(
                'F1 Dashboard - New Password Reset Code',
                f'''Hello {user.username},

You requested a new password reset code for your F1 Dashboard account.

Your new temporary password is: {temp_password}

Please enter this password on the next page to reset your password.

If you didn't request this reset, please ignore this email.

Best regards,
F1 Dashboard Team''',
                'dattatarun86@gmail.com',
                [user.email],
                fail_silently=False,
            )

            messages.success(request, f'New password sent to {user.email}')
            return render(request, 'forgot_password.html', {'step': 2, 'email': user.email})

        except Exception as e:
            messages.error(request, 'Failed to send email. Please try again.')
            return render(request, 'forgot_password.html', {'step': 2})

    except User.DoesNotExist:
        messages.error(request, 'User not found. Please try again.')
        return redirect('forgot_password')