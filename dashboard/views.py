from django.shortcuts import get_object_or_404, render, get_list_or_404
from django.db.models import Count, Avg, Q, Max, Sum
from django.db.models import CharField, Value  # Django's Value, not torch
from django.db.models.functions import Concat

from data.models import (
    Event, RaceResult, QualifyingResult, Circuit, Team, Driver, 
    Session, SessionType, ridgeregression, xgboostprediction, CatBoostPrediction,
    UserProfile, CreditTransaction, Bet, Achievement, TrackSpecialization, MarketMaker, MarketOrder
)
from django.core.paginator import Paginator
from django.http import JsonResponse
import logging
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login as auth_login, authenticate, logout as auth_logout
from django.shortcuts import redirect
from django import forms
from django.contrib.auth.models import User
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.core.mail import send_mail
from django.template.loader import render_to_string
from django.conf import settings
from django.utils import timezone

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
                    'laps': result.laps or 0
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
        if result.status and ('DNF' in result.status or 'Retired' in result.status):
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
    
    # Available models configuration - UPDATED to include CatBoost
    AVAILABLE_MODELS = {
        'ridge_regression': {
            'name': 'Ridge Regression',
            'model_class': ridgeregression,
            'available_rounds': list(range(1, 15)),  # Rounds 1-14 (completed) + upcoming
            'status': 'active',
            'model_name_filter': 'ridge_regression'  # Added for database filtering
        },
        'xgboost': {
            'name': 'XGBoost',
            'model_class': xgboostprediction,  # Now available
            'available_rounds': list(range(1, 15)),  # Update based on your available data
            'status': 'active',  # Changed from 'coming_soon' to 'active'
            'model_name_filter': 'xgboost_regression'  # Added for database filtering
        },
        'catboost': {
            'name': 'CatBoost Ensemble',
            'model_class': CatBoostPrediction,  # Now available
            'available_rounds': list(range(1, 15)),  # Update based on your available data
            'status': 'active',  # Changed from 'coming_soon' to 'active'
            'model_name_filter': 'catboost_ensemble'  # Added for database filtering
        }
    }
    
    # Get selected model from request (default to ridge_regression)
    selected_model_key = request.GET.get('model', 'ridge_regression')
    if selected_model_key not in AVAILABLE_MODELS:
        selected_model_key = 'ridge_regression'
    
    selected_model = AVAILABLE_MODELS[selected_model_key]
    
    # Get all events for 2025 season, ordered by round
    all_events = Event.objects.filter(year=2025).select_related('circuit').order_by('round')
    
    # Get available rounds for selected model
    available_rounds = selected_model['available_rounds']
    
    # Create display string for available rounds
    if available_rounds:
        available_rounds_display = f"{min(available_rounds)}-{max(available_rounds)}"
    else:
        available_rounds_display = "Coming Soon"
    
    # Prepare data for all races - RENAMED from races_data to results
    results = []
    
    for event in all_events:
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
                predictions_qs = predictions_qs.filter(model_name=selected_model['model_name_filter'])
            
            predictions = list(predictions_qs)
        else:
            predictions = []

        # Get actual results for this event (RaceResult)
        from data.models import RaceResult
        actuals_qs = RaceResult.objects.filter(session__event=event).select_related('driver', 'team')
        actuals = list(actuals_qs) if actuals_qs else []

        # Build a set of all drivers who have either a prediction or an actual result
        driver_ids = set()
        for pred in predictions:
            driver_ids.add(pred.driver_id)
        for act in actuals:
            driver_ids.add(act.driver_id)

        # Build a mapping for quick lookup
        pred_map = {p.driver_id: p for p in predictions}
        act_map = {a.driver_id: a for a in actuals}

        comparison = []
        for driver_id in driver_ids:
            pred = pred_map.get(driver_id)
            act = act_map.get(driver_id)
            driver = pred.driver if pred else (act.driver if act else None)
            
            # Get team information
            team_name = 'N/A'
            team_class = 'team-default'
            team_color = '#666666'  # Default color
            points = None
            
            if act and act.team:
                team_name = act.team.name
                team_class = f"team-{act.team.name.lower().replace(' ', '')}"
                # You can add team colors here based on team name
                team_colors = {
                    'Mercedes': '#00D2BE',
                    'Ferrari': '#DC0000',
                    'Red Bull': '#0600EF',
                    'McLaren': '#FF8700',
                    'Alpine': '#0090FF',
                    'Aston Martin': '#006F62',
                    'Williams': '#005AFF',
                    'AlphaTauri': '#2B4562',
                    'Alfa Romeo': '#900000',
                    'Haas': '#FFFFFF'
                }
                team_color = team_colors.get(act.team.name, '#666666')
                points = act.points
            elif pred:
                # Try to get team from prediction if available
                team_name = getattr(pred, 'team', 'N/A')
                team_class = 'team-default'
                points = getattr(pred, 'points', None)
            
            # Get CatBoost specific features if available
            catboost_features = {}
            if pred and selected_model_key == 'catboost':
                catboost_features = {
                    'track_category': getattr(pred, 'track_category', 'N/A'),
                    'track_power_sensitivity': getattr(pred, 'track_power_sensitivity', 'N/A'),
                    'track_overtaking_difficulty': getattr(pred, 'track_overtaking_difficulty', 'N/A'),
                    'track_qualifying_importance': getattr(pred, 'track_qualifying_importance', 'N/A'),
                    'ridge_prediction': getattr(pred, 'ridge_prediction', 'N/A'),
                    'xgboost_prediction': getattr(pred, 'xgboost_prediction', 'N/A'),
                    'ensemble_prediction': getattr(pred, 'ensemble_prediction', 'N/A'),
                    'prediction_confidence': getattr(pred, 'prediction_confidence', 'N/A'),
                }
            
            comparison.append({
                'driver': f"{driver.given_name} {driver.family_name}" if driver else 'N/A',
                'predicted': int(round(pred.predicted_position)) if pred else 'N/A',  # Convert to int for cleaner display
                'actual': act.position if act and act.position is not None else (pred.actual_position if pred and pred.actual_position is not None else 'N/A'),
                'difference': (act.position - int(round(pred.predicted_position))) if pred and act and act.position is not None and pred.predicted_position is not None else 'N/A',
                'driver_number': driver.permanent_number if driver and driver.permanent_number else '',
                'team': team_name,
                'team_class': team_class,
                'team_color': team_color,  # Added team color
                'team_slug': '',
                'confidence': round(pred.confidence * 100, 1) if pred and hasattr(pred, 'confidence') and pred.confidence else 'N/A',
                'is_correct': (act.position == int(round(pred.predicted_position))) if pred and act and act.position is not None and pred.predicted_position is not None else False,
                'points': points if points is not None else 'N/A',
                'catboost_features': catboost_features,  # Add CatBoost specific features
            })

        # Sort comparison: if actual results are available, order by actual; else by predicted
        if any(item['actual'] != 'N/A' for item in comparison):
            comparison.sort(key=lambda x: (x['actual'] if x['actual'] != 'N/A' else 9999))
        else:
            comparison.sort(key=lambda x: (x['predicted'] if x['predicted'] != 'N/A' else 9999))

        # Only show coming soon if there is no data at all
        show_coming_soon = len(comparison) == 0

        race_data = {
            'event': event,
            'comparison': comparison,
            'show_coming_soon': show_coming_soon,
        }
        results.append(race_data)
    
    context = {
        'results': results,
        'available_rounds': available_rounds,
        'available_rounds_display': available_rounds_display,
        'total_races': len(all_events),
        'races_with_predictions': len(available_rounds),
        'model_name': selected_model['name'],
        'selected_model_key': selected_model_key,
        'available_models': AVAILABLE_MODELS,
        'model_status': selected_model['status'],
        'error': None,
        'predictions_locked': predictions_locked,
        'is_authenticated': request.user.is_authenticated,
    }
    
    return render(request, 'prediction.html', context)

def standings(request):
    # Get all 2025 events
    event_2025_ids = Event.objects.filter(year=2025).values_list('id', flat=True)
    # Driver standings: sum points for each driver
    driver_points = (
        RaceResult.objects
        .filter(session__event_id__in=event_2025_ids)
        .values('driver', 'driver__given_name', 'driver__family_name', 'driver__permanent_number')
        .annotate(points=Sum('points'))
        .order_by('-points', 'driver__family_name', 'driver__given_name')
    )
    driver_standings = []
    for idx, d in enumerate(driver_points, 1):
        driver_standings.append({
            'position': idx,
            'number': d['driver__permanent_number'],
            'name': f"{d['driver__given_name']} {d['driver__family_name']}",
            'points': d['points'] or 0,
        })
    # Team standings: sum points for each team
    team_points = (
        RaceResult.objects
        .filter(session__event_id__in=event_2025_ids)
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
    return render(request, 'standings.html', {'driver_standings': driver_standings, 'team_standings': team_standings})

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
        ml_prediction = None
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
        
        # Update user profile with new betting statistics
        profile.update_betting_stats(credits_staked, won=False, payout=0)
        
        # Create credit transaction
        CreditTransaction.objects.create(
            user=request.user,
            transaction_type='BET_PLACED',
            amount=-credits_staked,
            description=f'Bet placed on {event.name} - {bet.get_bet_type_display()}',
            balance_after=profile.credits,
            bet=bet
        )
        
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
    
    # Separate active and completed bets
    active_bets = bets.filter(status='pending')
    completed_bets = bets.filter(status__in=['won', 'lost'])
    
    # Calculate betting statistics
    total_bets = bets.count()
    won_bets = completed_bets.filter(status='won').count()
    win_rate = (won_bets / total_bets * 100) if total_bets > 0 else 0
    
    total_wagered = bets.aggregate(total=Sum('credits_staked'))['total'] or 0
    total_won = completed_bets.filter(status='won').aggregate(total=Sum('payout_received'))['total'] or 0
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
    base_odds = get_historical_odds(bet_type, driver, team, event, position)
    
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


def get_historical_odds(bet_type, driver, team, event, position=None):
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
                base_odds = max(1.2, min(15.0, 1 / dnf_rate))
    
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
        recent_odds = Bet.objects.filter(
            event=event,
            bet_type=bet_type,
            driver=driver,
            team=team
        ).order_by('-created_at').values_list('odds', flat=True)[:10]
        
        odds_trend = 'stable'
        if len(recent_odds) >= 2:
            first_odds = recent_odds[-1]
            last_odds = recent_odds[0]
            if last_odds > first_odds * 1.1:
                odds_trend = 'increasing'
            elif last_odds < first_odds * 0.9:
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
                    transaction_type='MARKET_ORDER',
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
    
    context = {
        'current_event': current_event,
        'active_bets': active_bets,
        'user_credits': request.user.profile.credits if hasattr(request.user, 'profile') else 0,
    }
    
    return render(request, 'live_updates.html', context)