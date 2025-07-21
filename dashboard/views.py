from django.shortcuts import get_object_or_404, render, get_list_or_404
from django.db.models import Count, Avg, Q, Max, Sum
from data.models import (
    Event, RaceResult, QualifyingResult, Circuit, Team, Driver, 
    Session, SessionType, ridgeregression
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
    # Available models configuration
    AVAILABLE_MODELS = {
        'ridge_regression': {
            'name': 'Ridge Regression',
            'model_class': ridgeregression,
            'available_rounds': list(range(1, 12)),  # Rounds 1-11
            'status': 'active'
        },
        'xgboost': {
            'name': 'XGBoost',
            'model_class': None,  # Will be added when model is ready
            'available_rounds': [],  # No rounds available yet
            'status': 'coming_soon'
        },
        'lightgbm': {
            'name': 'LightGBM', 
            'model_class': None,  # Will be added when model is ready
            'available_rounds': [],  # No rounds available yet
            'status': 'coming_soon'
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
        predictions_qs = selected_model['model_class'].objects.filter(
            event=event,
            model_name=selected_model_key,
            year=2025,
            round_number=event.round
        ) if (selected_model['model_class'] is not None) else selected_model['model_class']
        predictions = list(predictions_qs) if predictions_qs else []

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
            team_name = 'N/A'
            team_class = 'team-default'
            points = None
            if act and act.team:
                team_name = act.team.name
                team_class = f"team-{act.team.name.lower().replace(' ', '')}"
                points = act.points
            elif pred:
                # Try to get team from prediction if available
                team_name = getattr(pred, 'team', 'N/A')
                team_class = 'team-default'
                points = getattr(pred, 'points', None)
            comparison.append({
                'driver': f"{driver.given_name} {driver.family_name}" if driver else 'N/A',
                'predicted': round(pred.predicted_position, 2) if pred else 'N/A',
                'actual': act.position if act and act.position is not None else (pred.actual_position if pred and pred.actual_position is not None else 'N/A'),
                'difference': (act.position - round(pred.predicted_position)) if pred and act and act.position is not None and pred.predicted_position is not None else 'N/A',
                'driver_number': driver.permanent_number if driver and driver.permanent_number else '',
                'team': team_name,
                'team_class': team_class,
                'team_slug': '',
                'confidence': round(pred.confidence * 100, 1) if pred and hasattr(pred, 'confidence') and pred.confidence else 'N/A',
                'is_correct': (act.position == round(pred.predicted_position)) if pred and act and act.position is not None and pred.predicted_position is not None else False,
                'points': points if points is not None else 'N/A',
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