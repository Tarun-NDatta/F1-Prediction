from django.shortcuts import get_object_or_404, render, get_list_or_404
from data.models import Event, ridgeregression
from django.db.models import Count, Avg


def home(request):
    return render(request, 'home.html')

def results(request):
    return render(request, 'results.html')


def prediction(request):
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
        race_data = {
            'event': event,
            'has_predictions': event.round in available_rounds and selected_model['status'] == 'active',
            'comparison': [],
            'accuracy_stats': None,
            'coming_soon': event.round not in available_rounds or selected_model['status'] == 'coming_soon'
        }
        
        # Only process predictions if model is active and has data for this round
        if (event.round in available_rounds and 
            selected_model['status'] == 'active' and 
            selected_model['model_class'] is not None):
            
            # Get predictions for this event
            predictions = selected_model['model_class'].objects.filter(
                event=event,
                model_name=selected_model_key,
                year=2025,
                round_number=event.round
            ).select_related('driver', 'event').order_by('predicted_position')
            
            # Prepare comparison data
            comparison = []
            correct_predictions = 0
            total_predictions = 0
            position_differences = []
            top3_correct = 0
            
            for pred in predictions:
                total_predictions += 1
                difference = 'N/A'
                is_correct = False
                
                if pred.actual_position is not None:
                    difference = pred.actual_position - pred.predicted_position
                    position_differences.append(abs(difference))
                    
                    # Check if prediction is exactly correct
                    is_correct = pred.actual_position == round(pred.predicted_position)
                    if is_correct:
                        correct_predictions += 1
                    
                    # Check top 3 accuracy
                    predicted_pos = round(pred.predicted_position)
                    actual_pos = pred.actual_position
                    if predicted_pos <= 3 and actual_pos <= 3:
                        top3_correct += 1
                
                comparison.append({
                    'driver': f"{pred.driver.given_name} {pred.driver.family_name}",
                    'predicted': round(pred.predicted_position, 2),
                    'actual': pred.actual_position if pred.actual_position is not None else 'N/A',
                    'difference': difference,
                    'driver_number': pred.driver.permanent_number or '',
                    'team': 'N/A',  # Add team lookup if you have team data
                    'team_slug': '',
                    'confidence': round(pred.confidence * 100, 1) if hasattr(pred, 'confidence') and pred.confidence else None,
                    'is_correct': is_correct,
                })
            
            race_data['comparison'] = comparison
            
            # Calculate accuracy stats if we have actual results
            if position_differences:
                # Calculate how many drivers were predicted in top 3
                top3_predicted_count = len([p for p in predictions if round(p.predicted_position) <= 3 and p.actual_position is not None])
                top3_accuracy = round((top3_correct / top3_predicted_count) * 100, 1) if top3_predicted_count > 0 else 0
                
                accuracy_stats = {
                    'accuracy': round((correct_predictions / total_predictions) * 100, 1),
                    'top3_accuracy': f"{top3_accuracy}%",
                    'avg_position_diff': round(sum(position_differences) / len(position_differences), 2),
                    'correct_predictions': correct_predictions,
                    'total_predictions': total_predictions,
                    'model_name': selected_model_key
                }
                race_data['accuracy_stats'] = accuracy_stats
        
        results.append(race_data)  # CHANGED from races_data to results
    
    context = {
        'results': results,  # RENAMED from races_data
        'available_rounds': available_rounds,
        'available_rounds_display': available_rounds_display,
        'total_races': len(all_events),
        'races_with_predictions': len(available_rounds),
        'model_name': selected_model['name'],
        'selected_model_key': selected_model_key,
        'available_models': AVAILABLE_MODELS,
        'model_status': selected_model['status'],
        'error': None,
    }
    
    return render(request, 'prediction.html', context)