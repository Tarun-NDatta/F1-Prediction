from django.shortcuts import get_object_or_404, render, get_list_or_404
from data.models import Event,ridgeregression



def home(request):
    return render(request, 'home.html')

def results(request):
    return render(request, 'results.html')



def prediction(request):
    # Get the event for 2025 round 5
    current_event = get_object_or_404(Event, year=2025, round=5)

    model_name = 'ridge_regression'
    
    # Query ridgeregression for that event and model
    predictions = ridgeregression.objects.filter(
        event=current_event,
        model_name=model_name,
        year=2025,
        round_number=5
    ).select_related('driver', 'event').order_by('predicted_position')

    # Prepare the data to pass to the template
    comparison = []
    for pred in predictions:
        comparison.append({
            'driver': f"{pred.driver.given_name} {pred.driver.family_name}",
            'predicted': round(pred.predicted_position, 2),
            'actual': pred.actual_position if pred.actual_position is not None else 'N/A',
            'difference': (pred.actual_position - pred.predicted_position) if pred.actual_position is not None else 'N/A',
            'driver_number': pred.driver.permanent_number or '',
            'team': 'N/A',  # No direct team relation, so mark as N/A or leave blank
            'team_slug': '',  # same here
            'confidence': None,  # no confidence field in your model currently
            'is_correct': pred.actual_position == round(pred.predicted_position) if pred.actual_position is not None else False,
        })

    context = {
        'current_event': current_event,
        'comparison': comparison,
        'accuracy_stats': None,  # You can calculate and add later
        'error': None,
        'selected_round': 5,
        'available_rounds': [5],  # For now just 5
    }
    return render(request, 'prediction.html', context)