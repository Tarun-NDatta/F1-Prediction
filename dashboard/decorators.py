from functools import wraps
from django.http import JsonResponse
from django.shortcuts import redirect
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from data.models import UserProfile

def subscription_required(allowed_models=None):
    """
    Decorator to check if user has access to specific ML models based on subscription tier.
    
    Args:
        allowed_models: List of model names that require subscription access
                       If None, allows all models based on user's tier
    """
    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def _wrapped_view(request, *args, **kwargs):
            try:
                profile = request.user.profile
            except UserProfile.DoesNotExist:
                # Create profile if it doesn't exist
                profile = UserProfile.objects.create(user=request.user)
            
            # Get requested model from request
            requested_model = request.GET.get('model', 'ridge_regression')
            
            # Check if user can access the requested model
            if not profile.can_access_model(requested_model):
                # Get user's available models for error message
                available_models = profile.get_available_models()
                subscription_info = profile.get_subscription_display_info()
                
                if request.headers.get('Content-Type') == 'application/json' or request.is_ajax():
                    return JsonResponse({
                        'error': 'Subscription upgrade required',
                        'message': f'The {requested_model} model requires a {subscription_info["name"]} subscription or higher.',
                        'available_models': available_models,
                        'current_tier': profile.subscription_tier,
                        'upgrade_required': True
                    }, status=403)
                else:
                    messages.error(
                        request, 
                        f'The {requested_model} model requires a subscription upgrade. '
                        f'Your current {subscription_info["name"]} tier includes: {", ".join(available_models)}'
                    )
                    # Redirect to subscription page or back to predictions with basic model
                    return redirect(f'{request.path}?model=ridge_regression')
            
            return view_func(request, *args, **kwargs)
        return _wrapped_view
    return decorator

def model_access_required(model_name):
    """
    Decorator to check access to a specific ML model.
    
    Args:
        model_name: The specific model name to check access for
    """
    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def _wrapped_view(request, *args, **kwargs):
            try:
                profile = request.user.profile
            except UserProfile.DoesNotExist:
                profile = UserProfile.objects.create(user=request.user)
            
            if not profile.can_access_model(model_name):
                subscription_info = profile.get_subscription_display_info()
                
                if request.headers.get('Content-Type') == 'application/json' or request.is_ajax():
                    return JsonResponse({
                        'error': 'Access denied',
                        'message': f'Access to {model_name} requires subscription upgrade.',
                        'current_tier': profile.subscription_tier,
                        'required_upgrade': True
                    }, status=403)
                else:
                    messages.error(
                        request,
                        f'Access to {model_name} requires a subscription upgrade. '
                        f'Current tier: {subscription_info["name"]}'
                    )
                    return redirect('prediction')
            
            return view_func(request, *args, **kwargs)
        return _wrapped_view
    return decorator

def get_user_model_context(user):
    """
    Helper function to get model access context for templates.
    
    Args:
        user: Django User object
        
    Returns:
        dict: Context with model access information
    """
    try:
        profile = user.profile
    except (UserProfile.DoesNotExist, AttributeError):
        if user.is_authenticated:
            profile = UserProfile.objects.create(user=user)
        else:
            return {
                'available_models': [],
                'subscription_info': None,
                'is_authenticated': False
            }
    
    return {
        'available_models': profile.get_available_models(),
        'subscription_info': profile.get_subscription_display_info(),
        'subscription_tier': profile.subscription_tier,
        'is_subscription_active': profile.is_subscription_active,
        'is_authenticated': user.is_authenticated
    }
