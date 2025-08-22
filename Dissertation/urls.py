"""
URL configuration for Dissertation project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from dashboard import views as view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', view.home, name='home'),
    path('results/', view.results, name='results'),
    path('prediction/', view.prediction, name='prediction'),
    path('standings/', view.standings, name='standings'),
    path('statistics/', view.statistics, name='statistics'),
    path('driver-analytics/', view.driver_analytics, name='driver_analytics'),
    path('team-analytics/', view.team_analytics, name='team_analytics'),
    path('credits/', view.credits, name='credits'),
    path('portfolio/', view.portfolio, name='portfolio'),
    path('circuits/', view.circuits, name='circuits'),
    path('circuit/<int:circuit_id>/', view.circuit_detail, name='circuit_detail'),
    path('mark_circuit_visited/<int:circuit_id>/', view.mark_circuit_visited, name='mark_circuit_visited'),
    path('betting/', view.betting, name='betting'),
    path('place_bet/', view.place_bet, name='place_bet'),
    path('get_odds/', view.get_real_time_odds, name='get_real_time_odds'),
    path('market_depth/<int:event_id>/', view.market_depth, name='market_depth'),
    path('get_market_depth/<int:market_maker_id>/', view.get_market_depth_data, name='get_market_depth_data'),
    path('place_market_order/', view.place_market_order, name='place_market_order'),
    path('risk_settings/', view.risk_settings, name='risk_settings'),
    path('live_updates/', view.live_updates, name='live_updates'),
    # Mock Race Simulation URLs
    path('api/mock-race/start/', view.start_mock_race, name='start_mock_race'),
    path('api/mock-race/status/', view.get_race_status, name='get_race_status'),
    path('api/mock-race/event/', view.trigger_race_event, name='trigger_race_event'),
    path('api/mock-race/results/', view.get_final_results, name='get_final_results'),
    path('api/mock-race/stop/', view.stop_mock_race, name='stop_mock_race'),
    path('my_bets/', view.my_bets, name='my_bets'),
    path('register/', view.register, name='register'),
    path('login/', view.login_view, name='login'),
    path('logout/', view.logout_view, name='logout'),
    path('activate/<uidb64>/<token>/', view.activate, name='activate'),
    path('forgot-password/', view.forgot_password, name='forgot_password'),
    path('verify-temp-password/', view.verify_temp_password, name='verify_temp_password'),
    path('reset-password/', view.reset_password, name='reset_password'),
    path('resend-temp-password/', view.resend_temp_password, name='resend_temp_password'),
    path('subscription/', view.subscription_management, name='subscription_management'),
]
