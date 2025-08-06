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
    path('credits/', view.credits, name='credits'),
    path('portfolio/', view.portfolio, name='portfolio'),
    path('circuits/', view.circuits, name='circuits'),
    path('circuit/<int:circuit_id>/', view.circuit_detail, name='circuit_detail'),
    path('betting/', view.betting, name='betting'),
    path('place_bet/', view.place_bet, name='place_bet'),
    path('get_odds/', view.get_real_time_odds, name='get_real_time_odds'),
    path('market_depth/<int:event_id>/', view.market_depth, name='market_depth'),
    path('get_market_depth/<int:market_maker_id>/', view.get_market_depth_data, name='get_market_depth_data'),
    path('place_market_order/', view.place_market_order, name='place_market_order'),
    path('risk_settings/', view.risk_settings, name='risk_settings'),
    path('live_updates/', view.live_updates, name='live_updates'),
    path('my_bets/', view.my_bets, name='my_bets'),
    path('register/', view.register, name='register'),
    path('login/', view.login_view, name='login'),
    path('logout/', view.logout_view, name='logout'),
    path('activate/<uidb64>/<token>/', view.activate, name='activate'),
]
