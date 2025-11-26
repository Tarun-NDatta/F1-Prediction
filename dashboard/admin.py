from django.contrib import admin
from .models import RaceIncident



from data.models import Bet, MarketMaker, MarketOrder
from django.contrib import messages

# Avoid duplicate registration if data app already registered these models
for model in [Bet, MarketMaker, MarketOrder]:
    try:
        admin.site.unregister(model)
    except Exception:
        pass


@admin.action(description="Settle selected bets as WON (credit payouts)")
def settle_as_won(modeladmin, request, queryset):
    count = 0
    for bet in queryset:
        if bet.status == 'PENDING':
            bet.settle(True)
            count += 1
    messages.success(request, f"Settled {count} bet(s) as WON.")

@admin.action(description="Settle selected bets as LOST")
def settle_as_lost(modeladmin, request, queryset):
    count = 0
    for bet in queryset:
        if bet.status == 'PENDING':
            bet.settle(False)
            count += 1
    messages.info(request, f"Settled {count} bet(s) as LOST.")

@admin.register(Bet)
class BetAdmin(admin.ModelAdmin):
    list_display = ('user', 'event', 'bet_type', 'credits_staked', 'odds', 'status', 'payout_received', 'created_at')
    list_filter = ('status', 'bet_type', 'event')
    search_fields = ('user__username', 'event__name')
    actions = [settle_as_won, settle_as_lost]

@admin.register(MarketMaker)
class MarketMakerAdmin(admin.ModelAdmin):
    list_display = ('event', 'bet_type', 'driver', 'team', 'current_odds', 'available_liquidity', 'total_volume', 'total_bets')
    search_fields = ('event__name', 'driver__family_name', 'team__name')

@admin.register(MarketOrder)
class MarketOrderAdmin(admin.ModelAdmin):
    list_display = ('user', 'market_maker', 'order_type', 'side', 'amount', 'status', 'average_price', 'created_at')
    list_filter = ('status', 'order_type', 'side')
