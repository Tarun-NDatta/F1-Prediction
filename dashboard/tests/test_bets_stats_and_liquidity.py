from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from datetime import date

from data.models import Circuit, Event, Bet, MarketMaker, Driver, Team

class BetsStatsAndLiquidityTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='u', password='p', email='u@e.com', is_active=True)
        self.client.login(username='u', password='p')
        self.circuit = Circuit.objects.create(name='Stat Circuit', country='X')
        self.event = Event.objects.create(name='Stat GP', year=2025, round=1, date=date(2025,1,1), circuit=self.circuit)
        self.driver = Driver.objects.create(given_name='D', family_name='One', permanent_number=1, driver_ref='d_ref', driver_id='d_id', nationality='X')
        self.team = Team.objects.create(name='T')
        # Ensure profile has credits
        self.user.profile.credits = 10000
        self.user.profile.save()

    def test_my_bets_stats(self):
        # Create bets with different statuses
        Bet.objects.create(user=self.user, event=self.event, bet_type='PODIUM_FINISH', driver=self.driver, credits_staked=200, odds=2.0, potential_payout=400, status='PENDING')
        Bet.objects.create(user=self.user, event=self.event, bet_type='PODIUM_FINISH', driver=self.driver, credits_staked=300, odds=1.5, potential_payout=450, status='WON', payout_received=450)
        Bet.objects.create(user=self.user, event=self.event, bet_type='PODIUM_FINISH', driver=self.driver, credits_staked=500, odds=3.0, potential_payout=1500, status='LOST', payout_received=0)
        resp = self.client.get(reverse('my_bets'))
        self.assertEqual(resp.status_code, 200)
        ctx = resp.context
        self.assertEqual(ctx['total_bets'], 3)
        self.assertEqual(ctx['won_bets'], 1)
        self.assertEqual(int(ctx['win_rate']), int(100 * 1/3))
        self.assertEqual(ctx['total_wagered'], 200+300+500)
        self.assertEqual(ctx['total_won'], 450)
        self.assertEqual(ctx['net_profit'], 450 - (200+300+500))

    def test_market_maker_liquidity_and_volume(self):
        mm = MarketMaker.objects.create(event=self.event, bet_type='PODIUM_FINISH', driver=self.driver, team=None, current_odds=2.0, base_odds=2.0, available_liquidity=1000, max_exposure=500)
        # Place two bets via view to update state and deduct credits
        payload = {'event_id': str(self.event.id), 'bet_type': 'PODIUM_FINISH', 'driver_id': str(self.driver.id), 'credits_staked': '200'}
        resp = self.client.post(reverse('place_bet'), data=payload)
        self.assertEqual(resp.status_code, 200)
        resp = self.client.post(reverse('place_bet'), data=payload)
        self.assertEqual(resp.status_code, 200)
        mm.refresh_from_db()
        # total_volume tracks sum of stakes; total_bets equals count of bets placed
        self.assertEqual(mm.total_volume, 400)
        self.assertEqual(mm.total_bets, 2)
        # Check market depth recent_activity length is <= 10 and contains entries
        depth = mm.get_market_depth()
        self.assertTrue(1 <= len(depth['recent_activity']) <= 10)

