from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User

from data.models import (
    Event, Session, SessionType, Driver, Team, RaceResult,
    CatBoostPrediction, Circuit, UserProfile, Bet, MarketMaker, MarketOrder
)

from datetime import date, datetime


class BettingFlowTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='bettor', password='secret', email='b@e.com')
        self.client.login(username='bettor', password='secret')
        # Ensure profile exists with default credits
        self.profile = self.user.profile
        self.profile.credits = 5000
        self.profile.save()

        # Minimal domain data
        self.circuit = Circuit.objects.create(name='Test Circuit', country='Nowhere')
        self.event = Event.objects.create(name='Test GP', year=2025, round=1, date=date(2025,1,1), circuit=self.circuit)
        st = SessionType.objects.create(name='Race', session_type='RACE')
        self.session = Session.objects.create(event=self.event, session_type=st, date=datetime(2025,1,1,15,0,0))
        self.team = Team.objects.create(name='Test Team')
        self.driver = Driver.objects.create(given_name='Test', family_name='Driver', permanent_number=99)
        # A historical result to avoid divide-by-zero in odds
        RaceResult.objects.create(session=self.session, driver=self.driver, team=self.team, position=3, points=15)

        # One ML prediction for odds adjustment
        CatBoostPrediction.objects.create(
            driver=self.driver,
            event=self.event,
            year=2025,
            round_number=1,
            predicted_position=2.0,
            prediction_confidence=0.7,
            model_name='catboost_ensemble'
        )

    def test_place_bet_podium_success(self):
        url = reverse('place_bet')
        payload = {
            'event_id': str(self.event.id),
            'bet_type': 'PODIUM_FINISH',
            'driver_id': str(self.driver.id),
            'credits_staked': '500',
        }
        resp = self.client.post(url, data=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get('success'))
        self.assertIn('bet_id', data)

        bet = Bet.objects.get(id=data['bet_id'])
        self.profile.refresh_from_db()
        self.assertLess(self.profile.credits, 5000)  # deducted
        self.assertGreaterEqual(bet.potential_payout, 500)
        # MarketMaker created/updated
        self.assertEqual(MarketMaker.objects.filter(event=self.event, driver=self.driver, bet_type='PODIUM_FINISH').count(), 1)

    def test_place_bet_insufficient_credits(self):
        # Reduce credits so can_place_bet fails
        self.profile.credits = 100
        self.profile.save()
        url = reverse('place_bet')
        payload = {
            'event_id': str(self.event.id),
            'bet_type': 'PODIUM_FINISH',
            'driver_id': str(self.driver.id),
            'credits_staked': '1000',
        }
        resp = self.client.post(url, data=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data.get('success'))
        self.assertIn('error', data)
        self.assertIn('Insufficient', data['error'])

    def test_place_bet_liquidity_error(self):
        # Pre-create a market maker with very low liquidity
        MarketMaker.objects.create(
            event=self.event,
            bet_type='PODIUM_FINISH',
            driver=self.driver,
            team=None,
            current_odds=2.0,
            base_odds=2.0,
            available_liquidity=200,
            max_exposure=500,
        )
        url = reverse('place_bet')
        payload = {
            'event_id': str(self.event.id),
            'bet_type': 'PODIUM_FINISH',
            'driver_id': str(self.driver.id),
            'credits_staked': '500',
        }
        resp = self.client.post(url, data=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertFalse(data.get('success'))
        self.assertIn('liquidity', data.get('error','').lower())

    def test_get_real_time_odds_returns_ml_prediction(self):
        url = reverse('get_real_time_odds')
        payload = {
            'event_id': str(self.event.id),
            'bet_type': 'PODIUM_FINISH',
            'driver_id': str(self.driver.id),
        }
        resp = self.client.post(url, data=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get('success'))
        self.assertIn('odds', data)
        self.assertIn('market_stats', data)
        self.assertIn('ml_prediction', data)
        self.assertIsInstance(data['ml_prediction'].get('predicted_position'), float)

    def test_get_market_depth_data(self):
        # Need a market maker first
        mm = MarketMaker.objects.create(
            event=self.event,
            bet_type='PODIUM_FINISH',
            driver=self.driver,
            team=None,
            current_odds=2.0,
            base_odds=2.0,
            available_liquidity=10000,
            max_exposure=5000,
        )
        url = reverse('get_market_depth_data', args=[mm.id])
        resp = self.client.get(url)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get('success'))
        self.assertIn('depth_data', data)
        self.assertIn('odds_ladder', data['depth_data'])

    def test_place_market_order_limit_creates_pending_order(self):
        mm = MarketMaker.objects.create(
            event=self.event,
            bet_type='PODIUM_FINISH',
            driver=self.driver,
            team=None,
            current_odds=2.0,
            base_odds=2.0,
            available_liquidity=10000,
            max_exposure=5000,
        )
        url = reverse('place_market_order')
        payload = {
            'market_maker_id': str(mm.id),
            'order_type': 'LIMIT',
            'side': 'BUY',
            'amount': '250',
            'limit_price': '2.1',
        }
        resp = self.client.post(url, data=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data.get('success'))
        order = MarketOrder.objects.get(id=data['order_id'])
        self.assertEqual(order.status, 'PENDING')
        # Credits should not change for LIMIT order not executed
        self.profile.refresh_from_db()
        self.assertEqual(self.profile.credits, 5000)

