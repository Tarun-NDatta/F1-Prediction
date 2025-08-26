from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from datetime import date, datetime

from data.models import Circuit, Event, SessionType, Session, Driver, Team, Bet

class MorePageTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='u', password='p')
        self.client.login(username='u', password='p')
        self.circuit = Circuit.objects.create(name='Test Circuit', country='Nowhere')
        self.event = Event.objects.create(name='Test GP', year=2025, round=1, date=date(2025,1,1), circuit=self.circuit)
        st = SessionType.objects.create(name='Race', session_type='RACE')
        self.session = Session.objects.create(event=self.event, session_type=st, date=datetime(2025,1,1,15,0,0))
        self.team = Team.objects.create(name='TeamX')
        self.driver = Driver.objects.create(given_name='First', family_name='Last', permanent_number=77)

    def test_driver_and_team_analytics_pages(self):
        resp = self.client.get(reverse('driver_analytics'))
        self.assertEqual(resp.status_code, 200)
        resp = self.client.get(reverse('team_analytics'))
        self.assertEqual(resp.status_code, 200)

    def test_circuits_and_detail_pages(self):
        resp = self.client.get(reverse('circuits'))
        self.assertEqual(resp.status_code, 200)
        resp = self.client.get(reverse('circuit_detail', args=[self.circuit.id]))
        self.assertEqual(resp.status_code, 200)

    def test_my_bets_context_splits_statuses(self):
        # Create different status bets
        Bet.objects.create(user=self.user, event=self.event, bet_type='PODIUM_FINISH', driver=self.driver, credits_staked=100, odds=2.0, potential_payout=200, status='PENDING')
        Bet.objects.create(user=self.user, event=self.event, bet_type='PODIUM_FINISH', driver=self.driver, credits_staked=150, odds=2.0, potential_payout=300, status='WON')
        Bet.objects.create(user=self.user, event=self.event, bet_type='PODIUM_FINISH', driver=self.driver, credits_staked=200, odds=2.0, potential_payout=400, status='LOST')
        resp = self.client.get(reverse('my_bets'))
        self.assertEqual(resp.status_code, 200)
        self.assertIn('active_bets', resp.context)
        self.assertIn('completed_bets', resp.context)
        self.assertEqual(resp.context['active_bets'].count(), 1)
        self.assertEqual(resp.context['completed_bets'].count(), 2)

