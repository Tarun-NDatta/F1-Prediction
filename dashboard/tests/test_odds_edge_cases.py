from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from datetime import date, datetime

from data.models import Circuit, Event, SessionType, Session, Driver, Team, RaceResult

class OddsEdgeCaseTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='u', password='p')
        self.client.login(username='u', password='p')
        self.circuit = Circuit.objects.create(name='Odds Circuit', country='Nowhere')
        self.event = Event.objects.create(name='Odds GP', year=2025, round=1, date=date(2025,1,1), circuit=self.circuit)
        st = SessionType.objects.create(name='Race', session_type='RACE')
        self.session = Session.objects.create(event=self.event, session_type=st, date=datetime(2025,1,1,15,0,0))
        self.team = Team.objects.create(name='TeamO')
        self.driver = Driver.objects.create(given_name='A', family_name='A', permanent_number=10, driver_ref='oa_ref', driver_id='oa_id', nationality='X')
        # Minimal history to avoid divide-by-zero; DNF entries etc.
        RaceResult.objects.create(session=self.session, driver=self.driver, team=self.team, position=10, status='Finished', points=1)

    def test_real_time_odds_bounds(self):
        url = reverse('get_real_time_odds')
        payload = {'event_id': str(self.event.id), 'bet_type': 'DNF_PREDICTION', 'driver_id': str(self.driver.id)}
        resp = self.client.post(url, data=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json().get('success'))
        odds = float(resp.json()['odds'])
        self.assertTrue(1.1 <= odds <= 50.0)

    def test_head_to_head_missing_opponent(self):
        url = reverse('get_real_time_odds')
        payload = {'event_id': str(self.event.id), 'bet_type': 'HEAD_TO_HEAD', 'driver_id': str(self.driver.id)}
        resp = self.client.post(url, data=payload)
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json().get('success'))

