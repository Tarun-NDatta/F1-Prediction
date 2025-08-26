from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from datetime import date, datetime

from data.models import Circuit, Event, SessionType, Session, Driver, Team, ridgeregression

class PredictionEdgeCaseTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='u', password='p')
        self.client.login(username='u', password='p')
        self.circuit = Circuit.objects.create(name='EC Circuit', country='Nowhere')
        self.event = Event.objects.create(name='EC GP', year=2025, round=1, date=date(2025,1,1), circuit=self.circuit)
        st = SessionType.objects.create(name='Race', session_type='RACE')
        self.session = Session.objects.create(event=self.event, session_type=st, date=datetime(2025,1,1,12,0,0))
        self.driver = Driver.objects.create(given_name='A', family_name='A', permanent_number=10, driver_ref='ea_ref', driver_id='ea_id', nationality='X')
        self.team = Team.objects.create(name='T')
        ridgeregression.objects.create(driver=self.driver, event=self.event, year=2025, round_number=1, predicted_position=5.0, model_name='ridge_regression')

    def test_model_switch_and_labels_mae_alignment(self):
        # Ridge
        resp = self.client.get(reverse('prediction') + '?model=ridge_regression')
        self.assertEqual(resp.status_code, 200)
        self.assertIn('per_race_labels', resp.context)
        self.assertIn('per_race_mae', resp.context)
        labels = resp.context['per_race_labels']
        mae = resp.context['per_race_mae']
        self.assertEqual(len(labels), len(mae))
        # Switch to catboost (may not have data; page should still render)
        resp = self.client.get(reverse('prediction') + '?model=catboost')
        self.assertEqual(resp.status_code, 200)

