from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from datetime import date, datetime

from data.models import Circuit, Event, SessionType, Session, Driver, Team, CatBoostPrediction, TrackSpecialization

class PredictionsTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='u', password='p')
        self.client.login(username='u', password='p')
        self.circuit = Circuit.objects.create(name='Test Circuit', country='Nowhere')
        self.event = Event.objects.create(name='Test GP', year=2025, round=1, date=date(2025,1,1), circuit=self.circuit)
        st = SessionType.objects.create(name='Race', session_type='RACE')
        self.session = Session.objects.create(event=self.event, session_type=st, date=datetime(2025,1,1,15,0,0))
        self.driver = Driver.objects.create(given_name='Test', family_name='Driver', permanent_number=99, driver_ref='td_ref', driver_id='td_id', nationality='X')
        self.team = Team.objects.create(name='Test Team')
        # At least one prediction row
        CatBoostPrediction.objects.create(driver=self.driver, event=self.event, year=2025, round_number=1, predicted_position=2.4, model_name='catboost_ensemble')

    def test_prediction_page_renders_and_ranking(self):
        resp = self.client.get(reverse('prediction') + '?model=catboost')
        self.assertEqual(resp.status_code, 200)
        # context contains per_race_labels / mae arrays
        self.assertIn('per_race_labels', resp.context)
        self.assertIn('per_race_mae', resp.context)
        # chart bootstrap in HTML
        self.assertContains(resp, 'MAE (lower is better)')

    def test_bias_correction_does_not_crash(self):
        # Add track specialization and ensure no crash
        TrackSpecialization.objects.create(circuit=self.circuit, category='STREET')
        resp = self.client.get(reverse('prediction') + '?model=catboost')
        self.assertEqual(resp.status_code, 200)

