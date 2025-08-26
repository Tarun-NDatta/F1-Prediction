from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from data.models import Event, Session, SessionType, Driver, Team, RaceResult, ridgeregression

class PredictionViewTests(TestCase):
    def setUp(self):
        # Minimal data setup for predictions view
        self.user = User.objects.create_user(username='u', password='p')
        self.client = Client()
        self.client.login(username='u', password='p')

        self.team = Team.objects.create(name='Test Team')
        self.driver = Driver.objects.create(given_name='Test', family_name='Driver', permanent_number=99)
        # Event requires date and circuit
        from data.models import Circuit
        self.circuit = Circuit.objects.create(name='Test Circuit', country='Nowhere')
        from datetime import date
        self.event = Event.objects.create(name='Test Grand Prix', year=2025, round=1, date=date(2025,1,1), circuit=self.circuit)
        # Create basic session type + race session
        st = SessionType.objects.create(name='Race', session_type='RACE')
        from datetime import datetime
        self.session = Session.objects.create(event=self.event, session_type=st, date=datetime(2025,1,1,15,0,0))

        # Actual result
        RaceResult.objects.create(session=self.session, driver=self.driver, team=self.team, position=5, points=10)
        # Prediction
        ridgeregression.objects.create(event=self.event, year=2025, round_number=1, driver=self.driver, predicted_position=7)

    def test_prediction_view_renders(self):
        resp = self.client.get(reverse('prediction'))
        self.assertEqual(resp.status_code, 200)
        # Context includes results and comparison labels for charts
        self.assertIn('results', resp.context)
        self.assertIn('comparison_labels', resp.context)

class RaceIncidentModelTests(TestCase):
    def test_create_incident(self):
        from dashboard.models import RaceIncident
        inc = RaceIncident.objects.create(
            year=2025, round=1, event_name='Test Grand Prix',
            type='START_ISSUE', description='Brake fire prevented start',
            driver_name='Carlos Sainz', lap=None
        )
        self.assertEqual(str(inc), '2025 R1 - Test Grand Prix: START_ISSUE (Carlos Sainz)')
