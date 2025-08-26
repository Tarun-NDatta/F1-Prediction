from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from data.models import Event, Session, SessionType, Driver, Team, RaceResult, ridgeregression, Circuit
from datetime import date, datetime

class PredictionPageContextTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='u', password='p')
        self.client = Client()
        self.client.login(username='u', password='p')

        self.circuit = Circuit.objects.create(name='Test Circuit', country='Nowhere')
        self.event = Event.objects.create(name='Test GP', year=2025, round=1, date=date(2025,1,1), circuit=self.circuit)
        st = SessionType.objects.create(name='Race', session_type='RACE')
        self.session = Session.objects.create(event=self.event, session_type=st, date=datetime(2025,1,1,15,0,0))
        self.team = Team.objects.create(name='Test Team')
        self.driver = Driver.objects.create(given_name='Test', family_name='Driver', permanent_number=99)
        RaceResult.objects.create(session=self.session, driver=self.driver, team=self.team, position=5, points=10)
        ridgeregression.objects.create(event=self.event, year=2025, round_number=1, driver=self.driver, predicted_position=7)

    def test_has_incident_notes_fields(self):
        resp = self.client.get(reverse('prediction'))
        self.assertEqual(resp.status_code, 200)
        self.assertIn('results', resp.context)
        # Ensure consistency and top10 fields exist
        self.assertIn('per_race_labels', resp.context)
        self.assertIn('per_race_mae', resp.context)
        self.assertIn('top10_hits', resp.context)
        self.assertIn('top10_misses', resp.context)

