from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from datetime import date, datetime

from data.models import Circuit, Event, SessionType, Session, Driver, Team, ridgeregression, RaceResult

class PredictionRanksTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='u', password='p')
        self.client.login(username='u', password='p')
        self.circuit = Circuit.objects.create(name='Rank Circuit', country='Nowhere')
        self.event = Event.objects.create(name='Rank GP', year=2025, round=1, date=date(2025,1,1), circuit=self.circuit)
        st = SessionType.objects.create(name='Race', session_type='RACE')
        self.session = Session.objects.create(event=self.event, session_type=st, date=datetime(2025,1,1,15,0,0))
        self.team = Team.objects.create(name='TeamR')
        self.d1 = Driver.objects.create(given_name='A', family_name='A', permanent_number=11, driver_ref='a_ref', driver_id='a_id', nationality='X')
        self.d2 = Driver.objects.create(given_name='B', family_name='B', permanent_number=12, driver_ref='b_ref', driver_id='b_id', nationality='X')
        # Two predictions fractional
        ridgeregression.objects.create(driver=self.d1, event=self.event, year=2025, round_number=1, predicted_position=5.4, model_name='ridge_regression')
        ridgeregression.objects.create(driver=self.d2, event=self.event, year=2025, round_number=1, predicted_position=2.1, model_name='ridge_regression')
        # Actual results
        RaceResult.objects.create(session=self.session, driver=self.d1, team=self.team, position=6)
        RaceResult.objects.create(session=self.session, driver=self.d2, team=self.team, position=2)

    def test_predicted_ranks_are_integers_1_to_n(self):
        resp = self.client.get(reverse('prediction') + '?model=ridge_regression')
        self.assertEqual(resp.status_code, 200)
        results = resp.context['results']
        # Find our event
        comp = None
        for r in results:
            if r['event'].id == self.event.id:
                comp = r['comparison']
                break
        self.assertIsNotNone(comp)
        preds = [item['predicted'] for item in comp if item['actual'] != 'N/A']
        self.assertTrue(all(isinstance(p, int) for p in preds))
        self.assertTrue(all(1 <= p <= len(preds) for p in preds))

