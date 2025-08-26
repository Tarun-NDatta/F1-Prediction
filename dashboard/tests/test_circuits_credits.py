from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from datetime import date

from data.models import Circuit, Event

class CircuitVisitCreditsTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='u', password='p')
        self.client.login(username='u', password='p')
        self.circuit = Circuit.objects.create(name='Alpha Circuit', country='X')
        Event.objects.create(name='Alpha GP', year=2025, round=1, date=date(2025,1,1), circuit=self.circuit)

    def test_mark_circuit_visited_awards_credits_once(self):
        profile = self.user.profile
        start = profile.credits
        url = reverse('mark_circuit_visited', args=[self.circuit.id])
        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get('status'), 'success')
        profile.refresh_from_db()
        self.assertEqual(profile.credits, start + 100)
        # Visit again: no double-award
        resp = self.client.post(url)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get('status'), 'already_visited')
        profile.refresh_from_db()
        self.assertEqual(profile.credits, start + 100)

