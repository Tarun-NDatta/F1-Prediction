from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from datetime import date

from data.models import Circuit, Event

class UISmokeTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='u', password='p', email='u@e.com', is_active=True)

    def test_betting_requires_login(self):
        resp = self.client.get(reverse('betting'))
        self.assertEqual(resp.status_code, 302)
        self.client.login(username='u', password='p')
        # Create minimal event for betting page
        circuit = Circuit.objects.create(name='Test Circuit', country='Nowhere')
        Event.objects.create(name='Test GP', year=2025, round=1, date=date(2025,1,1), circuit=circuit)
        resp = self.client.get(reverse('betting'))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Prediction Market')
        self.assertContains(resp, 'Credits to Stake')

    def test_market_depth_page(self):
        self.client.login(username='u', password='p')
        circuit = Circuit.objects.create(name='C1', country='X')
        ev = Event.objects.create(name='E1', year=2025, round=1, date=date(2025,1,1), circuit=circuit)
        resp = self.client.get(reverse('market_depth', args=[ev.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Market Depth')

