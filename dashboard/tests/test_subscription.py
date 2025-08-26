from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User

class SubscriptionTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='sub', password='pass', email='s@e.com', is_active=True)
        self.client.login(username='sub', password='pass')

    def test_upgrade_and_cancel(self):
        # GET page
        resp = self.client.get(reverse('subscription_management'))
        self.assertEqual(resp.status_code, 200)
        # Upgrade to PREMIUM
        resp = self.client.post(reverse('subscription_management'), data={'action': 'upgrade', 'tier': 'PREMIUM'})
        self.assertEqual(resp.status_code, 302)
        # Check profile
        self.user.refresh_from_db()
        profile = self.user.profile
        self.assertEqual(profile.subscription_tier, 'PREMIUM')
        self.assertTrue(profile.is_subscription_active)
        # Cancel back to BASIC
        resp = self.client.post(reverse('subscription_management'), data={'action': 'cancel'})
        self.assertEqual(resp.status_code, 302)
        self.user.refresh_from_db()
        self.assertEqual(self.user.profile.subscription_tier, 'BASIC')

