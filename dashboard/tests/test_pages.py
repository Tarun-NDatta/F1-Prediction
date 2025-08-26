from django.test import TestCase, Client
from django.urls import reverse

class PageSmokeTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_public_pages_render(self):
        pages = ['home', 'results', 'prediction', 'standings', 'statistics', 'circuits']
        for name in pages:
            resp = self.client.get(reverse(name))
            self.assertEqual(resp.status_code, 200)

    def test_auth_pages_render(self):
        pages = ['register', 'login', 'forgot_password', 'subscription_management']
        for name in pages:
            resp = self.client.get(reverse(name))
            # subscription_management requires login; expect redirect
            if name == 'subscription_management':
                self.assertIn(resp.status_code, (302,))
            else:
                self.assertEqual(resp.status_code, 200)

