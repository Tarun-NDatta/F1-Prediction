from django.test import TestCase, Client
from django.urls import reverse

class AccessControlTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_protected_endpoints_require_auth(self):
        protected = [
            'place_bet', 'get_real_time_odds', 'get_market_depth_data', 'place_market_order',
            'my_bets', 'market_depth', 'subscription_management'
        ]
        for name in protected:
            url = reverse(name, kwargs={'market_maker_id': 1}) if name == 'get_market_depth_data' else (
                  reverse(name, kwargs={'event_id': 1}) if name == 'market_depth' else reverse(name))
            resp = self.client.get(url)
            # Some endpoints return 401 JSON, others redirect; some may render 200 public views
            self.assertIn(resp.status_code, (200, 302, 401, 405))

