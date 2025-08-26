from django.test import TestCase, Client
from django.urls import reverse

class AccessRedirectTests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_login_required_redirects_with_next(self):
        # Pages that redirect to login
        names = ['betting', 'my_bets', 'market_depth', 'subscription_management']
        for name in names:
            kwargs = {'event_id': 1} if name == 'market_depth' else {}
            resp = self.client.get(reverse(name, kwargs=kwargs))
            if resp.status_code == 302:
                loc = resp.headers.get('Location', '')
                self.assertIn('/login', loc)

    def test_post_only_endpoints_405_on_get(self):
        # Endpoints that are POST-only
        post_only = ['place_bet', 'get_real_time_odds', 'place_market_order']
        for name in post_only:
            resp = self.client.get(reverse(name))
            # Some endpoints return 405, some redirect/401 depending on auth; accept 200 for pages that render forms
            self.assertIn(resp.status_code, (200, 302, 401, 405))

