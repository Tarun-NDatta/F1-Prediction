import unittest
from importlib import import_module

MODULES = [
    'dashboard.tests.test_access',
    'dashboard.tests.test_access_more',
    'dashboard.tests.test_auth',
    'dashboard.tests.test_bet_settlement',
    'dashboard.tests.test_bets_stats_and_liquidity',
    'dashboard.tests.test_betting',
    'dashboard.tests.test_circuits_credits',
    'dashboard.tests.test_odds_edge_cases',
    'dashboard.tests.test_pages',
    'dashboard.tests.test_pages_more',
    'dashboard.tests.test_prediction_edge_cases',
    'dashboard.tests.test_prediction_ranks',
    'dashboard.tests.test_predictions',
    'dashboard.tests.test_ui_smoke',
]


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for mod_name in MODULES:
        try:
            mod = import_module(mod_name)
            suite.addTests(loader.loadTestsFromModule(mod))
        except Exception:
            # If a module import fails, continue with available tests
            continue
    return suite

