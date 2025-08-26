from django.test import TestCase
from django.contrib.auth.models import User
from datetime import date, datetime

from data.models import (
    Event, Session, SessionType, Driver, Team, Circuit, Bet
)


class BetSettlementTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='bettor', password='secret')
        self.profile = self.user.profile
        self.profile.credits = 5000
        self.profile.save()

        self.circuit = Circuit.objects.create(name='Test Circuit', country='Nowhere')
        self.event = Event.objects.create(name='Test GP', year=2025, round=1, date=date(2025,1,1), circuit=self.circuit)
        st = SessionType.objects.create(name='Race', session_type='RACE')
        self.session = Session.objects.create(event=self.event, session_type=st, date=datetime(2025,1,1,15,0,0))
        self.team = Team.objects.create(name='Test Team')
        self.driver = Driver.objects.create(given_name='Test', family_name='Driver', permanent_number=99)

    def test_settle_winning_bet_credits_payout(self):
        bet = Bet.objects.create(
            user=self.user,
            event=self.event,
            bet_type='PODIUM_FINISH',
            driver=self.driver,
            credits_staked=500,
            odds=2.0,
            potential_payout=1000,
            status='PENDING'
        )
        # Simulate stake deduction as place_bet would do
        self.profile.credits -= 500
        self.profile.save()

        settled = bet.settle(True, actual_result='P2')
        self.assertTrue(settled)
        bet.refresh_from_db()
        self.profile.refresh_from_db()
        self.assertEqual(bet.status, 'WON')
        self.assertEqual(bet.payout_received, 1000)
        self.assertEqual(self.profile.credits, 5500)  # 5000 - 500 + 1000

    def test_settle_losing_bet_no_payout(self):
        bet = Bet.objects.create(
            user=self.user,
            event=self.event,
            bet_type='PODIUM_FINISH',
            driver=self.driver,
            credits_staked=500,
            odds=2.0,
            potential_payout=1000,
            status='PENDING'
        )
        # Simulate stake deduction as place_bet would do
        self.profile.credits -= 500
        self.profile.save()

        settled = bet.settle(False, actual_result='P11')
        self.assertTrue(settled)
        bet.refresh_from_db()
        self.profile.refresh_from_db()
        self.assertEqual(bet.status, 'LOST')
        self.assertEqual(bet.payout_received, 0)
        self.assertEqual(self.profile.credits, 4500)

