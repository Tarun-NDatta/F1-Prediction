from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from data.models import Event, Driver, Team, Bet, UserProfile
from decimal import Decimal
import random

class Command(BaseCommand):
    help = 'Create sample bets for testing the betting system'

    def handle(self, *args, **options):
        # Get or create a test user
        user, created = User.objects.get_or_create(
            username='testuser',
            defaults={'email': 'test@example.com'}
        )
        
        # Get or create user profile
        profile, created = UserProfile.objects.get_or_create(
            user=user,
            defaults={'credits': 10000}
        )
        
        # Get some events, drivers, and teams
        events = list(Event.objects.all()[:5])
        drivers = list(Driver.objects.all()[:10])
        teams = list(Team.objects.all()[:5])
        
        if not events or not drivers or not teams:
            self.stdout.write(
                self.style.ERROR('Not enough data to create sample bets. Please ensure you have events, drivers, and teams.')
            )
            return
        
        bet_types = ['podium', 'position', 'dnf', 'qualifying', 'fastest_lap', 'weather']
        bet_statuses = ['pending', 'won', 'lost']
        
        # Create sample bets
        bets_created = 0
        for i in range(10):
            event = random.choice(events)
            driver = random.choice(drivers)
            team = random.choice(teams)
            bet_type = random.choice(bet_types)
            status = random.choice(bet_statuses)
            credits_staked = random.randint(50, 500)
            odds = round(random.uniform(1.5, 5.0), 2)
            
            # Calculate credits won if bet was won
            credits_won = 0
            if status == 'won':
                credits_won = int(credits_staked * odds)
            
            bet = Bet.objects.create(
                user=user,
                event=event,
                bet_type=bet_type,
                driver=driver,
                team=team,
                position=random.randint(1, 20) if bet_type == 'position' else None,
                credits_staked=credits_staked,
                odds=odds,
                status=status,
                credits_won=credits_won
            )
            bets_created += 1
        
        # Update user profile statistics
        total_bets = Bet.objects.filter(user=user).count()
        won_bets = Bet.objects.filter(user=user, status='won').count()
        total_wagered = sum(bet.credits_staked for bet in Bet.objects.filter(user=user))
        total_won = sum(bet.credits_won for bet in Bet.objects.filter(user=user, status='won'))
        
        profile.total_bets_placed = total_bets
        profile.total_credits_won = total_won
        profile.total_credits_lost = total_wagered - total_won
        profile.save()
        
        self.stdout.write(
            self.style.SUCCESS(f'Successfully created {bets_created} sample bets for user {user.username}')
        )
        self.stdout.write(f'Total bets: {total_bets}')
        self.stdout.write(f'Won bets: {won_bets}')
        self.stdout.write(f'Total wagered: {total_wagered} credits')
        self.stdout.write(f'Total won: {total_won} credits') 