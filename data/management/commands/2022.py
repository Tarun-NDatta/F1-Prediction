import requests
from django.core.management.base import BaseCommand
from data.models import Circuit, Race, Driver, Team, Result
from datetime import datetime

class Command(BaseCommand):
    help = 'Fetch and save all 2022 F1 race results from Ergast API'

    def handle(self, *args, **options):
        url = 'http://ergast.com/api/f1/2022/results.json?limit=1000'
        response = requests.get(url)
        if response.status_code != 200:
            self.stdout.write(self.style.ERROR('Failed to fetch data'))
            return
        
        data = response.json()
        races = data['MRData']['RaceTable']['Races']
        
        for race_data in races:
            # Circuit
            circuit_data = race_data['Circuit']
            circuit, _ = Circuit.objects.get_or_create(
                name=circuit_data['circuitName'],
                defaults={
                    'location': circuit_data['Location']['locality'],
                    'country': circuit_data['Location']['country'],
                }
            )

            # Race
            race_date = datetime.strptime(race_data['date'], '%Y-%m-%d').date()
            race, _ = Race.objects.get_or_create(
                year=2022,
                round=race_data['round'],
                name=race_data['raceName'],
                date=race_date,
                circuit=circuit
            )
            
            # Results
            for res in race_data['Results']:
                # Driver
                driver_data = res['Driver']
                driver, _ = Driver.objects.get_or_create(
                    driver_id=driver_data['driverId'],
                    defaults={
                        'given_name': driver_data['givenName'],
                        'family_name': driver_data['familyName'],
                        'nationality': driver_data['nationality'],
                    }
                )
                # Team
                team_name = res['Constructor']['name']
                team, _ = Team.objects.get_or_create(name=team_name)

                # Result entry
                Result.objects.update_or_create(
                    race=race,
                    driver=driver,
                    defaults={
                        'team': team,
                        'position': int(res.get('position')) if res.get('position') else None,
                        'points': float(res.get('points')) if res.get('points') else 0.0,
                        'laps': int(res.get('laps')) if res.get('laps') else None,
                        'status': res.get('status'),
                        'time': res.get('Time', {}).get('time') if res.get('Time') else None,
                    }
                )
            
            self.stdout.write(self.style.SUCCESS(f"Saved race: {race.name}"))
        
        self.stdout.write(self.style.SUCCESS('Finished fetching all 2022 race results'))
