# data/management/commands/race_results.py
import os
import time
from datetime import timedelta
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.dateparse import parse_date
import fastf1
from data.models import Circuit, Team, Driver, Event, Session, SessionType, RaceResult


class Command(BaseCommand):
    help = 'Fetch and store F1 race results using FastF1'

    def add_arguments(self, parser):
        parser.add_argument('--years', nargs='+', type=int, default=[2022, 2023, 2024],
                            help='Years to fetch data for (space separated)')
        parser.add_argument('--force', action='store_true',
                            help='Force fresh download ignoring cache')
        parser.add_argument('--rounds', nargs='+', type=int, default=None,
                            help='Specific rounds to fetch (space separated)')

    def handle(self, *args, **options):
        cache_dir = 'fastf1_cache'
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
        self.stdout.write(f"Using cache directory: {os.path.abspath(cache_dir)}")

        years = options['years']
        force_refresh = options['force']
        specific_rounds = options['rounds']

        # Ensure session type exists for race
        session_type, _ = SessionType.objects.get_or_create(
            session_type='RACE', defaults={'name': 'Race'}
        )

        for year in years:
            self.stdout.write(self.style.SUCCESS(f"Processing season {year}..."))
            try:
                schedule = fastf1.get_event_schedule(year, include_testing=False)
                events = schedule[schedule['EventFormat'] == 'conventional']

                if specific_rounds:
                    events = events[events['RoundNumber'].isin(specific_rounds)]
                    self.stdout.write(f"Filtered to rounds: {specific_rounds}")

                for i, (_, event_data) in enumerate(events.iterrows(), 1):
                    round_num = event_data['RoundNumber']
                    self.stdout.write(f"[{i}/{len(events)}] Round {round_num}: {event_data['EventName']}")

                    session = fastf1.get_session(year, round_num, 'R')

                    load_params = {'telemetry': False, 'weather': False, 'messages': False}
                    if force_refresh:
                        for param in ['force_rerun', 'force', 'force_restore']:
                            try:
                                load_params[param] = True
                                session.load(**load_params)
                                break
                            except TypeError:
                                load_params.pop(param)
                        else:
                            session.load(**load_params)
                    else:
                        session.load(**load_params)

                    self.process_race_results(session, event_data, session_type)
                    time.sleep(0.5)

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error processing season {year}: {e}"))

    def process_race_results(self, session, event_data, session_type):
        circuit, _ = Circuit.objects.update_or_create(
            circuit_ref=event_data.get('CircuitKey', event_data['Location'].lower().replace(' ', '_')),
            defaults={
                'name': event_data['Location'],
                'location': event_data['Location'],
                'country': event_data['Country'],
            }
        )

        with transaction.atomic():
            event_obj, created = Event.objects.update_or_create(
                year=event_data['EventDate'].year,
                round=event_data['RoundNumber'],
                defaults={
                    'name': event_data['EventName'],
                    'official_name': event_data.get('OfficialEventName', ''),
                    'date': parse_date(event_data['EventDate'].strftime('%Y-%m-%d')),
                    'circuit': circuit,
                    'event_format': 'CONVENTIONAL',
                }
            )

            session_obj, _ = Session.objects.update_or_create(
                event=event_obj,
                session_type=session_type,
                defaults={'date': session.date}
            )

            results = session.results
            if results is None or results.empty:
                self.stdout.write(self.style.WARNING(f"No race results for {event_obj}"))
                return

            for _, row in results.iterrows():
                team, _ = Team.objects.get_or_create(
                    team_ref=row['TeamName'].lower().replace(' ', '_'),
                    defaults={'name': row['TeamName']}
                )

                driver, _ = Driver.objects.update_or_create(
                    driver_ref=row['Abbreviation'],
                    defaults={
                        'driver_id': row['DriverNumber'],
                        'given_name': row['FirstName'],
                        'family_name': row['LastName'],
                        'nationality': row.get('Country', ''),
                        'code': row['Abbreviation'],
                        'permanent_number': row['DriverNumber'],
                    }
                )

                def to_duration(val):
                    return val if isinstance(val, timedelta) else None

                grid_pos = row.get('Grid')
                finish_pos = row.get('Position')

                position_gain = None
                if grid_pos is not None and finish_pos is not None:
                    try:
                        position_gain = int(grid_pos) - int(finish_pos)
                    except Exception:
                        position_gain = None

                RaceResult.objects.update_or_create(
                    session=session_obj,
                    driver=driver,
                    defaults={
                        'team': team,
                        'position': int(row['Position']) if row['Position'] else None,
                        'position_text': str(row.get('Status', '')),
                        'points': float(row.get('Points', 0)),
                        'status': row.get('Status', ''),
                        'laps': row.get('Laps'),
                        'time': to_duration(row.get('Time')),
                        'fastest_lap_rank': row.get('FastestLapRank'),
                        'fastest_lap_time': to_duration(row.get('FastestLapTime')),
                        'fastest_lap_speed': row.get('FastestLapSpeed'),
                        'pit_stops': row.get('PitStops'),
                        'grid_position': grid_pos,
                        'position_gain': position_gain,
                        'tyre_stints': row.get('TyreStints'),
                    }
                )
            self.stdout.write(self.style.SUCCESS(f"Saved race results for {event_obj}"))
