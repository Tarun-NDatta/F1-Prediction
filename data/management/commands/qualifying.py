# data/management/commands/qualifying.py
import os
import time
from datetime import timedelta
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.dateparse import parse_date
import fastf1
from data.models import Circuit, Team, Driver, Event, Session, SessionType, QualifyingResult


class Command(BaseCommand):
    help = 'Fetch and store F1 qualifying results using FastF1'

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

        # Ensure session type exists for qualifying
        session_type, _ = SessionType.objects.get_or_create(
            session_type='QUALIFYING', defaults={'name': 'Qualifying'}
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

                    session = fastf1.get_session(year, round_num, 'Q')

                    load_params = {'telemetry': False, 'weather': False, 'messages': False}
                    # Try force reload if requested
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

                    self.process_qualifying(session, event_data, session_type)
                    time.sleep(0.5)

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error processing season {year}: {e}"))

    def process_qualifying(self, session, event_data, session_type):
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

            qualifying_results = session.results
            if qualifying_results is None or qualifying_results.empty:
                self.stdout.write(self.style.WARNING(f"No qualifying data for {event_obj}"))
                return

            for _, row in qualifying_results.iterrows():
                team, _ = Team.objects.get_or_create(
                    team_ref=row['TeamName'].lower().replace(' ', '_'),
                    defaults={'name': row['TeamName']}
                )

                driver, created = Driver.objects.get_or_create(
                    driver_id=row['DriverNumber'],  # lookup by unique driver_id
                    defaults={
                        'driver_ref': row['Abbreviation'].strip(),
                        'given_name': row['FirstName'],
                        'family_name': row['LastName'],
                        'nationality': row.get('Country', ''),
                        'code': row['Abbreviation'],
                        'permanent_number': row['DriverNumber'],
                    }
                )

                # If it already existed, only update fields that might have changed:
                if not created:
                    driver.given_name = row['FirstName']
                    driver.family_name = row['LastName']
                    driver.nationality = row.get('Country', '')
                    driver.code = row['Abbreviation']
                    driver.permanent_number = row['DriverNumber']
                    driver.save()


                def to_duration(val):
                    return val if isinstance(val, timedelta) else None

                QualifyingResult.objects.update_or_create(
                    session=session_obj,
                    driver=driver,
                    defaults={
                        'team': team,
                        'position': int(row['Position']) if row['Position'] else None,
                        'q1': to_duration(row.get('Q1')),
                        'q2': to_duration(row.get('Q2')),
                        'q3': to_duration(row.get('Q3')),
                        'q1_millis': row.get('Q1Millis'),
                        'q2_millis': row.get('Q2Millis'),
                        'q3_millis': row.get('Q3Millis'),
                        'pole_delta': None,  # You can add logic to calculate if needed
                        'status': row.get('Status', ''),
                        'laps': row.get('Laps', None),
                    }
                )

            self.stdout.write(self.style.SUCCESS(f"Saved qualifying results for {event_obj}"))
