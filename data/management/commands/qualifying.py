import os
import time
from datetime import timedelta
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.dateparse import parse_date
from django.utils import timezone
import fastf1
from data.models import Circuit, Team, Driver, Event, Session, SessionType, QualifyingResult


class Command(BaseCommand):
    help = 'Fetch and store F1 qualifying results using FastF1'

    def add_arguments(self, parser):
        parser.add_argument('--years', nargs='+', type=int, default=[2022, 2023, 2024, 2025],
                            help='Years to fetch data for (space separated)')
        parser.add_argument('--year', type=int, default=None,
                            help='Single year to fetch data for (alternative to --years)')
        parser.add_argument('--force', action='store_true',
                            help='Force fresh download ignoring cache')
        parser.add_argument('--rounds', nargs='+', type=int, default=None,
                            help='Specific rounds to fetch (space separated)')
        parser.add_argument('--round', type=int, default=None,
                            help='Single round to fetch (alternative to --rounds)')
        parser.add_argument('--debug', action='store_true',
                            help='Show debug information about event formats')

    def handle(self, *args, **options):
        cache_dir = 'fastf1_cache'
        os.makedirs(cache_dir, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
        self.stdout.write(f"Using cache directory: {os.path.abspath(cache_dir)}")

        # Handle both singular and plural arguments
        if options['year']:
            years = [options['year']]
        else:
            years = options['years']

        if options['round']:
            specific_rounds = [options['round']]
        else:
            specific_rounds = options['rounds']

        force_refresh = options['force']
        debug_mode = options['debug']

        # Ensure session type exists for qualifying
        session_type, _ = SessionType.objects.get_or_create(
            session_type='QUALIFYING', defaults={'name': 'Qualifying'}
        )

        for year in years:
            self.stdout.write(self.style.SUCCESS(f"Processing season {year}..."))
            try:
                schedule = fastf1.get_event_schedule(year, include_testing=False)
                
                if debug_mode:
                    self.stdout.write("Available event formats:")
                    self.stdout.write(str(schedule['EventFormat'].unique()))
                    self.stdout.write("\nAll events:")
                    for _, event in schedule.iterrows():
                        self.stdout.write(f"Round {event['RoundNumber']}: {event['EventName']} - Format: {event.get('EventFormat', 'Unknown')}")
                
                # Don't filter by EventFormat - include all race weekends
                # Qualifying ('Q') session exists for both conventional and sprint weekends
                events = schedule

                if specific_rounds:
                    events = events[events['RoundNumber'].isin(specific_rounds)]
                    self.stdout.write(f"Filtered to rounds: {specific_rounds}")

                for i, (_, event_data) in enumerate(events.iterrows(), 1):
                    round_num = event_data['RoundNumber']
                    event_format = event_data.get('EventFormat', 'conventional')
                    
                    self.stdout.write(f"[{i}/{len(events)}] Round {round_num}: {event_data['EventName']} (Format: {event_format})")

                    try:
                        # Always get the main qualifying session ('Q') - exists for all weekend formats
                        session = fastf1.get_session(year, round_num, 'Q')

                        # Load session data with minimal overhead
                        load_params = {'telemetry': False, 'weather': False, 'messages': False}
                        if force_refresh:
                            # Try different force parameters depending on FastF1 version
                            for param in ['force_rerun', 'force', 'force_restore']:
                                try:
                                    load_params[param] = True
                                    session.load(**load_params)
                                    break
                                except TypeError:
                                    load_params.pop(param, None)
                            else:
                                session.load(**load_params)
                        else:
                            session.load(**load_params)

                        # Check if session loaded successfully and has results
                        if not hasattr(session, 'results') or session.results is None or session.results.empty:
                            self.stdout.write(self.style.WARNING(f"No qualifying results data available for Round {round_num}"))
                            continue

                        self.process_qualifying(session, event_data, session_type, event_format)
                        time.sleep(0.5)

                    except Exception as e:
                        self.stdout.write(self.style.ERROR(f"Error processing Round {round_num}: {str(e)}"))
                        if debug_mode:
                            import traceback
                            self.stdout.write(self.style.ERROR(traceback.format_exc()))
                        continue

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error processing season {year}: {e}"))
                if debug_mode:
                    import traceback
                    self.stdout.write(self.style.ERROR(traceback.format_exc()))

    def get_or_create_driver(self, row):
        """
        Safely get or create a driver, handling potential conflicts
        """
        driver_number = row['DriverNumber']
        driver_ref = row['Abbreviation'].strip()
        
        # Try to find existing driver by driver_id first
        try:
            driver = Driver.objects.get(driver_id=driver_number)
            # Update existing driver with latest info
            driver.driver_ref = driver_ref
            driver.given_name = row['FirstName']
            driver.family_name = row['LastName']
            driver.nationality = row.get('Country', '')
            driver.code = row['Abbreviation']
            driver.permanent_number = driver_number
            driver.save()
            return driver
        except Driver.DoesNotExist:
            pass
        
        # Try to find by driver_ref
        try:
            driver = Driver.objects.get(driver_ref=driver_ref)
            # Update existing driver with latest info
            driver.driver_id = driver_number
            driver.given_name = row['FirstName']
            driver.family_name = row['LastName']
            driver.nationality = row.get('Country', '')
            driver.code = row['Abbreviation']
            driver.permanent_number = driver_number
            driver.save()
            return driver
        except Driver.DoesNotExist:
            pass
        
        # Create new driver
        try:
            driver = Driver.objects.create(
                driver_id=driver_number,
                driver_ref=driver_ref,
                given_name=row['FirstName'],
                family_name=row['LastName'],
                nationality=row.get('Country', ''),
                code=row['Abbreviation'],
                permanent_number=driver_number,
            )
            return driver
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error creating driver {driver_ref}: {e}"))
            # As a last resort, try to find any existing driver with same name
            try:
                driver = Driver.objects.get(
                    given_name=row['FirstName'],
                    family_name=row['LastName']
                )
                return driver
            except Driver.DoesNotExist:
                raise e

    def process_qualifying(self, session, event_data, session_type, event_format='conventional'):
        """
        Process qualifying results for all race weekend formats (conventional, sprint, etc.)
        """
        circuit, _ = Circuit.objects.update_or_create(
            circuit_ref=event_data.get('CircuitKey', event_data['Location'].lower().replace(' ', '_')),
            defaults={
                'name': event_data['Location'],
                'location': event_data['Location'],
                'country': event_data['Country'],
            }
        )

        with transaction.atomic():
            # Determine event format for database
            if event_format in ['sprint_qualifying', 'sprint', 'sprint_shootout']:
                db_event_format = 'SPRINT'
            else:
                db_event_format = 'CONVENTIONAL'
            
            event_obj, created = Event.objects.update_or_create(
                year=event_data['EventDate'].year,
                round=event_data['RoundNumber'],
                defaults={
                    'name': event_data['EventName'],
                    'official_name': event_data.get('OfficialEventName', ''),
                    'date': parse_date(event_data['EventDate'].strftime('%Y-%m-%d')),
                    'circuit': circuit,
                    'event_format': db_event_format,
                }
            )

            # Handle timezone-aware datetime for session date
            session_date = session.date
            if session_date and timezone.is_naive(session_date):
                session_date = timezone.make_aware(session_date)

            session_obj, _ = Session.objects.update_or_create(
                event=event_obj,
                session_type=session_type,
                defaults={'date': session_date}
            )

            qualifying_results = session.results
            if qualifying_results is None or qualifying_results.empty:
                self.stdout.write(self.style.WARNING(f"No qualifying data for {event_obj}"))
                return

            processed_count = 0
            for _, row in qualifying_results.iterrows():
                try:
                    team, _ = Team.objects.get_or_create(
                        team_ref=row['TeamName'].lower().replace(' ', '_'),
                        defaults={'name': row['TeamName']}
                    )

                    driver = self.get_or_create_driver(row)

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
                            'pole_delta': None,
                            'status': row.get('Status', ''),
                            'laps': row.get('Laps', None),
                        }
                    )
                    processed_count += 1
                    
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Error processing qualifying result for driver {row.get('Abbreviation', 'Unknown')}: {e}"))
                    continue

            self.stdout.write(self.style.SUCCESS(f"Saved {processed_count} qualifying results for {event_obj.name} Round {event_obj.round}"))