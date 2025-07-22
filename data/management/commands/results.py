import os
import time
import pandas as pd
from datetime import timedelta
from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils.dateparse import parse_date
from django.utils import timezone
import fastf1
from data.models import Circuit, Team, Driver, Event, Session, SessionType, RaceResult, PitStop


class Command(BaseCommand):
    help = 'Fetch and store F1 race results, pit stops, and weather data using FastF1'

    def add_arguments(self, parser):
        parser.add_argument('--years', nargs='+', type=int, default=[2022, 2023, 2024],
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

        session_type, _ = SessionType.objects.get_or_create(
            session_type='RACE', defaults={'name': 'Race'}
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

                events = schedule
                if specific_rounds:
                    events = events[events['RoundNumber'].isin(specific_rounds)]
                    self.stdout.write(f"Filtered to rounds: {specific_rounds}")

                for i, (_, event_data) in enumerate(events.iterrows(), 1):
                    round_num = event_data['RoundNumber']
                    event_format = event_data.get('EventFormat', 'conventional')
                    
                    self.stdout.write(f"[{i}/{len(events)}] Round {round_num}: {event_data['EventName']}")

                    try:
                        session = fastf1.get_session(year, round_num, 'R')
                        load_params = {'telemetry': False, 'weather': True, 'messages': False}
                        if force_refresh:
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

                        if not hasattr(session, 'results') or session.results is None or session.results.empty:
                            self.stdout.write(self.style.WARNING(f"No race results data available for Round {round_num}"))
                            continue

                        self.process_race_results(session, event_data, session_type, event_format)
                        self.process_pit_stops(session, session_type)
                        self.process_weather_data(session, event_data)
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
        try:
            driver_number = row['DriverNumber']
            driver_ref = row['Abbreviation'].strip()
            try:
                driver = Driver.objects.get(driver_id=driver_number)
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
            try:
                driver = Driver.objects.get(driver_ref=driver_ref)
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
            try:
                driver = Driver.objects.get(
                    given_name=row['FirstName'],
                    family_name=row['LastName']
                )
                return driver
            except Driver.DoesNotExist:
                raise e

    def process_race_results(self, session, event_data, session_type, event_format='conventional'):
        circuit, _ = Circuit.objects.update_or_create(
            circuit_ref=event_data.get('CircuitKey', event_data['Location'].lower().replace(' ', '_')),
            defaults={
                'name': event_data['Location'],
                'location': event_data['Location'],
                'country': event_data['Country'],
            }
        )

        with transaction.atomic():
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

            session_date = session.date
            if session_date and timezone.is_naive(session_date):
                session_date = timezone.make_aware(session_date)

            session_obj, _ = Session.objects.update_or_create(
                event=event_obj,
                session_type=session_type,
                defaults={'date': session_date}
            )

            results = session.results
            if results is None or results.empty:
                self.stdout.write(self.style.WARNING(f"No race results for {event_obj}"))
                return

            processed_count = 0
            for _, row in results.iterrows():
                try:
                    team, _ = Team.objects.get_or_create(
                        team_ref=row['TeamName'].lower().replace(' ', '_'),
                        defaults={'name': row['TeamName']}
                    )

                    driver = self.get_or_create_driver(row)

                    def to_duration(val):
                        return val if isinstance(val, timedelta) else None

                    grid_pos = row.get('Grid')
                    finish_pos = row.get('Position')

                    position_gain = None
                    if grid_pos is not None and finish_pos is not None:
                        try:
                            position_gain = int(grid_pos) - int(finish_pos)
                        except (ValueError, TypeError):
                            position_gain = None

                    RaceResult.objects.update_or_create(
                        session=session_obj,
                        driver=driver,
                        defaults={
                            'team': team,
                            'position': int(row['Position']) if pd.notna(row['Position']) else None,
                            'position_text': str(row.get('Status', '')),
                            'points': float(row.get('Points', 0)) if pd.notna(row.get('Points')) else 0,
                            'status': row.get('Status', ''),
                            'laps': int(row.get('Laps')) if pd.notna(row.get('Laps')) else None,
                            'time': to_duration(row.get('Time')),
                            'time_millis': int(row.get('Time').total_seconds() * 1000) if pd.notna(row.get('Time')) and isinstance(row.get('Time'), timedelta) else None,
                            'fastest_lap_rank': int(row.get('FastestLapRank')) if pd.notna(row.get('FastestLapRank')) else None,
                            'fastest_lap_time': to_duration(row.get('FastestLapTime')),
                            'fastest_lap_speed': float(row.get('FastestLapSpeed')) if pd.notna(row.get('FastestLapSpeed')) else None,
                            'pit_stops': int(row.get('PitStops')) if pd.notna(row.get('PitStops')) else None,
                            'grid_position': int(grid_pos) if pd.notna(grid_pos) else None,
                            'position_gain': position_gain,
                            'tyre_stints': row.get('TyreStints', None)
                        }
                    )
                    processed_count += 1
                    
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Error processing result for driver {row.get('Abbreviation', 'Unknown')}: {e}"))
                    continue

            self.stdout.write(self.style.SUCCESS(f"Saved {processed_count} race results for {event_obj.name} Round {event_obj.round}"))

    def process_pit_stops(self, session, session_type):
        try:
            # Workaround for fastf1 v3.5.3: Filter laps where PitInTime or PitOutTime is not null
            laps = session.laps
            pit_stops = laps[laps['PitInTime'].notna() | laps['PitOutTime'].notna()].copy()
            if pit_stops.empty:
                self.stdout.write(self.style.WARNING(f"No pit stop data for {session.event['EventName']}"))
                return

            session_obj = Session.objects.get(
                event__year=session.event['EventDate'].year,
                event__round=session.event['RoundNumber'],
                session_type=session_type
            )

            processed_count = 0
            for _, pit in pit_stops.iterrows():
                try:
                    driver = Driver.objects.get(driver_ref=pit['Driver'].strip())
                    team = Team.objects.get(team_ref=pit['Team'].lower().replace(' ', '_'))
                    # Estimate pit stop duration using LapTime or fallback to 25 seconds
                    duration = pit.get('LapTime', None)
                    if pd.isna(duration) or not isinstance(duration, pd.Timedelta):
                        duration = timedelta(seconds=25)  # Default duration
                    else:
                        duration = timedelta(seconds=duration.total_seconds())
                    time_ms = int(duration.total_seconds() * 1000) if isinstance(duration, timedelta) else 25000

                    PitStop.objects.update_or_create(
                        session=session_obj,
                        driver=driver,
                        lap=int(pit['LapNumber']) if pd.notna(pit['LapNumber']) else 1,
                        defaults={
                            'team': team,
                            'time_millis': time_ms,
                            'duration': duration,
                            'tyre_compound': pit.get('Compound', None)
                        }
                    )
                    processed_count += 1

                except (Driver.DoesNotExist, Team.DoesNotExist) as e:
                    self.stdout.write(self.style.WARNING(f"Skipping pit stop for {pit.get('Driver', 'Unknown')}: {e}"))
                    continue
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Error processing pit stop for {pit.get('Driver', 'Unknown')}: {e}"))
                    continue

            self.stdout.write(self.style.SUCCESS(f"Saved {processed_count} pit stops for {session.event['EventName']}"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error processing pit stops for {session.event['EventName']}: {e}"))

    def process_weather_data(self, session, event_data):
        try:
            weather = session.weather_data
            if weather is None or weather.empty:
                self.stdout.write(self.style.WARNING(f"No weather data for {event_data['EventName']}"))
                return

            event_obj = Event.objects.get(
                year=event_data['EventDate'].year,
                round=event_data['RoundNumber']
            )

            # Aggregate weather data by taking the mean for numeric fields
            weather_data = {
                'rain': bool(weather['Rainfall'].mean() > 0.1) if 'Rainfall' in weather.columns else False,
                'safety_car': False,  # Placeholder, as FastF1 doesn't provide this
                'air_temp': float(weather['AirTemp'].mean()) if 'AirTemp' in weather.columns and pd.notna(weather['AirTemp']).any() else None,
                'track_temp': float(weather['TrackTemp'].mean()) if 'TrackTemp' in weather.columns and pd.notna(weather['TrackTemp']).any() else None,
                'humidity': float(weather['Humidity'].mean()) if 'Humidity' in weather.columns and pd.notna(weather['Humidity']).any() else None,
                'wind_speed': float(weather['WindSpeed'].mean()) if 'WindSpeed' in weather.columns and pd.notna(weather['WindSpeed']).any() else None,
                'wind_direction': float(weather['WindDirection'].mean()) if 'WindDirection' in weather.columns and pd.notna(weather['WindDirection']).any() else None
            }

            event_obj.weather_data = weather_data
            event_obj.save()
            self.stdout.write(self.style.SUCCESS(f"Saved weather data for {event_data['EventName']}"))

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error processing weather data for {event_data['EventName']}: {e}"))