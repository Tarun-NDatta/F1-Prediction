import numpy as np
from django.core.management.base import BaseCommand
from django.db.models import Avg, Q, F
from django.db import transaction
from data.models import (
    Driver, Team, Event, Session, SessionType,
    QualifyingResult, RaceResult, PitStop,
    DriverPerformance, TeamPerformance, TrackCharacteristics
)
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Extracts and stores engineered features for prediction models"
    
    # Constants
    RETIREMENT_STATUSES = ['DNF', 'DNS', 'DQ', 'Retired', 'Accident', 'Mechanical']
    DEFAULT_MIDFIELD_POS = 15
    DEFAULT_VARIANCE = 5
    DEFAULT_QUALI_AVG = 10
    DEFAULT_PIT_TIME = 25000  # Default pit time in milliseconds (25 seconds)
    TEMPORAL_DECAY_FACTOR = 0.85  # Weight multiplier per race (15% decay)

    def add_arguments(self, parser):
        parser.add_argument('--date', type=str, default='2025-12-31', help='Process events up to this date (YYYY-MM-DD)')
        parser.add_argument('--bulk-size', type=int, default=100, help='Batch size for bulk operations')
        parser.add_argument('--skip-errors', action='store_true', help='Continue processing even if some events fail')

    def handle(self, *args, **options):
        cutoff_date = options['date']
        bulk_size = options['bulk_size']
        skip_errors = options['skip_errors']

        events = Event.objects.filter(date__lte=cutoff_date).order_by('date')
        self.stdout.write(self.style.SUCCESS(f"Processing {events.count()} events up to {cutoff_date}"))

        for event in events:
            try:
                with transaction.atomic():
                    self.process_event(event, bulk_size)
                self.stdout.write(self.style.SUCCESS(f"Processed {event.year} {event.name}"))
            except Exception as e:
                error_msg = f"Error processing {event}: {str(e)}"
                if skip_errors:
                    self.stderr.write(self.style.WARNING(error_msg))
                    logger.warning(error_msg)
                else:
                    self.stderr.write(self.style.ERROR(error_msg))
                    logger.error(error_msg, exc_info=True)
                    raise

    def _bulk_update_or_create(self, model, items, match_fields, update_fields):
        """Custom implementation of bulk_update_or_create"""
        with transaction.atomic():
            existing_lookup = Q()
            for item in items:
                item_filter = Q()
                for field in match_fields:
                    item_filter &= Q(**{field: getattr(item, field)})
                existing_lookup |= item_filter
            
            existing_items = {
                tuple(getattr(item, field) for field in match_fields): item
                for item in model.objects.filter(existing_lookup)
            }
            
            to_update = []
            to_create = []
            
            for item in items:
                match_key = tuple(getattr(item, field) for field in match_fields)
                if match_key in existing_items:
                    existing = existing_items[match_key]
                    for field in update_fields:
                        setattr(existing, field, getattr(item, field))
                    to_update.append(existing)
                else:
                    to_create.append(item)
            
            if to_update:
                model.objects.bulk_update(to_update, update_fields)
            if to_create:
                model.objects.bulk_create(to_create)

    def process_event(self, event, bulk_size):
        self.stdout.write(f"\n\U0001F50D Processing {event.year} {event.name}...")

        race_session = Session.objects.filter(
            event=event,
            session_type__session_type='RACE'
        ).first()
        
        if not race_session:
            self.stdout.write(f"  \U0001F6AB No race session found for {event}")
            return

        self.generate_driver_features(event, bulk_size)
        self.generate_team_features(event, bulk_size)
        self.generate_track_characteristics(event.circuit)

    def generate_driver_features(self, event, bulk_size):
        drivers = Driver.objects.all()
        performances = []

        for driver in drivers:
            try:
                recent_results = list(RaceResult.objects.filter(
                    driver=driver,
                    session__event__date__lt=event.date
                ).order_by('-session__event__date')[:5])

                circuit_results = list(RaceResult.objects.filter(
                    session__event__circuit=event.circuit,
                    driver=driver,
                    session__event__date__lt=event.date
                ).order_by('-session__event__date'))

                recent_qualifying = list(QualifyingResult.objects.filter(
                    driver=driver,
                    session__event__date__lt=event.date
                ).order_by('-session__event__date')[:5])

                # Pit stop average for driver
                pit_stops = PitStop.objects.filter(
                    driver=driver,
                    session__event__date__lt=event.date
                ).order_by('-session__event__date')[:10]
                if not pit_stops:
                    logger.warning(f"No pit stops found for driver {driver} for events before {event}")
                pit_stop_avg = np.mean([p.time_millis for p in pit_stops]) if pit_stops else self.DEFAULT_PIT_TIME

                # DNF rate for driver
                dnf_rate = self._calc_dnf_rate(recent_results)

                features = {
                    'moving_avg_5': self._calc_weighted_avg(recent_results),
                    'position_variance': self._calc_position_variance(recent_results),
                    'qualifying_avg': self._calc_weighted_quali_avg(recent_qualifying),
                    'points_per_race': self._calc_weighted_avg(recent_results, 'points', 0),
                    'circuit_affinity': self._calc_weighted_avg(circuit_results),
                    'quali_improvement': self._calc_quali_improvement(recent_qualifying),
                    'teammate_battle': self._calc_teammate_comparison(driver, event),
                    'wet_weather_perf': self._calc_wet_weather_perf(driver, event.date),
                    'rivalry_performance': self._calc_rivalry_performance(driver, event.date),
                    'quali_race_delta': self._calc_quali_race_delta(driver, event.date),
                    'position_momentum': self._calc_position_momentum(driver, event.date),
                    'pit_stop_avg': pit_stop_avg,
                    'dnf_rate': dnf_rate
                }

                performances.append(DriverPerformance(
                    driver=driver,
                    event=event,
                    **features
                ))

                if len(performances) >= bulk_size:
                    self._bulk_update_or_create(
                        DriverPerformance,
                        performances,
                        ['driver', 'event'],
                        list(features.keys())
                    )
                    performances = []
                    
            except Exception as e:
                error_msg = f"Error processing driver {driver} for event {event}: {str(e)}"
                self.stderr.write(self.style.WARNING(error_msg))
                logger.warning(error_msg, exc_info=True)
                continue

        if performances:
            self._bulk_update_or_create(
                DriverPerformance,
                performances,
                ['driver', 'event'],
                list(features.keys())
            )

    def generate_team_features(self, event, bulk_size):
        teams = Team.objects.all()
        team_performances = []

        for team in teams:
            try:
                results = RaceResult.objects.filter(
                    team=team,
                    session__event__date__lt=event.date
                )

                last_10_races = list(results.order_by('-session__event__date')[:10])
                positions = [int(r.position) for r in last_10_races if r.position is not None]

                development_slope = 0
                if len(positions) > 1:
                    try:
                        development_slope = np.polyfit(range(len(positions)), positions, 1)[0]
                    except:
                        development_slope = 0

                # Team pit stop average
                pit_stops = PitStop.objects.filter(
                    team=team,
                    session__event__date__lt=event.date
                ).order_by('-session__event__date')[:10]
                if not pit_stops:
                    logger.warning(f"No pit stops found for team {team} for events before {event}")
                pit_stop_avg = np.mean([p.time_millis for p in pit_stops]) if pit_stops else self.DEFAULT_PIT_TIME

                features = {
                    'dnf_rate': self._calc_dnf_rate(results),
                    'pit_stop_avg': pit_stop_avg,
                    'reliability_score': 1 - self._calc_dnf_rate(results),
                    'development_slope': development_slope,
                    'qualifying_consistency': self._calc_team_quali_consistency(team, event.date),
                    'pit_stop_std': self._calc_pit_stop_variance(team, event.date),
                    'moving_avg_5': self._calc_weighted_avg(last_10_races),
                    'position_variance': self._calc_position_variance(last_10_races),
                    'qualifying_avg': self._calc_team_quali_avg(team, event.date)
                }

                team_performances.append(TeamPerformance(
                    team=team,
                    event=event,
                    **features
                ))

                if len(team_performances) >= bulk_size:
                    self._bulk_update_or_create(
                        TeamPerformance,
                        team_performances,
                        ['team', 'event'],
                        list(features.keys())
                    )
                    team_performances = []
                    
            except Exception as e:
                error_msg = f"Error processing team {team} for event {event}: {str(e)}"
                self.stderr.write(self.style.WARNING(error_msg))
                logger.warning(error_msg, exc_info=True)
                continue

        if team_performances:
            self._bulk_update_or_create(
                TeamPerformance,
                team_performances,
                ['team', 'event'],
                list(features.keys())
            )

    def generate_track_characteristics(self, circuit):
        try:
            if TrackCharacteristics.objects.filter(circuit=circuit).exists():
                return
            
            races = Event.objects.filter(circuit=circuit)
            results = RaceResult.objects.filter(session__event__circuit=circuit)
            
            if not races.exists():
                return
            
            overtaking = self._calc_overtaking_index(results)
            safety_car_prob = self._calc_safety_car_probability(races)
            
            TrackCharacteristics.objects.update_or_create(
                circuit=circuit,
                defaults={
                    'overtaking_index': overtaking,
                    'safety_car_probability': safety_car_prob,
                    'rain_impact': self._calc_rain_impact(circuit),
                    'avg_pit_loss': self._calc_avg_pit_loss(circuit)
                }
            )
        except Exception as e:
            error_msg = f"Error generating track characteristics for {circuit}: {str(e)}"
            self.stderr.write(self.style.WARNING(error_msg))
            logger.warning(error_msg, exc_info=True)

    def _calc_weighted_avg(self, results, value_field='position', default=15):
        if not results:
            return default
            
        values = []
        weights = []
        current_weight = 1.0
        
        for result in sorted(results, key=lambda r: r.session.event.date, reverse=True):
            value = getattr(result, value_field)
            if value is not None:
                values.append(value)
                weights.append(current_weight)
                current_weight *= self.TEMPORAL_DECAY_FACTOR
        
        if not values:
            return default
            
        return np.average(values, weights=weights)

    def _calc_weighted_quali_avg(self, qualifying_results):
        if not qualifying_results:
            return self.DEFAULT_QUALI_AVG
            
        positions = []
        weights = []
        current_weight = 1.0
        
        for quali in sorted(qualifying_results, 
                           key=lambda q: q.session.event.date, 
                           reverse=True):
            if quali.position:
                positions.append(quali.position)
                weights.append(current_weight)
                current_weight *= self.TEMPORAL_DECAY_FACTOR
        
        if not positions:
            return self.DEFAULT_QUALI_AVG
            
        return np.average(positions, weights=weights)

    def _calc_team_quali_avg(self, team, cutoff_date):
        try:
            positions = list(QualifyingResult.objects.filter(
                team=team,
                session__event__date__lt=cutoff_date,
                position__isnull=False
            ).order_by('-session__event__date')[:10].values_list('position', flat=True))
            
            if not positions:
                return self.DEFAULT_QUALI_AVG
            
            weights = [self.TEMPORAL_DECAY_FACTOR ** i for i in range(len(positions))]
            return np.average(positions, weights=weights)
        except:
            return self.DEFAULT_QUALI_AVG

    def _calc_position_variance(self, results):
        if len(results) < 2:
            return self.DEFAULT_VARIANCE
        positions = [r.position for r in results if r.position is not None]
        return np.std(positions) if positions else self.DEFAULT_VARIANCE

    def _calc_quali_improvement(self, qualifying_results):
        if len(qualifying_results) < 2:
            return 0
        
        recent = qualifying_results[0].position
        oldest = qualifying_results[-1].position
        
        if recent is None or oldest is None:
            return 0
            
        return oldest - recent

    def _calc_teammate_comparison(self, driver, event):
        try:
            driver_result = RaceResult.objects.filter(
                session__event=event,
                driver=driver
            ).first()

            if not driver_result:
                return 0.5
            
            teammate = RaceResult.objects.filter(
                session__event=event,
                team=driver_result.team
            ).exclude(driver=driver).first()

            if not teammate:
                return 0.5

            head_to_head = list(RaceResult.objects.filter(
                session__event__date__lt=event.date
            ).filter(
                Q(driver=driver) | Q(driver=teammate.driver)
            ).order_by('session__event__date'))

            events_ordered = sorted(
                set(r.session.event for r in head_to_head),
                key=lambda e: e.date
            )

            driver_results = []
            teammate_results = []

            for e in events_ordered:
                d_result = next((r for r in head_to_head if r.session.event == e and r.driver == driver), None)
                t_result = next((r for r in head_to_head if r.session.event == e and r.driver == teammate.driver), None)
                
                if d_result and t_result and d_result.position and t_result.position:
                    driver_results.append(d_result.position)
                    teammate_results.append(t_result.position)

            if not driver_results or not teammate_results:
                return 0.5

            wins = sum(1 for d, t in zip(driver_results, teammate_results) if d < t)
            return wins / len(driver_results)
            
        except Exception as e:
            error_msg = f"Error calculating teammate comparison for {driver} at {event}: {str(e)}"
            self.stderr.write(self.style.WARNING(error_msg))
            logger.warning(error_msg, exc_info=True)
            return 0.5

    def _calc_wet_weather_perf(self, driver, cutoff_date):
        try:
            wet_races = list(RaceResult.objects.filter(
                session__event__weather_data__rain=True,
                driver=driver,
                session__event__date__lt=cutoff_date
            ))
            
            if not wet_races:
                return self.DEFAULT_MIDFIELD_POS
            
            positions = [r.position for r in wet_races if r.position is not None]
            weights = [self.TEMPORAL_DECAY_FACTOR ** i for i in range(len(positions))]
            return np.average(positions, weights=weights) if positions else self.DEFAULT_MIDFIELD_POS
            
        except Exception as e:
            error_msg = f"Error calculating wet weather performance for {driver}: {str(e)}"
            self.stderr.write(self.style.WARNING(error_msg))
            logger.warning(error_msg, exc_info=True)
            return self.DEFAULT_MIDFIELD_POS

    # ... (previous code unchanged until _calc_rivalry_performance)

    def _calc_rivalry_performance(self, driver, cutoff_date, rival_range=3):
        try:
            recent_races = RaceResult.objects.filter(
                driver=driver,
                session__event__date__lt=cutoff_date
            ).order_by('-session__event__date')[:10]
            
            rival_ids = set()
            position_diffs = []
            
            for race in recent_races:
                if race.position is None:
                    continue
                rivals = RaceResult.objects.filter(
                    session=race.session,
                    position__range=(race.position - rival_range, race.position + rival_range)
                ).exclude(driver=driver)
                
                for rival in rivals:
                    if rival.position is not None:
                        rival_ids.add(rival.driver.id)
                        position_diffs.append(race.position - rival.position)
            
            if not rival_ids:
                logger.warning(f"No valid rivals found for driver {driver} before {cutoff_date}")
                return 0.5
                
            head_to_head = RaceResult.objects.filter(
                driver=driver,
                session__event__date__lt=cutoff_date
            ).filter(
                session__raceresult__driver__in=list(rival_ids)
            ).annotate(
                rival_position=F('session__raceresult__position')
            ).filter(
                rival_position__isnull=False,
                position__isnull=False
            )
            
            wins = sum(1 for r in head_to_head if r.position is not None and r.rival_position is not None and r.position < r.rival_position)
            return wins / len(head_to_head) if head_to_head else 0.5
            
        except Exception as e:
            logger.warning(f"Rivalry calculation error for driver {driver}: {str(e)}")
            return 0.5

# ... (rest of the code unchanged)
    def _calc_quali_race_delta(self, driver, cutoff_date):
        try:
            deltas = []
            races = RaceResult.objects.filter(
                driver=driver,
                session__event__date__lt=cutoff_date,
                grid_position__isnull=False
            ).order_by('-session__event__date')[:5]
            
            for race in races:
                if race.position and race.grid_position:
                    deltas.append(race.grid_position - race.position)
                    
            return np.mean(deltas) if deltas else 0
        except:
            return 0

    def _calc_position_momentum(self, driver, cutoff_date):
        try:
            positions = []
            races = RaceResult.objects.filter(
                driver=driver,
                session__event__date__lt=cutoff_date,
                position__isnull=False
            ).order_by('-session__event__date')[:3]
            
            for race in races:
                positions.append(race.position)
                
            if len(positions) < 2:
                return 0
                
            x = np.arange(len(positions))
            slope = np.polyfit(x, positions, 1)[0]
            return -slope
        except:
            return 0

    def _calc_dnf_rate(self, results):
        try:
            total = len(results)  # Use len() for lists
            if total == 0:
                return 0.1
            dnfs = sum(1 for r in results if r.status in self.RETIREMENT_STATUSES)
            return dnfs / total
        except Exception as e:
            error_msg = f"Error calculating DNF rate: {str(e)}"
            self.stderr.write(self.style.WARNING(error_msg))
            logger.warning(error_msg, exc_info=True)
            return 0.1

    def _calc_team_quali_consistency(self, team, cutoff_date):
        try:
            positions = list(QualifyingResult.objects.filter(
                team=team,
                session__event__date__lt=cutoff_date,
                position__isnull=False
            ).order_by('-session__event__date')[:10].values_list('position', flat=True))
            
            if len(positions) < 2:
                return 5
            
            return np.std(positions)
            
        except Exception as e:
            error_msg = f"Error calculating quali consistency for {team}: {str(e)}"
            self.stderr.write(self.style.WARNING(error_msg))
            logger.warning(error_msg, exc_info=True)
            return 5

    def _calc_pit_stop_variance(self, team, cutoff_date):
        try:
            pit_times = PitStop.objects.filter(
                team=team,
                session__event__date__lt=cutoff_date
            ).values_list('time_millis', flat=True)
            
            if len(pit_times) < 3:
                return self.DEFAULT_PIT_TIME / 4
                
            return np.std(pit_times)
        except:
            return self.DEFAULT_PIT_TIME / 4

    def _calc_overtaking_index(self, results):
        try:
            position_changes = []
            events = set(r.session.event for r in results)
            
            for event in events:
                race_results = list(results.filter(session__event=event).order_by('position'))
                if not race_results:
                    continue
                    
                race_results = [r for r in race_results if r.grid_position is not None]
                if not race_results:
                    continue
                    
                grid_positions = sorted(race_results, key=lambda x: x.grid_position)
                
                changes = []
                for result in race_results:
                    if result.position and result.grid_position:
                        changes.append(abs(result.position - result.grid_position))
                
                if changes:
                    position_changes.append(np.mean(changes))
                
            return np.mean(position_changes) if position_changes else 3.0
            
        except Exception as e:
            error_msg = f"Error calculating overtaking index: {str(e)}"
            self.stderr.write(self.style.WARNING(error_msg))
            logger.warning(error_msg, exc_info=True)
            return 3.0

    def _calc_safety_car_probability(self, events):
        try:
            total = events.count()
            if total == 0:
                return 0.3
                
            sc_events = events.filter(weather_data__safety_car=True).count()
            return sc_events / total
            
        except Exception as e:
            error_msg = f"Error calculating safety car probability: {str(e)}"
            self.stderr.write(self.style.WARNING(error_msg))
            logger.warning(error_msg, exc_info=True)
            return 0.3

    def _calc_rain_impact(self, circuit):
        try:
            wet_races = list(Event.objects.filter(circuit=circuit, weather_data__rain=True))
            if not wet_races:
                return 0.5
            
            avg_finish_diff = 0
            valid_races = 0
            
            for event in wet_races:
                dry_results = list(RaceResult.objects.filter(
                    session__event__circuit=circuit,
                    session__event__weather_data__rain=False,
                    session__event__date__lt=event.date
                ))
                
                dry_positions = [r.position for r in dry_results if r.position]
                if not dry_positions:
                    continue
                    
                dry_avg = np.mean(dry_positions)
                
                wet_results = list(RaceResult.objects.filter(session__event=event))
                wet_positions = [r.position for r in wet_results if r.position]
                if not wet_positions:
                    continue
                    
                wet_avg = np.mean(wet_positions)
                
                avg_finish_diff += (wet_avg - dry_avg)
                valid_races += 1
            
            if valid_races == 0:
                return 0.5
                
            return avg_finish_diff / valid_races
            
        except Exception as e:
            error_msg = f"Error calculating rain impact for {circuit}: {str(e)}"
            self.stderr.write(self.style.WARNING(error_msg))
            logger.warning(error_msg, exc_info=True)
            return 0.5

    def _calc_avg_pit_loss(self, circuit):
        try:
            pit_times = PitStop.objects.filter(
                session__event__circuit=circuit
            ).values_list('time_millis', flat=True)
            
            return np.mean(pit_times) / 1000 if pit_times else self.DEFAULT_PIT_TIME / 1000
            
        except Exception as e:
            error_msg = f"Error calculating pit loss for {circuit}: {str(e)}"
            self.stderr.write(self.style.WARNING(error_msg))
            logger.warning(error_msg, exc_info=True)
            return self.DEFAULT_PIT_TIME / 1000