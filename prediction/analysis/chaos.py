import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from django.db.models import Q

from data.models import (
    Event,
    Session,
    SessionType,
    RaceResult,
    CatBoostPrediction,
    TrackCharacteristics,
)


ACCIDENT_KEYWORDS = [
    'accident', 'collision', 'crash', 'spin', 'damage'
]
DNF_KEYWORDS = [
    'dnf', 'retired', 'engine', 'gearbox', 'hydraul', 'electrical', 'mechanical', 'tyre', 'puncture', 'fuel'
]
PENALTY_KEYWORDS = [
    'penalty', 'penalised', 'drive-through', 'stop-go', 'dsq', 'disqualified', 'black flag'
]


@dataclass
class RaceChaosMetrics:
    dnfs: int
    crashes: int
    penalties: int
    total_classified: int
    weather_flag: bool
    position_volatility: float  # mean absolute position change
    safety_car_proxy: Optional[float]  # use track safety car probability if available
    early_lap_incidents: int  # lap 1-3 incidents proxy
    median_pit_stops: float


class RaceEventClassifier:
    """Compute a 0-10 chaos score for an Event based on race metrics.

    Notes:
    - We apply an explicit rain bump (+3.0) if rain detected to emphasize weather chaos.
    - We will categorize by quantiles across events (10% clean, 30% chaotic) in higher-level analysis.
    """

    def __init__(self,
                 dnf_weight: float = 0.5,
                 crash_weight: float = 0.35,
                 penalty_weight: float = 0.1,
                 weather_weight: float = 0.2,
                 volatility_weight: float = 0.1,
                 rain_bump: float = 3.0):
        self.weights = dict(
            dnf=dnf_weight,
            crash=crash_weight,
            penalty=penalty_weight,
            weather=weather_weight,
            volatility=volatility_weight,
        )
        self.rain_bump = rain_bump

    def _get_race_session(self, event: Event) -> Optional[Session]:
        return (
            Session.objects.filter(event=event, session_type__session_type='RACE').first()
        )

    def compute_metrics(self, event: Event) -> Optional[RaceChaosMetrics]:
        session = self._get_race_session(event)
        if not session:
            return None

        results = list(RaceResult.objects.filter(session=session))
        if not results:
            return None

        total = len([r for r in results if r.position is not None or (r.status and r.status.strip())])

        status_texts = [(r.status or '').lower() for r in results]
        def contains_any(s: str, keys: List[str]) -> bool:
            return any(k in s for k in keys)

        dnfs = sum(1 for s in status_texts if contains_any(s, DNF_KEYWORDS))
        crashes = sum(1 for s in status_texts if contains_any(s, ACCIDENT_KEYWORDS))
        penalties = sum(1 for s in status_texts if contains_any(s, PENALTY_KEYWORDS))

        # Early lap incidents proxy: any accident/collision within first 3 laps if lap info exists
        early_lap_incidents = 0
        try:
            early_lap_incidents = sum(1 for r in results if r.lap and r.lap <= 3 and contains_any((r.status or '').lower(), ACCIDENT_KEYWORDS))
        except Exception:
            early_lap_incidents = 0

        # Median pit stops across finishers
        try:
            pit_list = [r.pit_stops for r in results if (r.pit_stops is not None and (r.position is not None))]
            median_pit_stops = float(np.median(pit_list)) if pit_list else 0.0
        except Exception:
            median_pit_stops = 0.0

        # Weather: use race session rain flag or event.weather_impact
        weather_flag = bool(getattr(session, 'rain', False)) or bool(getattr(event, 'weather_impact', None) and (event.weather_impact or 0) > 0.2)

        # Position volatility: use position_gain if present; else compute from grid
        swings: List[int] = []
        for r in results:
            if r.position_gain is not None:
                swings.append(abs(r.position_gain))
            else:
                if r.position is not None and r.grid_position is not None:
                    swings.append(abs((r.grid_position or 0) - (r.position or 0)))
        position_volatility = float(np.mean(swings)) if swings else 0.0

        safety_car_proxy = None
        try:
            tc = TrackCharacteristics.objects.filter(circuit=event.circuit).first()
            safety_car_proxy = tc.safety_car_probability if tc else None
        except Exception:
            safety_car_proxy = None

        return RaceChaosMetrics(
            dnfs=dnfs,
            crashes=crashes,
            penalties=penalties,
            total_classified=total,
            weather_flag=weather_flag,
            position_volatility=position_volatility,
            safety_car_proxy=safety_car_proxy,
            early_lap_incidents=early_lap_incidents,
            median_pit_stops=median_pit_stops,
        )

    def chaos_score_and_category(self, event: Event) -> Tuple[float, str, Optional[RaceChaosMetrics]]:
        metrics = self.compute_metrics(event)
        if not metrics or not metrics.total_classified:
            return 0.0, 'unknown', metrics

        dnfs_ratio = metrics.dnfs / max(metrics.total_classified, 1)
        crash_ratio = metrics.crashes / max(metrics.total_classified, 1)
        penalty_ratio = metrics.penalties / max(metrics.total_classified, 1)
        weather_component = 1.0 if metrics.weather_flag else 0.0

        # Normalize position volatility approximately to 0..1 by dividing by 10 (large swings rare)
        volatility_component = min(metrics.position_volatility / 6.0, 1.0)

        raw = (
            self.weights['dnf'] * dnfs_ratio +
            self.weights['crash'] * crash_ratio +
            self.weights['penalty'] * penalty_ratio +
            self.weights['weather'] * weather_component +
            self.weights['volatility'] * volatility_component
        )
        # Safety car proxy bump
        if metrics.safety_car_proxy and metrics.safety_car_proxy > 0.5:
            raw += 0.1
        # Rain bump
        if weather_component >= 1.0:
            raw += self.rain_bump / 10.0
        # Early lap incidents bump
        if metrics.early_lap_incidents and metrics.early_lap_incidents >= 2:
            raw += 0.1
        # High median pit stops suggests strategy/VSC/SC heavy
        if metrics.median_pit_stops and metrics.median_pit_stops >= 3:
            raw += 0.1

        score = float(max(0.0, min(10.0, 10.0 * raw)))
        if score <= 3.0:
            category = 'clean'
        elif score <= 6.0:
            category = 'moderate'
        else:
            category = 'chaotic'
        return score, category, metrics


def was_driver_affected_by_chaos(result: RaceResult, metrics: Optional[RaceChaosMetrics]) -> bool:
    """Heuristic to decide if driver was impacted by an unpredictable event.
    Aggressive criteria per dissertation spec.
    """
    status = (result.status or '').lower()
    # Any DNF/penalty/crash keywords
    if any(k in status for k in (DNF_KEYWORDS + ACCIDENT_KEYWORDS + PENALTY_KEYWORDS)):
        return True

    # Position swing threshold lowered to 5
    if result.position_gain is not None:
        if abs(result.position_gain) >= 5:
            return True
    else:
        if result.position is not None and result.grid_position is not None:
            if abs(result.grid_position - result.position) >= 5:
                return True

    # Many pit-stops in volatile race
    if metrics and metrics.position_volatility >= 5 and (result.pit_stops or 0) >= 3:
        return True

    # Grid penalties >3 places (detect via grid_position vs qualifying position proxy)
    # If qualifying result exists with session match, infer penalty via grid-start vs quali pos
    try:
        from data.models import QualifyingResult
        q = QualifyingResult.objects.filter(session__event=result.session.event, driver=result.driver).first()
        if q and q.position and result.grid_position:
            if (result.grid_position - q.position) >= 4:
                return True
    except Exception:
        pass

    return False


def analyze_prediction_errors_by_events(season: Optional[int] = 2025,
                                        model_name: str = 'catboost_ensemble') -> pd.DataFrame:
    """Build a row per driver-event linking predictions, results, and chaos annotations.

    If season is None, analyze all years available.
    """
    qs = CatBoostPrediction.objects.filter(model_name=model_name)
    if season is not None:
        qs = qs.filter(year=season)
    preds = qs.select_related('driver', 'event')
    if not preds.exists():
        return pd.DataFrame()

    classifier = RaceEventClassifier()

    rows: List[Dict] = []
    # Pre-fetch race sessions to minimize queries
    event_ids = sorted(set(p.event_id for p in preds))
    sessions = {
        e_id: Session.objects.filter(event_id=e_id, session_type__session_type='RACE').first()
        for e_id in event_ids
    }

    # Cache chaos per event
    chaos_cache: Dict[int, Tuple[float, str, Optional[RaceChaosMetrics]]] = {}
    # Precompute chaos for all events to allow quantile-based categories
    for e_id in event_ids:
        event = Event.objects.get(id=e_id)
        chaos_cache[e_id] = classifier.chaos_score_and_category(event)

    # Quantile-based categories: bottom 10% = clean, top 30% = chaotic, else moderate
    chaos_scores = np.array([chaos_cache[e_id][0] for e_id in event_ids], dtype=float)
    if chaos_scores.size:
        q10 = float(np.quantile(chaos_scores, 0.10))
        q70 = float(np.quantile(chaos_scores, 0.70))
        def recategorize(score: float) -> str:
            if score <= q10:
                return 'clean'
            if score >= q70:
                return 'chaotic'
            return 'moderate'
    else:
        q10 = q70 = None
        def recategorize(score: float) -> str:
            return 'moderate'

    for p in preds:
        event = p.event
        chaos_score, _category, metrics = chaos_cache[event.id]
        category = recategorize(chaos_score)

        # Find this driver's race result
        session = sessions.get(event.id)
        rr = None
        if session:
            rr = RaceResult.objects.filter(session=session, driver=p.driver).first()

        actual_pos = p.actual_position
        if actual_pos is None and rr:
            actual_pos = rr.position

        if actual_pos is None:
            # Skip if no actual
            continue

        err = abs((p.predicted_position or 0) - float(actual_pos))
        affected = False
        if rr:
            affected = was_driver_affected_by_chaos(rr, metrics)

        rows.append({
            'event_id': event.id,
            'event': f"{event.year} R{event.round} {event.name}",
            'driver_id': p.driver.id,
            'driver': str(p.driver),
            'predicted_pos': float(p.predicted_position),
            'actual_pos': int(actual_pos),
            'error': float(err),
            'chaos_score': float(chaos_score),
            'race_category': category,
            'driver_affected': bool(affected),
        })

    df = pd.DataFrame(rows)
    return df


class CounterfactualAnalyzer:
    """Counterfactual analyses for predictability ceiling under chaos.

    - analyze_clean_race_accuracy: MAE on unaffected drivers in clean races
    - simulate_perfect_chaos_prediction: MAE if chaos-affected drivers were perfectly predicted (error=0)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def analyze_clean_race_accuracy(self) -> float:
        mask = (self.df['race_category'] == 'clean') & (~self.df['driver_affected'])
        subset = self.df[mask]
        if subset.empty:
            return float('nan')
        return float(np.mean(np.abs(subset['error'])))

    def simulate_perfect_chaos_prediction(self) -> float:
        df_cf = self.df.copy()
        df_cf.loc[df_cf['driver_affected'] == True, 'error'] = 0.0
        return float(np.mean(np.abs(df_cf['error'])))

    def group_mae(self) -> pd.DataFrame:
        g = self.df.groupby(['race_category', 'driver_affected'])['error'].mean().reset_index().rename(columns={'error': 'mae'})
        return g

    def group_mae_by_pos_group(self) -> pd.DataFrame:
        def pos_group(pos):
            if pos <= 6:
                return 'front'
            if pos <= 15:
                return 'midfield'
            return 'back'
        df = self.df.copy()
        df['pos_group'] = df['actual_pos'].apply(pos_group)
        g = df.groupby(['pos_group','race_category','driver_affected'])['error'].mean().reset_index().rename(columns={'error':'mae'})
        return g

    def bootstrap_ci(self, mask: pd.Series, n_boot: int = 2000, ci: float = 0.95, random_state: int = 42) -> Tuple[float, float, float]:
        rng = np.random.default_rng(random_state)
        x = self.df.loc[mask, 'error'].to_numpy()
        if x.size == 0:
            return float('nan'), float('nan'), float('nan')
        boot = []
        n = x.size
        for _ in range(n_boot):
            sample = x[rng.integers(0, n, n)]
            boot.append(float(np.mean(sample)))
        boot = np.array(boot)
        lower = float(np.percentile(boot, (1-ci)/2*100))
        upper = float(np.percentile(boot, (1+ci)/2*100))
        return float(np.mean(x)), lower, upper

    def significance_test(self, mask_a: pd.Series, mask_b: pd.Series) -> Dict:
        xa = self.df.loc[mask_a, 'error'].to_numpy()
        xb = self.df.loc[mask_b, 'error'].to_numpy()
        if xa.size == 0 or xb.size == 0:
            return {'method': 'NA', 'p_value': float('nan')}
        try:
            from scipy.stats import mannwhitneyu
            stat, p = mannwhitneyu(xa, xb, alternative='two-sided')
            return {'method': 'mannwhitneyu', 'stat': float(stat), 'p_value': float(p)}
        except Exception:
            # Fallback: bootstrap difference in means
            rng = np.random.default_rng(42)
            n_boot = 2000
            diffs = []
            for _ in range(n_boot):
                sa = xa[rng.integers(0, xa.size, xa.size)]
                sb = xb[rng.integers(0, xb.size, xb.size)]
                diffs.append(float(np.mean(sa) - np.mean(sb)))
            diffs = np.array(diffs)
            p = float(2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0)))
            return {'method': 'bootstrap_diff_means', 'p_value': p}



def analyze_event(event_id: int, model_name: str = 'catboost_ensemble') -> Dict:
    """Analyze a single race event with driver-level reasons for errors.

    Returns a dict with keys: event, chaos_score, race_category, metrics, drivers (list).
    """
    from data.models import Event, Session, RaceResult, QualifyingResult

    event = Event.objects.filter(id=event_id).select_related('circuit').first()
    if not event:
        return {}

    classifier = RaceEventClassifier()
    chaos_score, category, metrics = classifier.chaos_score_and_category(event)

    # Get race session and results
    race_session = Session.objects.filter(event=event, session_type__session_type='RACE').first()
    rr_map: Dict[int, RaceResult] = {}
    if race_session:
        for rr in RaceResult.objects.filter(session=race_session).select_related('driver'):
            rr_map[rr.driver_id] = rr

    preds = CatBoostPrediction.objects.filter(event=event, model_name=model_name).select_related('driver')

    def reasons_for_driver(rr: Optional[RaceResult], is_retired: bool) -> List[str]:
        reasons: List[str] = []
        if not rr:
            reasons.append('No race result found')
            return reasons
        status = (rr.status or '').lower()
        # Retired/DNF only if truly retired (not a classified finisher)
        if is_retired:
            reasons.append('Retired/DNF')
        # Incidents and penalties can affect even classified finishers
        if any(k in status for k in ACCIDENT_KEYWORDS):
            reasons.append('Incident/collision')
        if any(k in status for k in PENALTY_KEYWORDS):
            reasons.append('Penalty impact')
        # Position swing (informational)
        swing = rr.position_gain if rr.position_gain is not None else (
            abs((rr.grid_position or 0) - (rr.position or 0)) if (rr.position is not None and rr.grid_position is not None) else 0
        )
        if swing is not None and abs(swing) >= 5:
            reasons.append(f'Large position swing ({int(abs(swing))})')
        # Grid penalty via quali
        try:
            q = QualifyingResult.objects.filter(session__event=event, driver=rr.driver).first()
            if q and q.position and rr.grid_position and (rr.grid_position - q.position) >= 4:
                reasons.append(f'Grid penalty (+{rr.grid_position - q.position})')
        except Exception:
            pass
        # Strategy heavy
        if metrics and metrics.position_volatility >= 5 and (rr.pit_stops or 0) >= 3:
            reasons.append('Strategy/VSC/SC-heavy race')
        # Early lap incident proxy
        if metrics and getattr(metrics, 'early_lap_incidents', 0) >= 1 and any(k in status for k in ACCIDENT_KEYWORDS):
            reasons.append('Early-lap incident involvement')
        return reasons

    drivers: List[Dict] = []
    for p in preds:
        rr = rr_map.get(p.driver_id)
        actual_pos = p.actual_position if p.actual_position is not None else (rr.position if rr else None)
        # If driver DNFed, show actual_pos as 'DNF' instead of a number for display clarity
        status = (rr.status or '').lower() if rr else ''
        is_retired = True if rr and any(k in status for k in DNF_KEYWORDS) else False
        if actual_pos is None and is_retired:
            actual_display = 'DNF'
        elif actual_pos is None:
            # No position and not retired: skip unreliable row
            continue
        else:
            actual_display = int(actual_pos)
        # Error should only compute against numeric finishes
        err = float(abs((p.predicted_position or 0) - float(actual_pos))) if isinstance(actual_display, int) else float(abs(p.predicted_position or 0))
        affected = was_driver_affected_by_chaos(rr, metrics) if rr else False
        # Position gain: positive means improved vs grid
        pos_gain = None
        if rr:
            if rr.position_gain is not None:
                pos_gain = int(rr.position_gain)
            elif rr.position is not None and rr.grid_position is not None:
                pos_gain = int((rr.grid_position or 0) - (rr.position or 0))
        drivers.append({
            'driver_id': p.driver_id,
            'driver': str(p.driver),
            'predicted_pos': float(p.predicted_position),
            'actual_pos': actual_display,
            'error': err,
            'affected': bool(affected),
            'position_gain': pos_gain,
            'reasons': reasons_for_driver(rr, is_retired),
        })

    # Sort by error for top of table
    drivers.sort(key=lambda d: d['error'], reverse=True)

    # Compute redistribution snapshot: top gainers/losers by position_gain
    gainers = [d for d in drivers if d.get('position_gain') is not None and d['position_gain'] > 0]
    losers = [d for d in drivers if d.get('position_gain') is not None and d['position_gain'] < 0]
    gainers.sort(key=lambda d: d['position_gain'], reverse=True)
    losers.sort(key=lambda d: d['position_gain'])  # most negative first
    top_gainers = [{'driver': d['driver'], 'position_gain': int(d['position_gain'])} for d in gainers[:3]]
    top_losers = [{'driver': d['driver'], 'position_gain': int(d['position_gain'])} for d in losers[:3]]

    ev = {
        'id': event.id,
        'name': event.name,
        'round': event.round,
        'year': event.year,
        'circuit': getattr(event.circuit, 'name', ''),
    }

    metrics_dict = {}
    if metrics:
        metrics_dict = {
            'dnfs': metrics.dnfs,
            'crashes': metrics.crashes,
            'penalties': metrics.penalties,
            'total_classified': metrics.total_classified,
            'weather_flag': bool(metrics.weather_flag),
            'position_volatility': float(metrics.position_volatility),
            'safety_car_proxy': float(metrics.safety_car_proxy) if metrics.safety_car_proxy is not None else None,
            'early_lap_incidents': int(getattr(metrics, 'early_lap_incidents', 0)),
            'median_pit_stops': float(getattr(metrics, 'median_pit_stops', 0.0)),
        }

    return {
        'event': ev,
        'chaos_score': float(chaos_score),
        'race_category': category,
        'metrics': metrics_dict,
        'drivers': drivers,
        'top_gainers': top_gainers,
        'top_losers': top_losers,
    }


    def simulate_perfect_chaos_prediction(self) -> float:
        df_cf = self.df.copy()
        df_cf.loc[df_cf['driver_affected'] == True, 'error'] = 0.0
        return float(np.mean(np.abs(df_cf['error'])))

    def group_mae(self) -> pd.DataFrame:
        g = self.df.groupby(['race_category', 'driver_affected'])['error'].mean().reset_index().rename(columns={'error': 'mae'})
        return g

    def group_mae_by_pos_group(self) -> pd.DataFrame:
        def pos_group(pos):
            if pos <= 6: return 'front'
            if pos <= 15: return 'midfield'
            return 'back'
        df = self.df.copy()
        df['pos_group'] = df['actual_pos'].apply(pos_group)
        g = df.groupby(['pos_group','race_category','driver_affected'])['error'].mean().reset_index().rename(columns={'error':'mae'})
        return g

    def bootstrap_ci(self, mask: pd.Series, n_boot: int = 2000, ci: float = 0.95, random_state: int = 42) -> Tuple[float, float, float]:
        rng = np.random.default_rng(random_state)
        x = self.df.loc[mask, 'error'].to_numpy()
        if x.size == 0:
            return float('nan'), float('nan'), float('nan')
        boot = []
        n = x.size
        for _ in range(n_boot):
            sample = x[rng.integers(0, n, n)]
            boot.append(float(np.mean(sample)))
        boot = np.array(boot)
        lower = float(np.percentile(boot, (1-ci)/2*100))
        upper = float(np.percentile(boot, (1+ci)/2*100))
        return float(np.mean(x)), lower, upper

    def significance_test(self, mask_a: pd.Series, mask_b: pd.Series) -> Dict:
        xa = self.df.loc[mask_a, 'error'].to_numpy()
        xb = self.df.loc[mask_b, 'error'].to_numpy()
        if xa.size == 0 or xb.size == 0:
            return {'method': 'NA', 'p_value': float('nan')}
        try:
            from scipy.stats import mannwhitneyu
            stat, p = mannwhitneyu(xa, xb, alternative='two-sided')
            return {'method': 'mannwhitneyu', 'stat': float(stat), 'p_value': float(p)}
        except Exception:
            # Fallback: bootstrap difference in means
            rng = np.random.default_rng(42)
            n_boot = 2000
            diffs = []
            for _ in range(n_boot):
                sa = xa[rng.integers(0, xa.size, xa.size)]
                sb = xb[rng.integers(0, xb.size, xb.size)]
                diffs.append(float(np.mean(sa) - np.mean(sb)))
            diffs = np.array(diffs)
            p = float(2 * min(np.mean(diffs >= 0), np.mean(diffs <= 0)))
            return {'method': 'bootstrap_diff_means', 'p_value': p}

