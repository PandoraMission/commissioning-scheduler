# visibility.py
"""
Visibility calculations using pandoravisibility with caching support.

This module:
- Computes visibility windows for observations using satellite TLE
- Caches visibility results both in-memory and on-disk
- Supports parallel computation for multiple targets
- Returns astropy Time objects for compatibility
"""

import os
import hashlib
import pickle
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Sequence, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from astropy.time import Time, TimeDelta
from astropy.coordinates import SkyCoord, get_sun
import astropy.units as u

from .models import Observation, SchedulerConfig
from .shine_scheduler import EphemerisProvider, EarthshinePointing

logger = logging.getLogger(__name__)

# Try to import pandoravisibility
try:
    from pandoravisibility import Visibility

    PANDORA_VISIBILITY_AVAILABLE = True
except ImportError:
    Visibility = None
    PANDORA_VISIBILITY_AVAILABLE = False
    logger.warning(
        "pandoravisibility not available - visibility calculations will fail"
    )


@dataclass
class VisibilityWindow:
    """A single visibility window."""

    start: Time
    end: Time

    @property
    def duration_minutes(self) -> float:
        """Duration of window in minutes."""
        return (self.end - self.start).to_value("min")

    @property
    def duration_seconds(self) -> float:
        """Duration of window in seconds."""
        return (self.end - self.start).to_value("sec")


@dataclass
class VisibilityResult:
    """Result container for visibility computation."""

    ra: float
    dec: float
    times: Time  # astropy Time array
    visible_bool: np.ndarray  # boolean array matching 'times'
    windows: List[VisibilityWindow]

    @property
    def total_visible_time_minutes(self) -> float:
        """Total visible time across all windows."""
        return sum(w.duration_minutes for w in self.windows)


def compute_antisolar_coordinates(
    time: Time, dec: float = 8.5
) -> Tuple[float, float]:
    """
    Compute antisolar coordinates at a given time.

    Args:
        time: Time for computation
        dec: Declination (default: 8.5 for CVZ)

    Returns:
        Tuple of (RA, DEC) in degrees
    """
    sun_coord = get_sun(time)
    antisolar_ra = (sun_coord.ra.deg + 180.0) % 360.0
    return antisolar_ra, dec


class VisibilityCalculator:
    """
    Compute visibility windows using pandoravisibility.

    Features:
    - In-memory and on-disk caching
    - Parallel computation support
    - Attaches visibility windows to Observation objects
    """

    def __init__(
        self,
        config: SchedulerConfig,
        start: Time,
        stop: Time,
        timestep_seconds: int = 60,
    ):
        """
        Initialize visibility calculator.

        Args:
            config: Scheduler configuration (contains TLE)
            start: Start time for visibility grid
            stop: Stop time for visibility grid
            timestep_seconds: Time step for visibility computation (default: 60s)
        """
        if not PANDORA_VISIBILITY_AVAILABLE:
            raise ImportError(
                "pandoravisibility is required but not installed. "
                "Install with: pip install git+https://github.com/PandoraMission/pandora-visibility.git"
            )

        self.config = config
        self.start = start
        self.stop = stop
        self.timestep_seconds = timestep_seconds

        # Create time grid
        n_steps = int((stop - start).to_value("sec") / timestep_seconds)
        deltas = np.arange(n_steps) * timestep_seconds * u.s
        self.times = start + TimeDelta(deltas)

        logger.info(
            f"Creating time grid: {len(self.times)} points from {start.iso} to {stop.iso}"
        )

        # Initialize pandoravisibility
        self.vis = Visibility(config.tle_line1, config.tle_line2)

        # Cache setup - ensure directory exists
        self.cache_dir = config.cache_dir
        if self.cache_dir is None:
            # Fallback to hidden directory in current working directory
            self.cache_dir = os.path.join(os.getcwd(), ".pandora_vis_cache")

        if config.use_visibility_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Using visibility cache directory: {self.cache_dir}")

        self._memory_cache: Dict[Tuple[float, float], VisibilityResult] = {}
        self._state_cache: Optional[Any] = None  # Cache the spacecraft state

        logger.info(
            f"Initialized VisibilityCalculator: {start.iso} to {stop.iso}, "
            f"{timestep_seconds}s steps, {len(self.times)} points"
        )

    def _get_spacecraft_state(self):
        """Get spacecraft state, computing or using cache."""
        if self._state_cache is None:
            logger.info(
                "Computing spacecraft state (this may take a moment)..."
            )
            self._state_cache = self.vis.get_state(self.times)
            logger.info("Spacecraft state computed")
        return self._state_cache

    def compute_for_observations(
        self,
        observations: Sequence[Observation],
        force_recompute: bool = False,
        parallel: bool = False,
        max_workers: int = 4,
        attach_to_observations: bool = True,
    ) -> Dict[str, VisibilityResult]:
        """
        Compute visibility for all observations.

        Args:
            observations: List of Observation objects
            force_recompute: If True, ignore caches
            parallel: If True, compute in parallel
            max_workers: Number of parallel workers
            attach_to_observations: If True, attach windows to observation.visibility_windows

        Returns:
            Dictionary mapping obs_id to VisibilityResult
        """
        # Pre-compute spacecraft state once for all targets
        self._get_spacecraft_state()

        # Group by unique (ra, dec) to avoid redundant computation
        target_map: Dict[Tuple[float, float], List[Observation]] = {}
        for obs in observations:
            if obs.boresight_ra is None or obs.boresight_dec is None:
                logger.warning(
                    f"Observation {obs.obs_id} missing coordinates, skipping visibility"
                )
                continue
            key = (float(obs.boresight_ra), float(obs.boresight_dec))
            target_map.setdefault(key, []).append(obs)

        logger.info(
            f"Computing visibility for {len(target_map)} unique targets"
        )

        # Compute visibility for each unique target
        results = {}

        if parallel and len(target_map) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_radec = {
                    executor.submit(
                        self._compute_for_target, ra, dec, force_recompute
                    ): (ra, dec)
                    for ra, dec in target_map.keys()
                }

                for future in as_completed(future_to_radec):
                    ra, dec = future_to_radec[future]
                    try:
                        vis_result = future.result()
                        # Attach to all observations with this coordinate
                        for obs in target_map[(ra, dec)]:
                            if attach_to_observations:
                                obs.visibility_windows = [
                                    (w.start.datetime, w.end.datetime)
                                    for w in vis_result.windows
                                ]
                            results[obs.obs_id] = vis_result
                    except Exception as e:
                        logger.error(
                            f"Error computing visibility for ({ra}, {dec}): {e}"
                        )
        else:
            for (ra, dec), obs_list in target_map.items():
                try:
                    vis_result = self._compute_for_target(
                        ra, dec, force_recompute
                    )
                    for obs in obs_list:
                        if attach_to_observations:
                            obs.visibility_windows = [
                                (w.start.datetime, w.end.datetime)
                                for w in vis_result.windows
                            ]
                        results[obs.obs_id] = vis_result
                except Exception as e:
                    logger.error(
                        f"Error computing visibility for ({ra}, {dec}): {e}"
                    )

        logger.info(f"Computed visibility for {len(results)} observations")
        return results

    def _compute_for_target(
        self, ra: float, dec: float, force_recompute: bool = False
    ) -> VisibilityResult:
        """
        Compute visibility for a single target.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            force_recompute: Ignore caches

        Returns:
            VisibilityResult
        """
        key = (ra, dec)

        # Check memory cache
        if not force_recompute and key in self._memory_cache:
            logger.debug(f"Using memory cache for ({ra:.4f}, {dec:.4f})")
            return self._memory_cache[key]

        # Check disk cache
        if not force_recompute and self.config.use_visibility_cache:
            cached = self._load_from_disk_cache(ra, dec)
            if cached is not None:
                logger.debug(f"Using disk cache for ({ra:.4f}, {dec:.4f})")
                self._memory_cache[key] = cached
                return cached

        # Compute visibility
        logger.info(f"Computing visibility for ({ra:.4f}, {dec:.4f})")

        # Create target coordinate
        target_coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")

        # Get visibility boolean array
        # Note: spacecraft state is already computed and cached
        visible_bool = self.vis.get_visibility(target_coord, self.times)

        # Convert to numpy boolean array if needed
        visible_bool = np.asarray(visible_bool, dtype=bool)

        # Find contiguous visible windows
        windows = self._extract_windows(visible_bool)

        result = VisibilityResult(
            ra=ra,
            dec=dec,
            times=self.times,
            visible_bool=visible_bool,
            windows=windows,
        )

        # Cache results
        self._memory_cache[key] = result
        if self.config.use_visibility_cache:
            self._save_to_disk_cache(ra, dec, result)

        logger.info(
            f"Found {len(windows)} visibility windows for ({ra:.4f}, {dec:.4f}), "
            f"total {result.total_visible_time_minutes:.1f} minutes"
        )

        return result

    def _extract_windows(
        self, visible_bool: np.ndarray
    ) -> List[VisibilityWindow]:
        """
        Extract contiguous visibility windows from boolean array.

        Args:
            visible_bool: Boolean array indicating visibility at each timestep

        Returns:
            List of VisibilityWindow objects
        """
        windows = []

        if len(visible_bool) == 0:
            return windows

        # Find transitions
        # Add padding to handle edge cases
        padded = np.concatenate([[False], visible_bool, [False]])
        transitions = np.diff(padded.astype(int))

        starts = np.where(transitions == 1)[0]  # Rising edge
        ends = np.where(transitions == -1)[0]  # Falling edge

        # Create windows
        for start_idx, end_idx in zip(starts, ends):
            # end_idx is the first non-visible point, so use end_idx - 1
            if end_idx > start_idx and end_idx <= len(self.times):
                window = VisibilityWindow(
                    start=self.times[start_idx],
                    end=self.times[min(end_idx, len(self.times) - 1)],
                )
                # Only include windows longer than 1 minute
                if window.duration_minutes >= 1.0:
                    windows.append(window)

        return windows

    def _get_cache_key(self, ra: float, dec: float) -> str:
        """Generate cache key for a target."""
        # Include TLE and time grid in hash to invalidate if changed
        cache_str = (
            f"{ra:.6f}_{dec:.6f}_"
            f"{self.config.tle_line1}_"
            f"{self.config.tle_line2}_"
            f"{self.start.iso}_{self.stop.iso}_{self.timestep_seconds}"
        )
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _load_from_disk_cache(
        self, ra: float, dec: float
    ) -> Optional[VisibilityResult]:
        """Load visibility result from disk cache."""
        cache_key = self._get_cache_key(ra, dec)
        cache_path = os.path.join(self.cache_dir, f"vis_{cache_key}.pkl")

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for ({ra}, {dec}): {e}")

        return None

    def _save_to_disk_cache(
        self, ra: float, dec: float, result: VisibilityResult
    ) -> None:
        """Save visibility result to disk cache."""
        cache_key = self._get_cache_key(ra, dec)
        cache_path = os.path.join(self.cache_dir, f"vis_{cache_key}.pkl")

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
            logger.debug(f"Saved visibility cache for ({ra}, {dec})")
        except Exception as e:
            logger.warning(f"Failed to save cache for ({ra}, {dec}): {e}")

    def clear_cache(self, memory: bool = True, disk: bool = False) -> None:
        """
        Clear visibility cache.

        Args:
            memory: Clear in-memory cache
            disk: Clear on-disk cache
        """
        if memory:
            self._memory_cache.clear()
            self._state_cache = None
            logger.info("Cleared memory cache")

        if disk:
            import glob

            cache_files = glob.glob(os.path.join(self.cache_dir, "vis_*.pkl"))
            for f in cache_files:
                try:
                    os.remove(f)
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {f}: {e}")
            logger.info(f"Cleared {len(cache_files)} disk cache files")

    def find_cvz_pointings_for_gaps(
        self,
        gaps: List[Tuple[Time, Time]],
        dec: float = 8.5,
        ra_step: float = 1.0,
    ) -> Dict[Tuple[Time, Time], Tuple[float, float]]:
        """
        Find visible CVZ pointings for a list of time gaps.

        Args:
            gaps: List of (start, end) Time tuples
            dec: CVZ declination
            ra_step: RA step size for search

        Returns:
            Dictionary mapping gaps to (RA, DEC) coordinates
        """
        cvz_pointings = {}

        for gap_start, gap_end in gaps:
            coords = find_visible_cvz_pointing(
                self, gap_start, gap_end, dec, ra_step
            )
            if coords:
                cvz_pointings[(gap_start, gap_end)] = coords
            else:
                # Fallback to antisolar even if not fully visible
                midpoint = gap_start + (gap_end - gap_start) / 2
                antisolar_ra, fallback_dec = compute_antisolar_coordinates(
                    midpoint, dec
                )
                logger.warning(
                    f"Using antisolar coordinates as fallback for gap "
                    f"{gap_start.iso} to {gap_end.iso}"
                )
                cvz_pointings[(gap_start, gap_end)] = (
                    antisolar_ra,
                    fallback_dec,
                )

        return cvz_pointings

    def compute_visibility_for_window(
        self,
        ra: float,
        dec: float,
        window_start: Time,
        window_end: Time,
        window_only: bool = False,  # NEW PARAMETER
        visibility_threshold: float = 1.0,  # NEW: for 100% requirement
    ) -> Union[bool, Tuple[bool, np.ndarray, Time]]:
        """
        Check if a target is visible during a specific time window.

        This is more efficient than computing full visibility when you only
        need to check a short time range.

        Args:
            ra: Right ascension (degrees)
            dec: Declination (degrees)
            window_start: Start of time window
            window_end: End of time window
            window_only: If True, compute visibility ONLY for this window (faster)
            visibility_threshold: Fraction of window that must be visible (default 1.0 = 100%)

        Returns:
            If window_only=False: bool (True if visible)
            If window_only=True: Tuple of (is_visible, visible_bool_array, time_array)
        """
        if window_only:
            # Compute visibility ONLY for this specific window - much faster!
            n_steps = max(
                3,
                int(
                    (window_end - window_start).to_value("sec")
                    / self.timestep_seconds
                ),
            )
            deltas = (
                np.linspace(
                    0, (window_end - window_start).to_value("sec"), n_steps
                )
                * u.s
            )
            window_times = window_start + TimeDelta(deltas)

            target_coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")
            visible_bool = self.vis.get_visibility(target_coord, window_times)
            visible_bool = np.asarray(visible_bool, dtype=bool)

            visibility_fraction = np.sum(visible_bool) / len(visible_bool)
            is_visible = visibility_fraction >= visibility_threshold

            return is_visible, visible_bool, window_times

        else:
            # Original behavior - use precomputed full grid
            start_idx = np.searchsorted(self.times.jd, window_start.jd)
            end_idx = np.searchsorted(self.times.jd, window_end.jd)

            if start_idx >= len(self.times) or end_idx == 0:
                # Window outside precomputed range - compute just for window
                logger.debug(
                    "Window outside precomputed range, computing visibility for window"
                )
                n_steps = int(
                    (window_end - window_start).to_value("sec")
                    / self.timestep_seconds
                )
                deltas = np.arange(n_steps) * self.timestep_seconds * u.s
                window_times = window_start + TimeDelta(deltas)

                target_coord = SkyCoord(
                    ra=ra, dec=dec, unit="deg", frame="icrs"
                )
                visible_bool = self.vis.get_visibility(
                    target_coord, window_times
                )

                visibility_fraction = np.sum(visible_bool) / len(visible_bool)
                return visibility_fraction >= visibility_threshold

            start_idx = max(0, start_idx)
            end_idx = min(len(self.times), end_idx)

            if end_idx <= start_idx:
                return False

            vis_result = self._compute_for_target(
                ra, dec, force_recompute=False
            )
            window_visible = vis_result.visible_bool[start_idx:end_idx]

            if len(window_visible) == 0:
                return False

            visibility_fraction = np.sum(window_visible) / len(window_visible)
            return visibility_fraction >= visibility_threshold

    def compute_earthshine_visibility(
        self, observations: List[Observation], force_recompute: bool = False
    ) -> None:
        """
        Compute visibility for Earthshine observations based on orbital position.

        For Earthshine, "visibility" means the spacecraft is at the required
        orbital position (not that a sky coordinate is visible).

        Args:
            observations: List of Earthshine observations
            force_recompute: Force recomputation
        """
        from .shine_scheduler import EphemerisProvider, EarthshinePointing

        # Filter for Earthshine observations only
        earthshine_obs = [
            o for o in observations if getattr(o, "is_earthshine", False)
        ]

        if not earthshine_obs:
            return

        logger.info(
            f"Computing orbital position visibility for {len(earthshine_obs)} Earthshine observations"
        )

        # Create ephemeris provider (reuse TLE from config)
        ephemeris = EphemerisProvider(
            tle_line1=self.config.tle_line1, tle_line2=self.config.tle_line2
        )

        earthshine_calc = EarthshinePointing(ephemeris)

        # For each Earthshine observation, find when spacecraft is at its orbital position
        for obs in earthshine_obs:
            orbital_pos = obs.orbital_position_deg
            max_drift = obs.max_orbital_drift_deg
            duration_minutes = (
                obs.calculated_duration_minutes or obs.duration or 0.0
            )

            logger.info(
                f"Computing orbital visibility for {obs.obs_id}: "
                f"position={orbital_pos}°, max_drift={max_drift}°"
            )

            # Find all times in our time grid when spacecraft is at this position
            windows = self._find_orbital_position_windows(
                earthshine_calc, orbital_pos, max_drift, duration_minutes
            )

            # Attach windows to observation
            obs.visibility_windows = [
                (w.start.datetime, w.end.datetime) for w in windows
            ]

            total_time = sum(w.duration_minutes for w in windows)
            logger.info(
                f"  Found {len(windows)} windows for {obs.obs_id}, "
                f"total {total_time:.1f} minutes"
            )

    def _find_orbital_position_windows(
        self,
        earthshine_calc: EarthshinePointing,
        target_position_deg: float,
        tolerance_deg: float,
        min_duration_minutes: float,
    ) -> List[VisibilityWindow]:
        """
        Find time windows when spacecraft is at target orbital position.

        Args:
            earthshine_calc: EarthshinePointing calculator
            target_position_deg: Target orbital position (0-360)
            tolerance_deg: Position tolerance
            min_duration_minutes: Minimum window duration

        Returns:
            List of VisibilityWindow objects
        """
        # Check orbital position at each time step
        at_position = np.zeros(len(self.times), dtype=bool)

        for i, time in enumerate(self.times):
            # Get spacecraft state
            sc_state = earthshine_calc.ephemeris.get_spacecraft_state(time)

            # Get orbital position
            current_position = earthshine_calc._get_orbital_position(sc_state)

            # Check if within tolerance
            delta = abs(current_position - target_position_deg)
            if delta > 180:
                delta = 360 - delta

            at_position[i] = delta <= tolerance_deg

        # Extract windows using existing method
        windows = self._extract_windows(at_position)

        # Filter by minimum duration
        windows = [
            w for w in windows if w.duration_minutes >= min_duration_minutes
        ]

        return windows


def find_visible_cvz_pointing(
    calc: VisibilityCalculator,
    start_time: Time,
    end_time: Time,
    initial_dec: float = 8.5,
    ra_step: float = 0.2,
    max_iterations: int = 100,
    skip_visibility_check: bool = False,
) -> Optional[Tuple[float, float]]:
    """
    Find visible CVZ pointing coordinates for a time window.

    Starts with antisolar coordinates and steps RA forward until a visible
    pointing is found.

    Args:
        calc: VisibilityCalculator instance
        start_time: Start of window requiring CVZ pointing
        end_time: End of window requiring CVZ pointing
        initial_dec: Declination for CVZ (default: 8.5)
        ra_step: Step size in RA degrees (default: 1.0)
        max_iterations: Maximum number of RA steps to try (default: 180)
        skip_visibility_check: If True, return antisolar coordinates without checking visibility

    Returns:
        Tuple of (RA, DEC) if visible pointing found, None otherwise
    """
    # Start with antisolar coordinates at midpoint of window
    midpoint = start_time + (end_time - start_time) / 2
    antisolar_ra, dec = compute_antisolar_coordinates(midpoint, initial_dec)

    # If skipping visibility check, return antisolar immediately
    if skip_visibility_check:
        logger.debug(
            f"Skipping CVZ visibility check, using antisolar RA={antisolar_ra:.2f}"
        )
        return antisolar_ra, dec

    required_duration_minutes = (end_time - start_time).to_value("min")

    logger.info(
        f"Finding CVZ pointing for {required_duration_minutes:.1f} min window "
        f"starting at antisolar RA={antisolar_ra:.2f}"
    )

    # Try antisolar coordinates first, then step forward in RA
    for i in range(max_iterations):
        test_ra = (antisolar_ra + i * ra_step) % 360.0

        # Use the efficient window-specific visibility check
        is_visible = calc.compute_visibility_for_window(
            test_ra, dec, start_time, end_time
        )

        if is_visible:
            logger.info(
                f"Found visible CVZ pointing: RA={test_ra:.2f}, DEC={dec:.2f} "
                f"(offset {i * ra_step:.1f}° from antisolar)"
            )
            return test_ra, dec

    logger.warning(
        f"Could not find visible CVZ pointing after {max_iterations} iterations "
        f"({max_iterations * ra_step}° range), using antisolar as fallback"
    )
    # Fallback to antisolar
    return antisolar_ra, dec


def compute_target_visibility(
    ra: float,
    dec: float,
    config: SchedulerConfig,
    start: Time,
    stop: Time,
    timestep_seconds: int = 60,
) -> VisibilityResult:
    """
    Convenience function to compute visibility for a single target.

    Args:
        ra: Right ascension (degrees)
        dec: Declination (degrees)
        config: Scheduler configuration
        start: Start time
        stop: Stop time
        timestep_seconds: Time step

    Returns:
        VisibilityResult
    """
    calc = VisibilityCalculator(config, start, stop, timestep_seconds)
    return calc._compute_for_target(ra, dec)


def get_default_cache_dir() -> str:
    """
    Get the default cache directory path.

    Returns:
        Path to default cache directory
    """
    import os

    home = os.path.expanduser("~")
    return os.path.join(home, ".pandora_scheduler_cache")


def clear_visibility_cache(
    cache_dir: Optional[str] = None, confirm: bool = True
) -> int:
    """
    Clear all visibility cache files.

    Args:
        cache_dir: Cache directory to clear (default: user's home cache)
        confirm: If True, ask for confirmation before deleting

    Returns:
        Number of files deleted
    """
    import glob

    if cache_dir is None:
        cache_dir = get_default_cache_dir()

    if not os.path.exists(cache_dir):
        logger.info(f"Cache directory does not exist: {cache_dir}")
        return 0

    cache_files = glob.glob(os.path.join(cache_dir, "vis_*.pkl"))

    if not cache_files:
        logger.info(f"No cache files found in {cache_dir}")
        return 0

    if confirm:
        print(f"Found {len(cache_files)} cache files in {cache_dir}")
        response = input("Delete all cache files? (yes/no): ").strip().lower()
        if response not in ["yes", "y"]:
            logger.info("Cache clearing cancelled")
            return 0

    deleted = 0
    for f in cache_files:
        try:
            os.remove(f)
            deleted += 1
        except Exception as e:
            logger.warning(f"Failed to remove {f}: {e}")

    logger.info(f"Deleted {deleted} cache files from {cache_dir}")
    return deleted


def get_cache_info(cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Get information about the visibility cache.

    Args:
        cache_dir: Cache directory to inspect (default: user's home cache)

    Returns:
        Dictionary with cache statistics
    """
    import glob

    if cache_dir is None:
        cache_dir = get_default_cache_dir()

    if not os.path.exists(cache_dir):
        return {
            "exists": False,
            "path": cache_dir,
            "file_count": 0,
            "total_size_mb": 0.0,
        }

    cache_files = glob.glob(os.path.join(cache_dir, "vis_*.pkl"))

    total_size = 0
    for f in cache_files:
        try:
            total_size += os.path.getsize(f)
        except Exception:
            pass

    return {
        "exists": True,
        "path": cache_dir,
        "file_count": len(cache_files),
        "total_size_mb": total_size / (1024 * 1024),
    }


def find_visible_cvz_pointing_adaptive(
    calc: VisibilityCalculator,
    start_time: Time,
    end_time: Time,
    initial_dec: float = 8.5,
    ra_search_range: float = 20.0,
    ra_step: float = 0.2,
    min_window_minutes: float = 2.0,
    skip_visibility_check: bool = False,
) -> Union[Tuple[float, float], List[Tuple[Time, Time, float, float]]]:
    """
    Find visible CVZ pointing(s) using adaptive multi-stage strategy.

    Stages:
    1. Bidirectional RA search at Dec=8.5
    2. Intelligent window splitting based on visibility data
    3. Fallback coordinates (±45° RA, ±45° Dec from CVZ)
    4. Constraint relaxation (last resort with warnings)

    Args:
        calc: VisibilityCalculator instance
        start_time: Start of window
        end_time: End of window
        initial_dec: CVZ declination (default: 8.5)
        ra_search_range: How many degrees to search on each side of antisolar (default: 5.0)
        ra_step: RA step size for search (default: 0.2)
        min_window_minutes: Minimum sub-window duration (default: 2.0)
        skip_visibility_check: If True, return antisolar without checking

    Returns:
        Either:
        - Tuple[float, float]: Single (RA, Dec) for entire window
        - List[Tuple[Time, Time, float, float]]: Multiple (start, end, RA, Dec) sub-windows
    """
    midpoint = start_time + (end_time - start_time) / 2
    antisolar_ra, dec = compute_antisolar_coordinates(midpoint, initial_dec)

    if skip_visibility_check:
        logger.debug(
            f"Skipping CVZ visibility check, using antisolar RA={antisolar_ra:.2f}"
        )
        return antisolar_ra, dec

    window_duration_minutes = (end_time - start_time).to_value("min")
    logger.info(
        f"Finding CVZ pointing for {window_duration_minutes:.1f} min window "
        f"(antisolar RA={antisolar_ra:.2f}, Dec={dec:.2f})"
    )

    # ========================================================================
    # STAGE 1: Bidirectional RA search at Dec=8.5
    # ========================================================================
    logger.debug("Stage 1: Bidirectional RA search at CVZ Dec")
    result = _bidirectional_ra_search(
        calc, start_time, end_time, antisolar_ra, dec, ra_search_range, ra_step
    )

    if result is not None:
        ra_found, dec_found = result
        logger.info(
            f"✓ Found single CVZ pointing: RA={ra_found:.2f}, Dec={dec_found:.2f} "
            f"(offset {((ra_found - antisolar_ra + 180) % 360 - 180):.1f}° from antisolar)"
        )
        return ra_found, dec_found

    logger.info("Stage 1 failed: No single CVZ pointing covers entire window")

    # ========================================================================
    # STAGE 2: Intelligent window splitting
    # ========================================================================
    logger.info("Stage 2: Attempting intelligent window splitting...")
    split_result = _split_window_with_cvz(
        calc,
        start_time,
        end_time,
        antisolar_ra,
        dec,
        ra_search_range,
        ra_step,
        min_window_minutes,
    )

    if split_result is not None:
        logger.info(
            f"✓ Successfully split window into {len(split_result)} sub-windows:"
        )
        for i, (t_start, t_end, ra, dec_val) in enumerate(split_result, 1):
            duration = (t_end - t_start).to_value("min")
            logger.info(
                f"  Sub-window {i}: {t_start.iso} to {t_end.iso} ({duration:.1f} min) "
                f"-> RA={ra:.2f}, Dec={dec_val:.2f}"
            )
        return split_result

    logger.warning(
        "Stage 2 failed: Could not split window with CVZ coordinates"
    )

    # ========================================================================
    # STAGE 3: Fallback coordinates
    # ========================================================================
    logger.warning("Stage 3: Trying fallback coordinates (far from CVZ)...")
    fallback_result = _try_fallback_coordinates_with_splitting(
        calc,
        start_time,
        end_time,
        antisolar_ra,
        initial_dec,
        min_window_minutes,
    )

    if fallback_result is not None:
        if isinstance(fallback_result, tuple) and len(fallback_result) == 2:
            # Single pointing worked
            logger.warning(
                f"✓ Fallback coordinate covers window: RA={fallback_result[0]:.2f}, "
                f"Dec={fallback_result[1]:.2f}"
            )
        else:
            # Multiple sub-windows
            logger.warning(
                f"✓ Fallback coordinates split into {len(fallback_result)} sub-windows"
            )
        return fallback_result

    logger.error("Stage 3 failed: Even fallback coordinates don't work!")

    # ========================================================================
    # STAGE 4: Constraint relaxation (LAST RESORT)
    # ========================================================================
    logger.error("=" * 80)
    logger.error(
        "⚠️  CONSTRAINT RELAXATION REQUIRED - SCHEDULE MAY BE INVALID  ⚠️"
    )
    logger.error("=" * 80)
    logger.error(f"Window: {start_time.iso} to {end_time.iso}")
    logger.error("No valid pointing found with standard constraints.")
    logger.error("Attempting progressive constraint relaxation...")

    relaxed_result = _find_with_relaxed_constraints(
        calc, start_time, end_time, antisolar_ra, initial_dec
    )

    logger.error("=" * 80)
    logger.error(
        f"⚠️  USING RELAXED CONSTRAINTS: RA={relaxed_result[0]:.2f}, "
        f"Dec={relaxed_result[1]:.2f}  ⚠️"
    )
    logger.error("=" * 80)

    return relaxed_result


def _bidirectional_ra_search(
    calc: VisibilityCalculator,
    start_time: Time,
    end_time: Time,
    antisolar_ra: float,
    dec: float,
    search_range: float,
    ra_step: float,
) -> Optional[Tuple[float, float]]:
    """
    Search RA bidirectionally from antisolar point.

    Search pattern: antisolar, antisolar+step, antisolar-step, antisolar+2*step, etc.
    """
    # Try antisolar first
    is_visible = calc.compute_visibility_for_window(
        antisolar_ra,
        dec,
        start_time,
        end_time,
        window_only=True,
        visibility_threshold=1.0,
    )[0]

    if is_visible:
        return antisolar_ra, dec

    # Bidirectional search
    n_steps = int(search_range / ra_step)

    for i in range(1, n_steps + 1):
        # Try positive offset
        test_ra_pos = (antisolar_ra + i * ra_step) % 360.0
        is_visible = calc.compute_visibility_for_window(
            test_ra_pos,
            dec,
            start_time,
            end_time,
            window_only=True,
            visibility_threshold=1.0,
        )[0]

        if is_visible:
            return test_ra_pos, dec

        # Try negative offset
        test_ra_neg = (antisolar_ra - i * ra_step) % 360.0
        is_visible = calc.compute_visibility_for_window(
            test_ra_neg,
            dec,
            start_time,
            end_time,
            window_only=True,
            visibility_threshold=1.0,
        )[0]

        if is_visible:
            return test_ra_neg, dec

    return None


def _split_window_with_cvz(
    calc: VisibilityCalculator,
    start_time: Time,
    end_time: Time,
    antisolar_ra: float,
    dec: float,
    ra_search_range: float,
    ra_step: float,
    min_window_minutes: float,
) -> Optional[List[Tuple[Time, Time, float, float]]]:
    """
    Intelligently split window based on visibility data from RA search.

    Strategy:
    1. Test all RAs in search range, get detailed visibility for each
    2. Find RA with longest visibility that overlaps window start
    3. Split window there, use that RA for first sub-window
    4. Recursively find pointing for remaining sub-window
    """
    # Collect visibility data for all tested RAs
    ra_visibility_data = []

    n_steps = int(ra_search_range / ra_step)
    test_ras = []

    # Build list of RAs to test (bidirectional)
    test_ras.append(antisolar_ra)
    for i in range(1, n_steps + 1):
        test_ras.append((antisolar_ra + i * ra_step) % 360.0)
        test_ras.append((antisolar_ra - i * ra_step) % 360.0)

    logger.debug(f"  Analyzing visibility for {len(test_ras)} RA values...")

    for test_ra in test_ras:
        is_visible, visible_bool, times = calc.compute_visibility_for_window(
            test_ra,
            dec,
            start_time,
            end_time,
            window_only=True,
            visibility_threshold=1.0,
        )

        if is_visible:
            # Found full coverage!
            return None  # Signal caller to use this as single pointing

        ra_visibility_data.append(
            {
                "ra": test_ra,
                "visible_bool": visible_bool,
                "times": times,
            }
        )

    # Find best RA for covering window start
    best_ra_data = None
    longest_start_coverage = 0

    for data in ra_visibility_data:
        if not data["visible_bool"][0]:
            # Doesn't cover window start
            continue

        # Find how long it stays visible from start
        visible_duration = 0
        for i, vis in enumerate(data["visible_bool"]):
            if not vis:
                break
            visible_duration = (data["times"][i] - start_time).to_value("min")

        if visible_duration > longest_start_coverage:
            longest_start_coverage = visible_duration
            best_ra_data = data

    if best_ra_data is None or longest_start_coverage < min_window_minutes:
        logger.debug(
            "  Cannot find RA covering window start for minimum duration"
        )
        return None

    # Split window at end of best RA's visibility
    split_time = None
    for i, vis in enumerate(best_ra_data["visible_bool"]):
        if not vis:
            split_time = best_ra_data["times"][i - 1] if i > 0 else start_time
            break

    if split_time is None:
        split_time = end_time

    # Create first sub-window
    sub_windows = [(start_time, split_time, best_ra_data["ra"], dec)]

    logger.debug(
        f"  First sub-window: {start_time.iso} to {split_time.iso}, "
        f"RA={best_ra_data['ra']:.2f} ({longest_start_coverage:.1f} min)"
    )

    # Remaining time
    remaining_duration = (end_time - split_time).to_value("min")

    if remaining_duration < min_window_minutes:
        # Done!
        return sub_windows

    # Try to fill remaining time
    logger.debug(
        f"  Remaining time: {remaining_duration:.1f} min, searching for second pointing..."
    )

    # Try bidirectional search for remaining window
    second_pointing = _bidirectional_ra_search(
        calc, split_time, end_time, antisolar_ra, dec, ra_search_range, ra_step
    )

    if second_pointing is not None:
        sub_windows.append(
            (split_time, end_time, second_pointing[0], second_pointing[1])
        )
        logger.debug(f"  Second sub-window: RA={second_pointing[0]:.2f}")
        return sub_windows

    # Second pointing not found with CVZ - return None to trigger fallback stage
    logger.debug("  Could not find second CVZ pointing for remaining time")
    return None


def _try_fallback_coordinates_with_splitting(
    calc: VisibilityCalculator,
    start_time: Time,
    end_time: Time,
    antisolar_ra: float,
    cvz_dec: float,
    min_window_minutes: float,
) -> Optional[
    Union[Tuple[float, float], List[Tuple[Time, Time, float, float]]]
]:
    """
    Try 4 fallback coordinates, with intelligent splitting if needed.

    Fallback coordinates (far from CVZ):
    1. (antisolar_ra - 45°, cvz_dec)
    2. (antisolar_ra + 45°, cvz_dec)
    3. (antisolar_ra, +53.5°)
    4. (antisolar_ra, -36.5°)
    """
    fallback_coords = [
        ((antisolar_ra - 45.0) % 360.0, cvz_dec, "RA-45°"),
        ((antisolar_ra + 45.0) % 360.0, cvz_dec, "RA+45°"),
        (antisolar_ra, 53.5, "Dec+45°"),
        (antisolar_ra, -36.5, "Dec-45°"),
    ]

    # First try: Can any single fallback cover entire window?
    for ra, dec, label in fallback_coords:
        is_visible = calc.compute_visibility_for_window(
            ra,
            dec,
            start_time,
            end_time,
            window_only=True,
            visibility_threshold=1.0,
        )[0]

        if is_visible:
            logger.info(
                f"  Fallback {label} covers entire window: RA={ra:.2f}, Dec={dec:.2f}"
            )
            return ra, dec

    # Second try: Can we split using fallback coordinates?
    logger.debug("  Single fallback failed, trying to split with fallbacks...")

    # Collect visibility data for all fallbacks
    fallback_visibility = []
    for ra, dec, label in fallback_coords:
        is_visible, visible_bool, times = calc.compute_visibility_for_window(
            ra,
            dec,
            start_time,
            end_time,
            window_only=True,
            visibility_threshold=1.0,
        )
        fallback_visibility.append(
            {
                "ra": ra,
                "dec": dec,
                "label": label,
                "visible_bool": visible_bool,
                "times": times,
            }
        )

    # Find best coverage for window start
    best_start = None
    longest_start = 0

    for data in fallback_visibility:
        if not data["visible_bool"][0]:
            continue

        duration = 0
        for i, vis in enumerate(data["visible_bool"]):
            if not vis:
                break
            duration = (data["times"][i] - start_time).to_value("min")

        if duration > longest_start and duration >= min_window_minutes:
            longest_start = duration
            best_start = data

    if best_start is None:
        return None

    # Split at end of best start coverage
    split_time = None
    for i, vis in enumerate(best_start["visible_bool"]):
        if not vis:
            split_time = best_start["times"][i - 1] if i > 0 else start_time
            break

    if split_time is None:
        split_time = end_time

    sub_windows = [
        (start_time, split_time, best_start["ra"], best_start["dec"])
    ]

    remaining = (end_time - split_time).to_value("min")
    if remaining < min_window_minutes:
        return sub_windows

    # Try to cover remaining with another fallback
    for data in fallback_visibility:
        is_visible = calc.compute_visibility_for_window(
            data["ra"],
            data["dec"],
            split_time,
            end_time,
            window_only=True,
            visibility_threshold=1.0,
        )[0]

        if is_visible:
            sub_windows.append((split_time, end_time, data["ra"], data["dec"]))
            logger.info(
                f"  Split using fallbacks: {best_start['label']} + {data['label']}"
            )
            return sub_windows

    return None


def _find_with_relaxed_constraints(
    calc: VisibilityCalculator,
    start_time: Time,
    end_time: Time,
    antisolar_ra: float,
    dec: float,
) -> Tuple[float, float]:
    """
    Last resort: progressively relax constraints until valid pointing found.

    Relaxation order:
    1. Moon: 25° → 20° → 15° → 10° → 5°
    2. Earth limb: 20° → 15° → 10° → 5°
    3. If still failing, return antisolar (should never happen)
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    # Save original constraints
    original_moon = calc.vis.moon_min
    original_earth = calc.vis.earthlimb_min

    # Try progressively relaxed constraints
    moon_limits = [20, 15, 10, 5] * u.deg
    earth_limits = [15, 10, 5] * u.deg

    try:
        # Try relaxing Moon first
        for moon_limit in moon_limits:
            calc.vis.moon_min = moon_limit
            logger.error(f"  Trying Moon limit: {moon_limit}")

            is_visible = calc.compute_visibility_for_window(
                antisolar_ra,
                dec,
                start_time,
                end_time,
                window_only=True,
                visibility_threshold=1.0,
            )[0]

            if is_visible:
                logger.error(
                    f"  ✓ Success with Moon={moon_limit}, Earth={original_earth}"
                )
                return antisolar_ra, dec

        # Try relaxing Earth limb
        calc.vis.moon_min = original_moon  # Reset Moon
        for earth_limit in earth_limits:
            calc.vis.earthlimb_min = earth_limit
            logger.error(f"  Trying Earth limb: {earth_limit}")

            is_visible = calc.compute_visibility_for_window(
                antisolar_ra,
                dec,
                start_time,
                end_time,
                window_only=True,
                visibility_threshold=1.0,
            )[0]

            if is_visible:
                logger.error(
                    f"  ✓ Success with Moon={original_moon}, Earth={earth_limit}"
                )
                return antisolar_ra, dec

        # Try both relaxed
        for moon_limit in moon_limits:
            for earth_limit in earth_limits:
                calc.vis.moon_min = moon_limit
                calc.vis.earthlimb_min = earth_limit
                logger.error(
                    f"  Trying Moon={moon_limit}, Earth={earth_limit}"
                )

                is_visible = calc.compute_visibility_for_window(
                    antisolar_ra,
                    dec,
                    start_time,
                    end_time,
                    window_only=True,
                    visibility_threshold=1.0,
                )[0]

                if is_visible:
                    logger.error(
                        f"  ✓ Success with Moon={moon_limit}, Earth={earth_limit}"
                    )
                    return antisolar_ra, dec

    finally:
        # Always restore original constraints
        calc.vis.moon_min = original_moon
        calc.vis.earthlimb_min = original_earth

    # Absolute fallback - should never reach here
    logger.error(
        "  ⚠️  EXTREME FALLBACK: Using antisolar despite constraints ⚠️"
    )
    return antisolar_ra, dec
