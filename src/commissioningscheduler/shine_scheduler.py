# shine_scheduler.py
"""
Moonshine and Earthshine observation generation and scheduling.

This module integrates the shine_scheduler pointing calculations with
the Pandora scheduler, handling:
- Template XML parsing
- Observation generation with synthetic RA/Dec
- Duration calculation from instrument parameters
- Block scheduling for Moonshine
- Flexible and block scheduling for Earthshine
"""

import logging
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import copy

from astropy.time import Time
import astropy.units as u
import numpy as np

from .models import (
    Observation,
    ObservationSequence,
    SchedulerConfig,
    EarthshineConfig,
    MoonshineConfig,
)
from .xml_io import ObservationParser
from .utils import parse_utc_time, format_utc_time

# Import our shine pointing code
try:
    from .shine import (
        EphemerisProvider,
        MoonshinePointing,
        EarthshinePointing,
        MoonshineResult,
        EarthshineResult,
    )

    SHINE_AVAILABLE = True
except ImportError:
    SHINE_AVAILABLE = False
    logging.warning(
        "shine_scheduler module not available - Moonshine/Earthshine scheduling disabled"
    )

logger = logging.getLogger(__name__)


class ShineObservationGenerator:
    """
    Generate Moonshine and Earthshine observations from template XMLs.

    This class:
    - Parses template XML files for instrument parameters
    - Calculates observation durations from parameters
    - Generates synthetic observations for each position/separation combination
    - Prepares observations for scheduler integration
    """

    def __init__(
        self, config: SchedulerConfig, ephemeris_provider: EphemerisProvider
    ):
        """
        Initialize generator.

        Args:
            config: Scheduler configuration
            ephemeris_provider: Ephemeris provider for pointing calculations
        """
        if not SHINE_AVAILABLE:
            raise ImportError(
                "shine_scheduler module required for Moonshine/Earthshine observations"
            )

        self.config = config
        self.ephemeris = ephemeris_provider

        # Initialize pointing calculators
        self.moonshine_calc = MoonshinePointing(ephemeris_provider)
        self.earthshine_calc = EarthshinePointing(ephemeris_provider)

        # Template cache
        self._templates: Dict[str, Observation] = {}

    def generate_moonshine_observations(
        self, template_path: str, start_time: Time, end_time: Time
    ) -> List[Observation]:
        """
        Generate all Moonshine observation combinations.

        Args:
            template_path: Path to 0342_000_template_SOC.xml
            start_time: Scheduling window start
            end_time: Scheduling window end

        Returns:
            List of Observation objects (one per position/separation combo)
        """
        if (
            not self.config.moonshine_config
            or not self.config.moonshine_config.enabled
        ):
            return []

        logger.info(
            f"Generating Moonshine observations from template: {template_path}"
        )

        # Parse template
        template = self._parse_template(template_path)
        template.is_moonshine = True

        # Calculate duration
        duration_minutes = self._calculate_observation_duration(template)
        logger.info(
            f"Moonshine observation duration: {duration_minutes} minutes"
        )

        # Generate observations for all combinations
        observations = []
        ms_config = self.config.moonshine_config
        sequence_num = 0

        # Determine which combinations to generate
        combinations_to_generate = self._get_moonshine_combinations(ms_config)

        logger.info(
            f"Generating {len(combinations_to_generate)} Moonshine combinations"
        )

        for angular_pos, limb_sep in combinations_to_generate:
            obs = self._create_moonshine_observation(
                template,
                angular_pos,
                limb_sep,
                duration_minutes,
                sequence_num,
            )
            observations.append(obs)
            sequence_num += 1

        logger.info(f"Generated {len(observations)} Moonshine observations")

        return observations

    def generate_earthshine_observations(
        self, template_path: str, start_time: Time, end_time: Time
    ) -> List[Observation]:
        """
        Generate all Earthshine observation combinations.

        Args:
            template_path: Path to 0341_000_template_SOC.xml
            start_time: Scheduling window start
            end_time: Scheduling window end

        Returns:
            List of Observation objects (one per position/separation combo)
        """
        if (
            not self.config.earthshine_config
            or not self.config.earthshine_config.enabled
        ):
            return []

        logger.info(
            f"Generating Earthshine observations from template: {template_path}"
        )

        # Parse template
        template = self._parse_template(template_path)
        template.is_earthshine = True

        # Calculate duration
        duration_minutes = self._calculate_observation_duration(template)
        logger.info(
            f"Earthshine observation duration: {duration_minutes} minutes"
        )

        # Generate observations for all combinations
        observations = []
        es_config = self.config.earthshine_config
        sequence_num = 0

        # Determine which combinations to generate
        combinations_to_generate = self._get_earthshine_combinations(es_config)

        logger.info(
            f"Generating {len(combinations_to_generate)} Earthshine combinations"
        )

        for orbital_pos, limb_sep in combinations_to_generate:
            obs = self._create_earthshine_observation(
                template,
                orbital_pos,
                limb_sep,
                duration_minutes,
                sequence_num,
            )
            observations.append(obs)
            sequence_num += 1

        logger.info(f"Generated {len(observations)} Earthshine observations")

        return observations

    def _parse_template(self, template_path: str) -> Observation:
        """
        Parse template XML file.

        Args:
            template_path: Path to template XML

        Returns:
            Observation object with instrument parameters
        """
        # Check cache
        if template_path in self._templates:
            return copy.deepcopy(self._templates[template_path])

        # Parse XML
        parser = ObservationParser()
        template = parser.parse_file(template_path)
        template.is_template = True

        # Cache for reuse
        self._templates[template_path] = template

        return copy.deepcopy(template)

    def _calculate_observation_duration(self, obs: Observation) -> float:
        """
        Calculate observation duration from instrument parameters.

        Uses the longer of visible or NIR camera durations (they run in parallel).
        Returns duration in minutes, rounded up.

        Args:
            obs: Observation with instrument parameters

        Returns:
            Duration in minutes (rounded up)
        """
        # Visible camera duration (CORRECTED: no readout time added)
        vis_duration_sec = 0.0
        if obs.num_total_frames_requested and obs.exposure_time_us:
            vis_duration_sec = (
                obs.num_total_frames_requested * obs.exposure_time_us / 1e6
            )
            logger.debug(
                f"Visible duration: {obs.num_total_frames_requested} frames × "
                f"{obs.exposure_time_us} μs = {vis_duration_sec:.2f} sec"
            )

        # NIR camera duration
        nir_duration_sec = 0.0
        if obs.sc_integrations and obs.sc_integrations > 0:
            frame_count_term = (
                (obs.sc_resets1 or 0)
                + (obs.sc_resets2 or 0)
                + (obs.sc_dropframes1 or 0)
                + (obs.sc_dropframes2 or 0)
                + (obs.sc_dropframes3 or 0)
                + (obs.sc_readframes or 0)
                + 1
            )
            pixel_term = (obs.roi_sizex * obs.roi_sizey) + (obs.roi_sizey * 12)
            integration_duration_sec = frame_count_term * pixel_term * 0.00001
            nir_duration_sec = integration_duration_sec * obs.sc_integrations

            logger.debug(
                f"NIR duration: {obs.sc_integrations} integrations × "
                f"{integration_duration_sec:.4f} sec = {nir_duration_sec:.2f} sec"
            )

        # Take maximum (cameras run in parallel)
        total_duration_sec = max(vis_duration_sec, nir_duration_sec)

        # Round up to nearest minute
        duration_minutes = math.ceil(total_duration_sec / 60.0)

        return duration_minutes

    def _get_moonshine_combinations(
        self, ms_config: MoonshineConfig
    ) -> List[Tuple[float, float]]:
        """
        Determine which Moonshine combinations to generate based on config.

        Supports three modes:
        1. Whitelist: Use explicit combinations list
        2. Blacklist: Generate all combos, then remove exclusions
        3. Default: Generate all combos from positions × separations

        Args:
            ms_config: Moonshine configuration

        Returns:
            List of (angular_position, limb_separation) tuples
        """
        # Mode 1: Whitelist - use explicit combinations
        if ms_config.combinations is not None:
            logger.info(
                "Using whitelist mode: explicit combinations specified"
            )
            combinations = [
                (combo["angular_position"], combo["limb_separation"])
                for combo in ms_config.combinations
            ]
            return combinations

        # Mode 2 & 3: Generate all combinations from positions × separations
        all_combinations = [
            (angular_pos, limb_sep)
            for angular_pos in ms_config.angular_positions
            for limb_sep in ms_config.limb_separations
        ]

        # Mode 2: Blacklist - remove exclusions
        if ms_config.exclude_combinations is not None:
            logger.info(
                f"Using blacklist mode: excluding {len(ms_config.exclude_combinations)} combinations"
            )

            # Convert exclusions to set of tuples for fast lookup
            exclusions = {
                (excl["angular_position"], excl["limb_separation"])
                for excl in ms_config.exclude_combinations
            }

            # Filter out excluded combinations
            filtered_combinations = [
                combo for combo in all_combinations if combo not in exclusions
            ]

            logger.info(
                f"Filtered {len(all_combinations)} combinations down to {len(filtered_combinations)} "
                f"(excluded {len(all_combinations) - len(filtered_combinations)})"
            )

            return filtered_combinations

        # Mode 3: Default - return all combinations
        logger.info(
            "Using default mode: all position × separation combinations"
        )
        return all_combinations

    def _create_moonshine_observation(
        self,
        template: Observation,
        angular_position: float,
        limb_separation: float,
        duration_minutes: float,
        sequence_num: int,
    ) -> Observation:
        """
        Create a Moonshine observation for specific position/separation.

        Args:
            template: Template observation with instrument parameters
            angular_position: Angular position around Moon (degrees)
            limb_separation: Distance from limb (degrees)
            duration_minutes: Calculated duration
            sequence_num: Sequence number for obs_id

        Returns:
            New Observation object
        """
        obs = copy.deepcopy(template)

        # Set identifiers
        obs.obs_id = f"0342_{sequence_num:03d}"
        obs.task_number = "0342"
        obs.target_name = (
            f"Mshine_Pos{angular_position:.0f}_Sep{limb_separation:.0f}"
        )

        # Set shine-specific fields
        obs.is_moonshine = True
        obs.is_template = False
        obs.angular_position_deg = angular_position
        obs.limb_separation_deg = limb_separation
        obs.calculated_duration_minutes = duration_minutes
        obs.duration = duration_minutes

        # RA/Dec will be calculated at schedule time
        obs.boresight_ra = None
        obs.boresight_dec = None

        logger.debug(
            f"Created {obs.obs_id}: pos={angular_position}°, sep={limb_separation}°"
        )

        return obs

    def _get_earthshine_combinations(
        self, es_config: EarthshineConfig
    ) -> List[Tuple[float, float]]:
        """
        Determine which Earthshine combinations to generate based on config.

        Supports three modes:
        1. Whitelist: Use explicit combinations list
        2. Blacklist: Generate all combos, then remove exclusions
        3. Default: Generate all combos from positions × separations

        Args:
            es_config: Earthshine configuration

        Returns:
            List of (orbital_position, limb_separation) tuples
        """
        # Mode 1: Whitelist - use explicit combinations
        if es_config.combinations is not None:
            logger.info(
                "Using whitelist mode: explicit combinations specified"
            )
            combinations = [
                (combo["orbital_position"], combo["limb_separation"])
                for combo in es_config.combinations
            ]
            return combinations

        # Mode 2 & 3: Generate all combinations from positions × separations
        all_combinations = [
            (orbital_pos, limb_sep)
            for orbital_pos in es_config.orbital_positions
            for limb_sep in es_config.limb_separations
        ]

        # Mode 2: Blacklist - remove exclusions
        if es_config.exclude_combinations is not None:
            logger.info(
                f"Using blacklist mode: excluding {len(es_config.exclude_combinations)} combinations"
            )

            # Convert exclusions to set of tuples for fast lookup
            exclusions = {
                (excl["orbital_position"], excl["limb_separation"])
                for excl in es_config.exclude_combinations
            }

            # Filter out excluded combinations
            filtered_combinations = [
                combo for combo in all_combinations if combo not in exclusions
            ]

            logger.info(
                f"Filtered {len(all_combinations)} combinations down to {len(filtered_combinations)} "
                f"(excluded {len(all_combinations) - len(filtered_combinations)})"
            )

            return filtered_combinations

        # Mode 3: Default - return all combinations
        logger.info(
            "Using default mode: all position × separation combinations"
        )
        return all_combinations

    def _create_earthshine_observation(
        self,
        template: Observation,
        orbital_position: float,
        limb_separation: float,
        duration_minutes: float,
        sequence_num: int,
    ) -> Observation:
        """
        Create an Earthshine observation for specific position/separation.

        Args:
            template: Template observation with instrument parameters
            orbital_position: Orbital position (degrees, 0=North)
            limb_separation: Distance from limb (degrees)
            duration_minutes: Calculated duration
            sequence_num: Sequence number for obs_id

        Returns:
            New Observation object
        """
        obs = copy.deepcopy(template)

        # Set identifiers
        obs.obs_id = f"0341_{sequence_num:03d}"
        obs.task_number = "0341"
        obs.target_name = (
            f"Eshine_Pos{orbital_position:.0f}_Sep{limb_separation:.0f}"
        )

        # Set shine-specific fields
        obs.is_earthshine = True
        obs.is_template = False
        obs.orbital_position_deg = orbital_position
        obs.limb_separation_deg = limb_separation
        obs.calculated_duration_minutes = duration_minutes
        obs.duration = duration_minutes
        obs.max_orbital_drift_deg = (
            self.config.earthshine_config.max_orbital_drift_deg
        )

        # RA/Dec will be calculated at schedule time
        obs.boresight_ra = None
        obs.boresight_dec = None

        logger.debug(
            f"Created {obs.obs_id}: orb_pos={orbital_position}°, sep={limb_separation}°"
        )

        return obs


class MoonshineScheduler:
    """
    Schedules Moonshine observations in a contiguous block around full Moon.

    This class:
    - Finds the best time window near the target date
    - Reserves a contiguous block for all Moonshine observations
    - Schedules observations sequentially with proper RA/Dec pointing
    - Handles Earth occulting Moon (skips occluded observations)
    - Creates ObservationSequence objects for the final schedule
    """

    def __init__(
        self, config: SchedulerConfig, ephemeris_provider: EphemerisProvider
    ):
        """
        Initialize Moonshine scheduler.

        Args:
            config: Scheduler configuration
            ephemeris_provider: Ephemeris provider for pointing calculations
        """
        self.config = config
        self.ephemeris = ephemeris_provider
        self.moonshine_calc = MoonshinePointing(ephemeris_provider)

    def schedule_block(
        self,
        moonshine_observations: List[Observation],
        start_time: Time,
        end_time: Time,
        existing_schedule: List[ObservationSequence],
        allow_overflow: bool = True,
    ) -> Tuple[List[ObservationSequence], List[Observation]]:
        """
        Schedule all Moonshine observations in a contiguous block.

        Args:
            moonshine_observations: List of Moonshine observations to schedule
            start_time: Scheduling window start
            end_time: Scheduling window end
            existing_schedule: Already scheduled sequences (to avoid conflicts)
            allow_overflow: Allow scheduling beyond end_time if necessary

        Returns:
            Tuple of (scheduled_sequences, unscheduled_observations)
        """
        if not moonshine_observations:
            return [], []

        ms_config = self.config.moonshine_config
        if not ms_config or not ms_config.enabled:
            return [], moonshine_observations

        logger.info(
            f"Scheduling {len(moonshine_observations)} Moonshine observations"
        )

        # Calculate total block duration needed
        total_duration_minutes = self._calculate_block_duration(
            moonshine_observations
        )
        logger.info(
            f"Moonshine block requires {total_duration_minutes:.1f} minutes"
        )

        # Find the best time window
        block_start_time = self._find_moonshine_window(
            start_time, end_time, total_duration_minutes, existing_schedule
        )

        if block_start_time is None:
            if allow_overflow:
                # Try scheduling after end_time
                logger.warning(
                    "Moonshine block doesn't fit in window, trying overflow"
                )
                block_start_time = self._find_moonshine_window(
                    end_time,
                    end_time + 7 * u.day,  # Search up to 7 days beyond
                    total_duration_minutes,
                    existing_schedule,
                )

        if block_start_time is None:
            logger.error("Could not find suitable window for Moonshine block")
            return [], moonshine_observations

        # Schedule the block
        scheduled_sequences, unscheduled = self._schedule_moonshine_block(
            moonshine_observations, block_start_time
        )

        logger.info(
            f"Scheduled {len(scheduled_sequences)} Moonshine sequences, "
            f"{len(unscheduled)} unscheduled"
        )

        # Handle occluded observations (schedule them later)
        if unscheduled:
            logger.info(
                "Attempting to schedule occluded Moonshine observations individually"
            )
            additional_sequences = self._schedule_occluded_observations(
                unscheduled,
                block_start_time + total_duration_minutes * u.min,
                end_time,
            )
            scheduled_sequences.extend(additional_sequences)

            # Update unscheduled list
            scheduled_ids = {seq.obs_id for seq in additional_sequences}
            unscheduled = [
                obs for obs in unscheduled if obs.obs_id not in scheduled_ids
            ]

        return scheduled_sequences, unscheduled

    def _calculate_block_duration(
        self, observations: List[Observation]
    ) -> float:
        """
        Calculate total duration needed for Moonshine block.

        Includes observation durations plus slew overhead.

        Args:
            observations: List of Moonshine observations

        Returns:
            Total duration in minutes
        """
        if not observations:
            return 0.0

        # Sum all observation durations
        obs_duration = sum(
            obs.calculated_duration_minutes or obs.duration or 0.0
            for obs in observations
        )

        # Add slew overhead between observations
        # Assume 1 minute per slew (since we're pointing to different Moon positions)
        slew_overhead = (
            len(observations) - 1
        ) * self.config.slew_overhead_minutes

        total_duration = obs_duration + slew_overhead

        logger.debug(
            f"Moonshine block: {obs_duration:.1f} min observations + "
            f"{slew_overhead:.1f} min slews = {total_duration:.1f} min total"
        )

        return total_duration

    def _find_moonshine_window(
        self,
        start_time: Time,
        end_time: Time,
        duration_minutes: float,
        existing_schedule: List[ObservationSequence],
    ) -> Optional[Time]:
        """
        Find best time window for Moonshine block near target date.

        Prefers times close to target_date within window_days.

        Args:
            start_time: Search start
            end_time: Search end
            duration_minutes: Required duration
            existing_schedule: Existing scheduled sequences

        Returns:
            Start time for block, or None if not found
        """
        ms_config = self.config.moonshine_config
        target_time = Time(ms_config.target_date)
        window_days = ms_config.window_days

        # Define preferred window around target date
        preferred_start = max(start_time, target_time - window_days * u.day)
        preferred_end = min(end_time, target_time + window_days * u.day)

        logger.info(
            f"Searching for Moonshine window: target={target_time.iso}, "
            f"preferred={preferred_start.iso} to {preferred_end.iso}"
        )

        # Search for available window with 30-minute resolution
        search_step_minutes = 30.0
        current_time = preferred_start

        while current_time <= preferred_end:
            # Check if this window is free
            block_end_time = current_time + duration_minutes * u.min

            if self._is_window_available(
                current_time, block_end_time, existing_schedule
            ):
                logger.info(f"Found Moonshine window: {current_time.iso}")
                return current_time

            current_time += search_step_minutes * u.min

        # If no window in preferred range, search entire range
        if preferred_start > start_time or preferred_end < end_time:
            logger.info(
                "No window in preferred range, searching entire window"
            )
            current_time = start_time

            while current_time <= end_time - duration_minutes * u.min:
                block_end_time = current_time + duration_minutes * u.min

                if self._is_window_available(
                    current_time, block_end_time, existing_schedule
                ):
                    logger.info(
                        f"Found Moonshine window (extended search): {current_time.iso}"
                    )
                    return current_time

                current_time += search_step_minutes * u.min

        return None

    def _is_window_available(
        self,
        start_time: Time,
        end_time: Time,
        existing_schedule: List[ObservationSequence],
    ) -> bool:
        """
        Check if time window is available (no conflicts with existing schedule).

        Args:
            start_time: Window start
            end_time: Window end
            existing_schedule: Existing sequences to check against

        Returns:
            True if window is available
        """
        start_dt = start_time.datetime
        end_dt = end_time.datetime

        for seq in existing_schedule:
            seq_start = seq.start
            seq_end = seq.stop

            # Check for overlap
            if start_dt < seq_end and end_dt > seq_start:
                return False

        return True

    def _schedule_moonshine_block(
        self, observations: List[Observation], block_start_time: Time
    ) -> Tuple[List[ObservationSequence], List[Observation]]:
        """
        Schedule Moonshine observations sequentially in the block.

        Args:
            observations: List of Moonshine observations
            block_start_time: When to start the block

        Returns:
            Tuple of (scheduled_sequences, occluded_observations)
        """
        scheduled = []
        occluded = []
        current_time = block_start_time

        # Sort observations for logical ordering (by position, then separation)
        sorted_obs = sorted(
            observations,
            key=lambda o: (o.angular_position_deg, o.limb_separation_deg),
        )

        logger.info(
            f"Scheduling Moonshine block starting at {current_time.iso}"
        )

        for i, obs in enumerate(sorted_obs):
            # Calculate pointing for this observation
            try:
                result = self.moonshine_calc.calculate_pointing(
                    current_time,
                    obs.angular_position_deg,
                    obs.limb_separation_deg,
                    check_earth_blockage=self.config.moonshine_config.check_moon_visibility,
                    earth_avoidance_deg=self.config.moonshine_config.earth_avoidance_deg,
                )

                # Check if Moon is visible (not occluded by Earth)
                if not result.moon_visible:
                    logger.warning(
                        f"Moon occluded for {obs.obs_id} at {current_time.iso}, "
                        f"will schedule individually later"
                    )
                    occluded.append(obs)
                    continue

                # Create scheduled sequence
                duration_minutes = (
                    obs.calculated_duration_minutes or obs.duration
                )
                end_time = current_time + duration_minutes * u.min

                sequence = self._create_moonshine_sequence(
                    obs, current_time, end_time, result
                )
                scheduled.append(sequence)

                logger.debug(
                    f"Scheduled {obs.obs_id}: {current_time.iso} - {end_time.datetime.isoformat()}, "
                    f"RA={result.ra_deg:.2f}°, Dec={result.dec_deg:.2f}°"
                )

                # Move to next observation time (include slew overhead)
                current_time = (
                    end_time  # + self.config.slew_overhead_minutes * u.min
                )

            except Exception as e:
                logger.error(f"Error scheduling {obs.obs_id}: {e}")
                occluded.append(obs)
                continue

        return scheduled, occluded

    def _schedule_occluded_observations(
        self,
        occluded_observations: List[Observation],
        earliest_time: Time,
        end_time: Time,
    ) -> List[ObservationSequence]:
        """
        Schedule occluded Moonshine observations individually at next available time.

        Args:
            occluded_observations: Observations that were occluded
            earliest_time: Earliest time to start searching
            end_time: Latest time to search

        Returns:
            List of scheduled sequences
        """
        scheduled = []
        search_step_minutes = 10.0  # Search every 10 minutes

        # Track the next available time (updates after each scheduled observation)
        next_available_time = earliest_time

        for obs in occluded_observations:
            current_time = (
                next_available_time  # Start from next available time
            )
            found = False

            logger.info(f"Searching for non-occluded time for {obs.obs_id}")

            # Search for next time when Moon is visible
            while current_time <= end_time:
                try:
                    result = self.moonshine_calc.calculate_pointing(
                        current_time,
                        obs.angular_position_deg,
                        obs.limb_separation_deg,
                        check_earth_blockage=True,
                        earth_avoidance_deg=self.config.moonshine_config.earth_avoidance_deg,
                    )

                    if result.moon_visible:
                        # Found a good time
                        duration_minutes = (
                            obs.calculated_duration_minutes or obs.duration
                        )
                        end_obs_time = current_time + duration_minutes * u.min

                        next_available_time = end_obs_time

                        sequence = self._create_moonshine_sequence(
                            obs, current_time, end_obs_time, result
                        )
                        scheduled.append(sequence)

                        logger.info(
                            f"Scheduled occluded {obs.obs_id} at {current_time.iso} "
                            f"(RA={result.ra_deg:.2f}°, Dec={result.dec_deg:.2f}°), "
                            f"next available: {next_available_time.iso}"
                        )
                        found = True
                        break

                except Exception as e:
                    logger.debug(
                        f"Error checking {obs.obs_id} at {current_time.iso}: {e}"
                    )

                current_time += search_step_minutes * u.min

            if not found:
                logger.warning(
                    f"Could not find non-occluded time for {obs.obs_id} "
                    f"within scheduling window"
                )

        return scheduled

    def _create_moonshine_sequence(
        self,
        obs: Observation,
        start_time: Time,
        end_time: Time,
        pointing_result: MoonshineResult,
    ) -> ObservationSequence:
        """
        Create ObservationSequence for a Moonshine observation.

        Args:
            obs: Source Observation
            start_time: Sequence start time
            end_time: Sequence end time
            pointing_result: Calculated pointing from MoonshinePointing

        Returns:
            ObservationSequence object
        """
        from .models import ObservationSequence

        sequence = ObservationSequence(
            obs_id=obs.obs_id,
            sequence_id="000",  # Single sequence per observation
            start=start_time.datetime,
            stop=end_time.datetime,
            parent_observation=obs,
            # Override pointing with calculated values
            boresight_ra=pointing_result.ra_deg,
            boresight_dec=pointing_result.dec_deg,
            target_name=obs.target_name,
            priority=obs.priority,
            # Duration breakdown
            science_duration_minutes=obs.calculated_duration_minutes,
            nir_duration_minutes=obs.nir_duration,
            vis_duration_minutes=obs.visible_duration,
        )

        return sequence


class EarthshineScheduler:
    """
    Schedules Earthshine observations in block mode.

    Block mode schedules all Earthshine observations in a contiguous time block,
    with potential gaps between observations while waiting for the spacecraft
    to reach the next required orbital position.
    """

    def __init__(
        self,
        config: SchedulerConfig,
        ephemeris_provider: EphemerisProvider,
        scheduler_ref=None,
    ):
        """
        Initialize Earthshine block scheduler.

        Args:
            config: Scheduler configuration
            ephemeris_provider: Ephemeris provider
        """
        self.config = config
        self.ephemeris = ephemeris_provider
        self.earthshine_calc = EarthshinePointing(ephemeris_provider)
        self.scheduler_ref = scheduler_ref

    def schedule_block(
        self,
        earthshine_observations: List[Observation],
        start_time: Time,
        end_time: Time,
        existing_schedule: List[ObservationSequence],
        allow_overflow: bool = True,
    ) -> Tuple[List[ObservationSequence], List[Observation]]:
        """
        Schedule all Earthshine observations in a contiguous block.

        The block starts at the earliest available time and continues until
        all observations are scheduled. Gaps between observations (while waiting
        for next orbital position) are filled with CVZ pointings by the main scheduler.

        Args:
            earthshine_observations: List of Earthshine observations to schedule
            start_time: Scheduling window start
            end_time: Scheduling window end
            existing_schedule: Already scheduled sequences
            allow_overflow: Allow scheduling beyond end_time if necessary

        Returns:
            Tuple of (scheduled_sequences, unscheduled_observations)
        """
        if not earthshine_observations:
            return [], []

        es_config = self.config.earthshine_config
        if not es_config or not es_config.enabled:
            return [], earthshine_observations

        logger.info(
            f"Scheduling {len(earthshine_observations)} Earthshine observations in block mode"
        )

        # Estimate block duration (conservative)
        block_duration_minutes = self._estimate_block_duration(
            earthshine_observations
        )
        logger.info(
            f"Earthshine block estimated duration: {block_duration_minutes:.1f} minutes"
        )

        # Find earliest available block start time
        block_start_time = self._find_earthshine_block_window(
            start_time, end_time, block_duration_minutes, existing_schedule
        )

        if block_start_time is None:
            if allow_overflow:
                logger.warning(
                    "Earthshine block doesn't fit in window, trying overflow"
                )
                block_start_time = self._find_earthshine_block_window(
                    end_time,
                    end_time + 7 * u.day,
                    block_duration_minutes,
                    existing_schedule,
                )

        if block_start_time is None:
            logger.error("Could not find suitable window for Earthshine block")
            return [], earthshine_observations

        # Schedule the block
        scheduled_sequences = self._schedule_earthshine_block(
            earthshine_observations, block_start_time, end_time
        )

        # Determine which observations were scheduled
        scheduled_ids = {seq.obs_id for seq in scheduled_sequences}
        unscheduled = [
            obs
            for obs in earthshine_observations
            if obs.obs_id not in scheduled_ids
        ]

        logger.info(
            f"Earthshine block: scheduled {len(scheduled_sequences)} sequences, "
            f"{len(unscheduled)} observations unscheduled"
        )

        return scheduled_sequences, unscheduled

    def _estimate_block_duration(
        self, observations: List[Observation]
    ) -> float:
        """
        Estimate total duration for Earthshine block.

        This is conservative - accounts for orbital period between positions.

        Args:
            observations: List of Earthshine observations

        Returns:
            Estimated duration in minutes
        """
        if not observations:
            return 0.0

        # Get unique orbital positions
        unique_positions = set(
            obs.orbital_position_deg for obs in observations
        )
        n_positions = len(unique_positions)

        # Observations per position
        obs_per_position = (
            len(observations) / n_positions if n_positions > 0 else 1
        )

        # Typical observation duration
        avg_duration = np.mean(
            [
                obs.calculated_duration_minutes or obs.duration or 0.0
                for obs in observations
            ]
        )

        # Estimate: Need roughly one orbit per position, plus observation durations
        orbital_period_minutes = 97.0  # Approximate for LEO

        # Conservative estimate: orbital_period × n_positions + total observation time
        total_obs_time = sum(
            obs.calculated_duration_minutes or obs.duration or 0.0
            for obs in observations
        )

        # Account for slew overhead
        slew_overhead = len(observations) * self.config.slew_overhead_minutes

        # Total: observation time + slews (gaps handled naturally by orbital mechanics)
        estimated_duration = (
            total_obs_time + slew_overhead + (orbital_period_minutes * 0.5)
        )

        logger.debug(
            f"Earthshine block estimate: {total_obs_time:.1f} min obs + "
            f"{slew_overhead:.1f} min slews + buffer = {estimated_duration:.1f} min"
        )

        return estimated_duration

    def _find_earthshine_block_window(
        self,
        start_time: Time,
        end_time: Time,
        duration_minutes: float,
        existing_schedule: List[ObservationSequence],
    ) -> Optional[Time]:
        """
        Find available time window for Earthshine block.

        Args:
            start_time: Search start
            end_time: Search end
            duration_minutes: Required duration
            existing_schedule: Existing sequences to avoid

        Returns:
            Start time for block, or None if not found
        """
        logger.info(
            f"Searching for Earthshine block window: {duration_minutes:.1f} min needed"
        )

        # Search with 30-minute resolution
        search_step_minutes = 30.0
        current_time = start_time

        while current_time <= end_time - duration_minutes * u.min:
            block_end_time = current_time + duration_minutes * u.min

            # Check if window is free
            if self._is_window_available(
                current_time, block_end_time, existing_schedule
            ):
                logger.info(
                    f"Found Earthshine block window: {current_time.iso}"
                )
                return current_time

            current_time += search_step_minutes * u.min

        return None

    def _is_window_available(
        self,
        start_time: Time,
        end_time: Time,
        existing_schedule: List[ObservationSequence],
    ) -> bool:
        """Check if time window is available (no conflicts)."""
        start_dt = start_time.datetime
        end_dt = end_time.datetime

        for seq in existing_schedule:
            # Check for overlap
            if start_dt < seq.stop and end_dt > seq.start:
                return False

        # Use scheduler's blocked time checking method
        if self.scheduler_ref is not None and hasattr(
            self.scheduler_ref, "_is_time_blocked"
        ):
            if self.scheduler_ref._is_time_blocked(start_dt, end_dt):
                logger.debug(f"Window {start_dt} to {end_dt} is blocked")
                return False

        return True

    def _schedule_earthshine_block(
        self,
        observations: List[Observation],
        block_start_time: Time,
        end_time: Time,
    ) -> List[ObservationSequence]:
        """
        Schedule Earthshine observations within the block.

        Observations are scheduled at the next occurrence of their required
        orbital position after the previous observation completes.

        Args:
            observations: List of Earthshine observations
            block_start_time: When to start the block
            end_time: Latest allowed end time

        Returns:
            List of scheduled ObservationSequence objects
        """
        scheduled = []
        current_time = block_start_time

        # Sort observations by orbital position for logical ordering
        sorted_obs = sorted(
            observations,
            key=lambda o: (o.orbital_position_deg, o.limb_separation_deg),
        )

        logger.info(
            f"Scheduling Earthshine block starting at {current_time.iso}"
        )

        for i, obs in enumerate(sorted_obs):
            logger.info(
                f"  Scheduling Earthshine {obs.obs_id} "
                f"(pos={obs.orbital_position_deg}°, sep={obs.limb_separation_deg}°)"
            )

            try:
                # Find next time when spacecraft is at required orbital position
                result = self.earthshine_calc.calculate_pointing(
                    current_time,
                    obs.orbital_position_deg,
                    obs.limb_separation_deg,
                    max_search_orbits=3,  # Search up to 3 orbits ahead
                    position_tolerance_deg=obs.max_orbital_drift_deg,
                )

                # Verify Sun constraints
                if (
                    not result.pointing_in_antisolar
                    or result.sun_angle_deg < 91.0
                ):
                    logger.warning(
                        f"    Sun constraint violated (angle={result.sun_angle_deg:.1f}°), skipping"
                    )
                    continue

                # Calculate schedule times
                schedule_time = result.time
                duration_minutes = (
                    obs.calculated_duration_minutes or obs.duration
                )
                end_obs_time = schedule_time + duration_minutes * u.min

                # Check if we've exceeded the time window
                if end_obs_time.datetime > end_time.datetime:
                    logger.warning(
                        f"    Observation {obs.obs_id} would extend beyond end time, stopping block"
                    )
                    break

                # Check orbital drift
                sc_end = self.ephemeris.get_spacecraft_state(end_obs_time)
                pos_end = self.earthshine_calc._get_orbital_position(sc_end)
                drift = abs(pos_end - result.orbital_position_deg)
                if drift > 180:
                    drift = 360 - drift

                if drift > obs.max_orbital_drift_deg:
                    logger.warning(
                        f"    Orbital drift {drift:.1f}° exceeds max {obs.max_orbital_drift_deg}° "
                        f"(scheduling anyway)"
                    )

                # Create sequence
                sequence = self._create_earthshine_sequence(
                    obs, schedule_time, end_obs_time, result
                )
                scheduled.append(sequence)

                logger.info(
                    f"    ✓ Scheduled at {schedule_time.iso}, "
                    f"RA={result.ra_deg:.2f}°, Dec={result.dec_deg:.2f}°"
                )

                # Update current time for next observation
                current_time = (
                    end_obs_time  # + self.config.slew_overhead_minutes * u.min
                )

            except Exception as e:
                logger.error(f"    Error scheduling {obs.obs_id}: {e}")
                continue

        return scheduled

    def _create_earthshine_sequence(
        self,
        obs: Observation,
        start_time: Time,
        end_time: Time,
        pointing_result: EarthshineResult,
    ) -> ObservationSequence:
        """
        Create ObservationSequence for an Earthshine observation.

        Args:
            obs: Source Observation
            start_time: Sequence start time
            end_time: Sequence end time
            pointing_result: Calculated pointing from EarthshinePointing

        Returns:
            ObservationSequence object
        """
        from .models import ObservationSequence

        sequence = ObservationSequence(
            obs_id=obs.obs_id,
            sequence_id="000",
            start=start_time.datetime,
            stop=end_time.datetime,
            parent_observation=obs,
            # Override pointing with calculated values
            boresight_ra=pointing_result.ra_deg,
            boresight_dec=pointing_result.dec_deg,
            target_name=obs.target_name,
            priority=obs.priority,
            # Duration breakdown
            science_duration_minutes=obs.calculated_duration_minutes,
            nir_duration_minutes=obs.nir_duration,
            vis_duration_minutes=obs.visible_duration,
            raw_xml_tree=obs.raw_xml_tree,
        )

        return sequence
