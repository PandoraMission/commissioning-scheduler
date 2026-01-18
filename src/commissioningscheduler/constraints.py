# constraints.py
"""
Scheduling constraints and validation.

This module handles:
- Data volume budgets
- Slew time and overhead calculations
- Observation splitting rules
- CVZ padding requirements
- Special observation constraints (no-overhead, etc.)
"""

import logging
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

from astropy.time import Time
import astropy.units as u

from .models import (
    Observation,
    ObservationSequence,
    SchedulerConfig,
    ContinuousObservationConstraint,
    EarthshineConfig,
    MoonshineConfig,
)
from .utils import calculate_angular_separation, compute_data_volume_gb

logger = logging.getLogger(__name__)


class ConstraintViolation(Enum):
    """Types of constraint violations."""

    DATA_VOLUME_EXCEEDED = "data_volume_exceeded"
    SLEW_TIME_EXCEEDED = "slew_time_exceeded"
    SEQUENCE_TOO_LONG = "sequence_too_long"
    INSUFFICIENT_VISIBILITY = "insufficient_visibility"
    OVERLAP_CONFLICT = "overlap_conflict"


@dataclass
class ConstraintResult:
    """Result of constraint checking."""

    valid: bool
    violations: List[ConstraintViolation] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)

    def add_violation(self, violation: ConstraintViolation, message: str):
        """Add a constraint violation."""
        self.valid = False
        self.violations.append(violation)
        self.messages.append(message)


@dataclass
class DataBudget:
    """Tracks data volume budget."""

    max_volume_gb: float
    used_volume_gb: float = 0.0

    @property
    def remaining_gb(self) -> float:
        """Remaining data budget."""
        return max(0.0, self.max_volume_gb - self.used_volume_gb)

    @property
    def percent_used(self) -> float:
        """Percentage of budget used."""
        if self.max_volume_gb <= 0:
            return 0.0
        return (self.used_volume_gb / self.max_volume_gb) * 100.0

    def can_accommodate(self, volume_gb: float) -> bool:
        """Check if budget can accommodate additional volume."""
        return self.used_volume_gb + volume_gb <= self.max_volume_gb

    def add(self, volume_gb: float) -> bool:
        """
        Add to used volume if within budget.

        Returns:
            True if added successfully, False if would exceed budget
        """
        if not self.can_accommodate(volume_gb):
            return False
        self.used_volume_gb += volume_gb
        return True


class ConstraintChecker:
    """
    Validates scheduling constraints.
    """

    def __init__(self, config: SchedulerConfig):
        """
        Initialize constraint checker.

        Args:
            config: Scheduler configuration
        """
        self.config = config
        self.data_budget = DataBudget(max_volume_gb=config.max_data_volume_gb)
        self.continuous_constraints: List[ContinuousObservationConstraint] = []

    def load_constraints(self, constraints_json: str):
        """Load constraints from JSON file."""
        with open(constraints_json) as f:
            data = json.load(f)

        # Load continuous observation constraints
        if "continuous_observations" in data:
            cont_data = data["continuous_observations"]
            self.continuous_constraints.append(
                ContinuousObservationConstraint(
                    tasks=cont_data.get("tasks", []),
                    observations=cont_data.get("observations", []),
                    description=cont_data.get("description", ""),
                )
            )

        # Load shine constraints
        self._load_shine_constraints(data)

    def requires_continuous_scheduling(self, obs: Observation) -> bool:
        """Check if observation must be scheduled continuously."""
        return any(c.applies_to(obs) for c in self.continuous_constraints)

    def _load_shine_constraints(self, data: Dict[str, Any]):
        """
        Load Earthshine and Moonshine configuration from constraints JSON.

        Args:
            data: Parsed JSON data from constraints file
        """
        from .utils import parse_utc_time  # Assuming this utility exists

        # Load Earthshine config
        if "earthshine" in data:
            es_data = data["earthshine"]
            self.config.earthshine_config = EarthshineConfig(
                enabled=es_data.get("enabled", False),
                orbital_positions=es_data.get(
                    "orbital_positions", [0, 90, 180, 270]
                ),
                limb_separations=es_data.get(
                    "limb_separations", [5, 10, 15, 20]
                ),
                max_orbital_drift_deg=es_data.get(
                    "max_orbital_drift_deg", 30.0
                ),
                scheduling_mode=es_data.get("scheduling_mode", "flexible"),
                block_priority=es_data.get("block_priority", "medium"),
            )
            logger.info(
                f"Earthshine enabled: {len(es_data.get('orbital_positions', []))} positions, "
                f"{len(es_data.get('limb_separations', []))} separations"
            )

        # Load Moonshine config
        if "moonshine" in data:
            ms_data = data["moonshine"]
            target_date_str = ms_data.get("target_date", "2026-02-01 00:00:00")
            target_date = parse_utc_time(target_date_str)

            # Load combination filters
            combinations = ms_data.get("combinations", None)
            exclude_combinations = ms_data.get("exclude_combinations", None)

            self.config.moonshine_config = MoonshineConfig(
                enabled=ms_data.get("enabled", False),
                target_date=target_date,
                window_days=ms_data.get("window_days", 3.0),
                angular_positions=ms_data.get(
                    "angular_positions", [0, 45, 90, 135, 180, 225, 270, 315]
                ),
                limb_separations=ms_data.get(
                    "limb_separations", [5, 10, 15, 20]
                ),
                check_moon_visibility=ms_data.get(
                    "check_moon_visibility", True
                ),
                earth_avoidance_deg=ms_data.get("earth_avoidance_deg", 20.0),
                combinations=combinations,
                exclude_combinations=exclude_combinations,
            )
            # Log configuration details
            if combinations:
                logger.info(
                    f"Moonshine enabled (whitelist mode): {len(combinations)} specific combinations"
                )
            elif exclude_combinations:
                n_total = len(ms_data.get("angular_positions", [])) * len(
                    ms_data.get("limb_separations", [])
                )
                n_excluded = len(exclude_combinations)
                logger.info(
                    f"Moonshine enabled (blacklist mode): {n_total - n_excluded} combinations "
                    f"({n_excluded} excluded)"
                )
            else:
                logger.info(
                    f"Moonshine enabled: target={target_date.isoformat()}, "
                    f"{len(ms_data.get('angular_positions', []))} positions, "
                    f"{len(ms_data.get('limb_separations', []))} separations"
                )

    def check_observation_schedulable(
        self, obs: Observation, current_time: datetime
    ) -> ConstraintResult:
        """
        Check if an observation can be scheduled.

        Args:
            obs: Observation to check
            current_time: Current scheduling time

        Returns:
            ConstraintResult
        """
        result = ConstraintResult(valid=True)

        # Check if coordinates are valid
        if obs.boresight_ra is None or obs.boresight_dec is None:
            # Skip this check for Earthshine/Moonshine - they get coordinates during special scheduling
            if getattr(obs, "is_earthshine", False) or getattr(
                obs, "is_moonshine", False
            ):
                # These observations will get coordinates during special scheduling
                logger.debug(
                    f"Skipping coordinate check for {obs.obs_id} (Earthshine/Moonshine)"
                )
            else:
                result.add_violation(
                    ConstraintViolation.INSUFFICIENT_VISIBILITY,
                    f"Observation {obs.obs_id} missing coordinates",
                )
                return result

        # Check if visibility windows exist
        if not obs.visibility_windows:
            result.add_violation(
                ConstraintViolation.INSUFFICIENT_VISIBILITY,
                f"Observation {obs.obs_id} has no visibility windows",
            )
            return result

        # Check if duration is computable (removed minimum duration check)
        if obs.duration is None:
            result.add_violation(
                ConstraintViolation.SEQUENCE_TOO_LONG,
                f"Observation {obs.obs_id} has no valid duration",
            )
            return result

        # Allow observations with any positive duration (even if very short)
        if obs.duration <= 0:
            result.add_violation(
                ConstraintViolation.SEQUENCE_TOO_LONG,
                f"Observation {obs.obs_id} has non-positive duration: {obs.duration}",
            )
            return result

        # Compute data volume
        if obs.nir_duration or obs.visible_duration:
            volume = compute_data_volume_gb(
                nir_minutes=obs.nir_duration or 0.0,
                vis_minutes=obs.visible_duration or 0.0,
                nir_rate_mbps=self.config.nir_data_rate_mbps,
                vis_rate_mbps=self.config.vis_data_rate_mbps,
            )

            if not self.data_budget.can_accommodate(volume):
                result.add_violation(
                    ConstraintViolation.DATA_VOLUME_EXCEEDED,
                    f"Observation {obs.obs_id} would exceed data budget "
                    f"({volume:.2f} GB needed, {self.data_budget.remaining_gb:.2f} GB remaining)",
                )

        return result

    def check_sequence_valid(
        self,
        seq: ObservationSequence,
        previous_seq: Optional[ObservationSequence] = None,
    ) -> ConstraintResult:
        """
        Check if an observation sequence is valid.

        Args:
            seq: Observation sequence to check
            previous_seq: Previous sequence (for slew calculation)

        Returns:
            ConstraintResult
        """
        result = ConstraintResult(valid=True)

        # Check duration doesn't exceed maximum
        if seq.duration_minutes > self.config.max_sequence_duration_minutes:
            result.add_violation(
                ConstraintViolation.SEQUENCE_TOO_LONG,
                f"Sequence {seq.obs_id}_{seq.sequence_id} duration "
                f"({seq.duration_minutes:.1f} min) exceeds maximum "
                f"({self.config.max_sequence_duration_minutes:.1f} min)",
            )

        # Check slew time if there's a previous sequence
        if previous_seq is not None:
            slew_time = self.compute_slew_time(
                previous_seq.boresight_ra or 0.0,
                previous_seq.boresight_dec or 0.0,
                seq.boresight_ra or 0.0,
                seq.boresight_dec or 0.0,
            )

            gap_minutes = (
                seq.start - previous_seq.stop
            ).total_seconds() / 60.0
            required_time = slew_time + self.config.setup_overhead_minutes

            if gap_minutes < required_time:
                result.add_violation(
                    ConstraintViolation.SLEW_TIME_EXCEEDED,
                    f"Insufficient time for slew: {gap_minutes:.1f} min available, "
                    f"{required_time:.1f} min required",
                )

        return result

    def compute_slew_time(
        self, from_ra: float, from_dec: float, to_ra: float, to_dec: float
    ) -> float:
        """
        Compute slew time in minutes.

        Args:
            from_ra: Starting RA (degrees)
            from_dec: Starting DEC (degrees)
            to_ra: Ending RA (degrees)
            to_dec: Ending DEC (degrees)

        Returns:
            Slew time in minutes
        """
        separation = calculate_angular_separation(
            from_ra, from_dec, to_ra, to_dec
        )
        slew_time = separation / self.config.slew_rate_deg_per_min
        return slew_time

    def compute_total_overhead(
        self, from_ra: float, from_dec: float, to_ra: float, to_dec: float
    ) -> float:
        """
        Compute total overhead time (slew + setup) in minutes.

        Args:
            from_ra: Starting RA (degrees)
            from_dec: Starting DEC (degrees)
            to_ra: Ending RA (degrees)
            to_dec: Ending DEC (degrees)

        Returns:
            Total overhead time in minutes
        """
        slew_time = self.compute_slew_time(from_ra, from_dec, to_ra, to_dec)
        return slew_time + self.config.setup_overhead_minutes

    def split_observation_into_sequences(
        self,
        obs: Observation,
        visibility_windows: List[Tuple[datetime, datetime]],
        start_from: datetime,
        needs_initial_overhead: bool = True,
        max_gap_between_sequences_hours: float = 72.0,
    ) -> List[Tuple[datetime, datetime, float]]:
        """
        Split observation into sequences fitting visibility windows.

        Returns (start, window_end, science_duration) where start already accounts
        for overhead positioning.

        Args:
            obs: Observation to split
            visibility_windows: Available visibility windows
            start_from: Earliest start time
            needs_initial_overhead: Whether first sequence needs 1-min overhead
            max_gap_between_sequences_hours: Maximum gap between sequences

        Returns:
            List of (start, window_end, science_duration_minutes) tuples
        """
        # Check if observation must be continuous
        if self.requires_continuous_scheduling(obs):
            logger.info(
                f"    Observation {obs.obs_id} requires continuous scheduling"
            )

            # Find a window that can fit the entire observation + overhead
            required_minutes = obs.duration
            overhead_minutes = (
                self.config.slew_overhead_minutes
                if needs_initial_overhead
                else 0.0
            )

            if overhead_minutes > 0:
                required_minutes += overhead_minutes

            for win_start, win_end in visibility_windows:
                if win_start < start_from:
                    win_start = start_from

                if win_start >= win_end:
                    continue

                available_minutes = (
                    win_end - win_start
                ).total_seconds() / 60.0

                if available_minutes >= required_minutes:
                    # Return single sequence with just the science duration
                    # Overhead will be added by the caller
                    logger.info(
                        f"    Found continuous window: {win_start} to {win_end} "
                        f"({available_minutes:.1f} min available, {required_minutes:.1f} min needed)"
                    )
                    return [(win_start, win_end, obs.duration)]

            logger.warning(
                f"    No single window can accommodate {obs.obs_id} continuously "
                f"(needs {required_minutes:.1f} minutes)"
            )
            return []

        logger.debug(
            f"    Splitting {obs.obs_id}: duration={obs.duration:.4f} min, needs_overhead={needs_initial_overhead}"
        )

        sequences = []
        remaining_minutes = obs.duration
        overhead_minutes = (
            self.config.slew_overhead_minutes
            if needs_initial_overhead
            else 0.0
        )
        overhead_to_apply = (
            overhead_minutes  # Track if overhead still needs to be applied
        )

        logger.debug(f"    Overhead to reserve: {overhead_minutes:.4f} min")

        if not obs.can_be_split:
            for win_start, win_end in visibility_windows:
                if win_start < start_from:
                    win_start = start_from

                available_minutes = (
                    win_end - win_start
                ).total_seconds() / 60.0

                if available_minutes >= (obs.duration + overhead_minutes):
                    return [(win_start, win_end, obs.duration)]

            logger.warning(f"{obs.obs_id} cannot be split and doesn't fit")
            return []

        last_seq_end = start_from

        for win_start, win_end in visibility_windows:
            if remaining_minutes <= 0:
                break

            if win_start < start_from:
                win_start = start_from

            if win_start >= win_end:
                continue

            # Check gap
            if last_seq_end > start_from:
                gap_hours = (win_start - last_seq_end).total_seconds() / 3600.0
                if gap_hours > max_gap_between_sequences_hours:
                    logger.warning(
                        f"{obs.obs_id}: Gap too large ({gap_hours:.1f}h)"
                    )
                    break

            # Pack this window
            current_pos = win_start

            logger.debug(f"    Window: {win_start} to {win_end}")

            while remaining_minutes > 0 and current_pos < win_end:
                available = (win_end - current_pos).total_seconds() / 60.0
                logger.debug(
                    f"      Available at {current_pos}: {available:.4f} min"
                )

                # Account for overhead on first sequence
                if overhead_to_apply > 0:
                    if available <= overhead_to_apply:
                        break  # Not enough room
                    # Reserve space for overhead but don't include in science duration
                    available_for_science = available - overhead_to_apply
                else:
                    available_for_science = available

                # Determine science duration
                seq_science = min(
                    remaining_minutes,
                    available_for_science,
                    self.config.max_sequence_duration_minutes,
                )

                if seq_science < 0.1:
                    break

                logger.debug(
                    f"      Returning sequence: science={seq_science:.4f} min at {current_pos}"
                )
                # Add sequence
                sequences.append((current_pos, win_end, seq_science))
                remaining_minutes -= seq_science

                # Advance position by science time PLUS overhead (if this was first sequence)
                if overhead_to_apply > 0:
                    current_pos += timedelta(
                        minutes=seq_science + overhead_to_apply
                    )
                    overhead_to_apply = (
                        0.0  # Only first sequence gets overhead
                    )
                else:
                    current_pos += timedelta(minutes=seq_science)

                last_seq_end = current_pos

        if remaining_minutes > 0.01:
            logger.warning(
                f"{obs.obs_id}: {remaining_minutes:.1f} min could not be scheduled"
            )

        logger.debug(
            f"    Constraint checker returning {len(sequences)} sequences"
        )

        logger.debug(f"    Final sequences being returned:")
        for idx, (s, e, sci) in enumerate(sequences):
            logger.debug(
                f"      Seq {idx+1}: start={s}, science={sci:.4f} min"
            )
        return sequences

    def add_sequence_to_budget(self, seq: ObservationSequence) -> bool:
        """
        Add sequence to data budget.

        Args:
            seq: ObservationSequence to add

        Returns:
            True if added successfully, False if would exceed budget
        """
        volume = compute_data_volume_gb(
            nir_minutes=seq.nir_duration_minutes or 0.0,
            vis_minutes=seq.vis_duration_minutes or 0.0,
            nir_rate_mbps=self.config.nir_data_rate_mbps,
            vis_rate_mbps=self.config.vis_data_rate_mbps,
        )

        if self.data_budget.add(volume):
            logger.debug(
                f"Added {volume:.2f} GB for sequence {seq.obs_id}_{seq.sequence_id}, "
                f"budget: {self.data_budget.percent_used:.1f}% used"
            )
            return True
        else:
            logger.warning(
                f"Cannot add sequence {seq.obs_id}_{seq.sequence_id}: "
                f"would exceed data budget"
            )
            return False

    def reset_budget(self):
        """Reset data budget to zero."""
        self.data_budget.used_volume_gb = 0.0
        logger.info("Reset data budget")


@dataclass
class SpecialConstraint:
    """Special constraint for specific observations."""

    obs_id: str
    constraint_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)


class SpecialConstraintHandler:
    """
    Handles special constraints for specific observations.

    Examples:
    - No overhead constraint (for certain calibration observations)
    - Custom keep-out angles (for Earth/Moon pointing tasks)
    - Fixed duration requirements
    """

    def __init__(self):
        """Initialize handler."""
        self.constraints: Dict[str, List[SpecialConstraint]] = {}

    def add_constraint(self, obs_id: str, constraint_type: str, **parameters):
        """
        Add a special constraint for an observation.

        Args:
            obs_id: Observation ID
            constraint_type: Type of constraint (e.g., "no_overhead")
            **parameters: Additional parameters for the constraint
        """
        constraint = SpecialConstraint(
            obs_id=obs_id,
            constraint_type=constraint_type,
            parameters=parameters,
        )

        if obs_id not in self.constraints:
            self.constraints[obs_id] = []

        self.constraints[obs_id].append(constraint)
        logger.info(f"Added {constraint_type} constraint for {obs_id}")

    def get_constraints(self, obs_id: str) -> List[SpecialConstraint]:
        """Get all constraints for an observation."""
        return self.constraints.get(obs_id, [])

    def has_constraint_type(self, obs_id: str, constraint_type: str) -> bool:
        """Check if observation has a specific constraint type."""
        constraints = self.get_constraints(obs_id)
        return any(c.constraint_type == constraint_type for c in constraints)

    def requires_no_overhead(self, obs_id: str) -> bool:
        """Check if observation requires no overhead time."""
        return self.has_constraint_type(obs_id, "no_overhead")

    def get_custom_keep_out_angles(
        self, obs_id: str
    ) -> Optional[Tuple[float, float, float]]:
        """
        Get custom keep-out angles if specified.

        Returns:
            Tuple of (sun, earth, moon) angles in degrees, or None
        """
        constraints = self.get_constraints(obs_id)
        for c in constraints:
            if c.constraint_type == "custom_keep_out":
                return (
                    c.parameters.get("sun", None),
                    c.parameters.get("earth", None),
                    c.parameters.get("moon", None),
                )
        return None


class Task0312Handler:
    """
    Special handler for task 0312 observation pattern.

    Task 0312 follows a specific pattern:
    - 5 min data, 10 min staring, 5 min data, 10 min staring, 5 min data, 10 min staring
    - 5 min data, 5 min data (consecutive), 10 min staring
    - 5 min data, 10 min staring, 5 min data, 10 min staring, 5 min data (final)

    Total: 8x 5-min data + 6x 10-min staring = 40 + 60 = 100 minutes
    Plus 1 minute overhead on first sequence = 101 minutes total
    """

    # Define the pattern (duration_minutes, sequence_type)
    SEQUENCE_PATTERN = [
        (5, "data"),  # 1
        (10, "staring"),  # 2
        (5, "data"),  # 3
        (10, "staring"),  # 4
        (5, "data"),  # 5
        (10, "staring"),  # 6
        (5, "data"),  # 7
        (5, "data"),  # 8 - consecutive data
        (10, "staring"),  # 9
        (5, "data"),  # 10
        (10, "staring"),  # 11
        (5, "data"),  # 12
        (10, "staring"),  # 13
        (5, "data"),  # 14 - final
    ]

    @staticmethod
    def is_task_0312(obs: Observation) -> bool:
        """Check if observation is 0312_000."""
        return obs.obs_id == "0312_000"

    @staticmethod
    def create_sequences(
        obs: Observation, start_time: datetime, needs_overhead: bool = True
    ) -> List[Tuple[datetime, datetime, float, str]]:
        """
        Create the special sequence pattern for task 0312.

        Args:
            obs: Observation object for 0312_000
            start_time: When to start the observation
            needs_overhead: Whether to add 1 minute overhead to first sequence

        Returns:
            List of (start, stop, science_duration, sequence_type) tuples
        """
        sequences = []
        current_time = start_time

        # Add overhead to first sequence if needed
        if needs_overhead:
            overhead_minutes = 1.0
        else:
            overhead_minutes = 0.0

        for idx, (duration_minutes, seq_type) in enumerate(
            Task0312Handler.SEQUENCE_PATTERN
        ):
            # First sequence includes overhead
            if idx == 0 and overhead_minutes > 0:
                total_duration = duration_minutes + overhead_minutes
            else:
                total_duration = duration_minutes

            seq_start = current_time
            seq_end = current_time + timedelta(minutes=total_duration)

            sequences.append((seq_start, seq_end, duration_minutes, seq_type))

            current_time = seq_end

        return sequences

    @staticmethod
    def calculate_scaled_parameters(
        obs: Observation, scale_factor: float
    ) -> Dict[str, int]:
        """
        Calculate scaled camera parameters for 5-minute data sequences.

        Args:
            obs: Original observation
            scale_factor: Scaling factor (0.05 for 5 minutes out of 100)

        Returns:
            Dictionary with scaled parameters
        """
        scaled_params = {}

        # Scale visible camera parameters
        if obs.num_total_frames_requested:
            scaled_frames = int(obs.num_total_frames_requested * scale_factor)
            # Ensure it's a multiple of FramesPerCoadd if specified
            if obs.frames_per_coadd and obs.frames_per_coadd > 0:
                scaled_frames = (
                    scaled_frames // obs.frames_per_coadd
                ) * obs.frames_per_coadd
            scaled_params["NumTotalFramesRequested"] = max(1, scaled_frames)

        # Scale NIR camera parameters
        if obs.sc_integrations:
            scaled_integrations = int(obs.sc_integrations * scale_factor)
            scaled_params["SC_Integrations"] = max(1, scaled_integrations)

        return scaled_params
