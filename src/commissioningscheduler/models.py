# models.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
import xml.etree.ElementTree as ET
from enum import Enum

from astropy.time import Time

# Forward reference for ValidationIssue - will be defined in diagnostics.py
# We use string annotation to avoid circular import
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from diagnostics import ValidationIssue


@dataclass
class Observation:
    """
    Represents an observation request from an input XML file.
    May require multiple ObservationSequences to fulfill if duration exceeds visibility window.
    """
    obs_id: str  # e.g., "0360_004"
    task_number: Optional[str] = None  # e.g., "0360"
    
    # Target information
    target_name: Optional[str] = None
    priority: Optional[float] = None
    
    # Pointing
    boresight_ra: Optional[float] = None
    boresight_dec: Optional[float] = None
    
    # Visible camera parameters
    num_total_frames_requested: Optional[int] = None
    exposure_time_us: Optional[float] = None
    visible_duration: Optional[float] = None  # minutes
    frames_per_coadd: Optional[int] = None
    
    # NIR camera parameters
    sc_integrations: Optional[int] = None
    sc_resets1: Optional[int] = None
    sc_resets2: Optional[int] = None
    sc_dropframes1: Optional[int] = None
    sc_dropframes2: Optional[int] = None
    sc_dropframes3: Optional[int] = None
    sc_readframes: Optional[int] = None
    roi_sizex: Optional[int] = None
    roi_sizey: Optional[int] = None
    nir_duration: Optional[float] = None  # minutes
    
    # Total duration
    duration: Optional[float] = None  # minutes
    
    # Scheduling properties
    can_be_split: bool = True
    visibility_windows: List[Tuple[float, float]] = field(default_factory=list)
    
    # Store the raw XML tree for cloning observation sequences
    raw_xml_tree: Optional[ET.Element] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and compute derived values."""
        # Validate coordinates if provided
        if self.boresight_ra is not None and not (0 <= self.boresight_ra <= 360):
            raise ValueError(f"RA must be between 0 and 360, got {self.boresight_ra}")
        if self.boresight_dec is not None and not (-90 <= self.boresight_dec <= 90):
            raise ValueError(f"DEC must be between -90 and 90, got {self.boresight_dec}")
        
        # Validate priority if provided
        if self.priority is not None and not (0 <= self.priority <= 10):
            raise ValueError(f"Priority must be between 0 and 10, got {self.priority}")
        
        # Calculate total duration if not provided
        if self.duration is None:
            nir = self.nir_duration or 0.0
            vis = self.visible_duration or 0.0
            if nir > 0 or vis > 0:
                self.duration = max(nir, vis)  # Cameras run in parallel

@dataclass
class ObservationSequence:
    """
    Represents a single scheduled observation sequence in the output.
    Multiple sequences may be needed to fulfill one Observation request.
    """
    obs_id: str
    sequence_id: str
    start: datetime
    stop: datetime
    
    # Reference to parent observation for copying parameters
    parent_observation: Optional[Observation] = None
    
    # Override parameters (if different from parent)
    target_name: Optional[str] = None
    boresight_ra: Optional[float] = None
    boresight_dec: Optional[float] = None
    priority: Optional[float] = None
    
    # Actual durations for this sequence (may be less than parent if split)
    science_duration_minutes: Optional[float] = None
    nir_duration_minutes: Optional[float] = None
    vis_duration_minutes: Optional[float] = None
    
    # Tracking for split observations
    remaining_frames: Optional[int] = None
    remaining_integrations: Optional[int] = None
    
    # Store raw XML for this specific sequence
    raw_xml_tree: Optional[ET.Element] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_minutes(self) -> float:
        """Total duration including overhead."""
        if self.start and self.stop:
            return (self.stop - self.start).total_seconds() / 60.0
        return 0.0
    
    @property
    def visit_id(self) -> str:
        """Extract visit ID from obs_id."""
        return self.obs_id.split('_')[0] if '_' in self.obs_id else self.obs_id
    
    def __post_init__(self):
        """Validate sequence."""
        if self.start >= self.stop:
            raise ValueError(f"Start time must be before stop time")

@dataclass
class Visit:
    """
    Represents a Visit containing one or more ObservationSequences.
    All sequences in a visit support the same observation task.
    """
    visit_id: str
    observation_sequences: List[ObservationSequence] = field(default_factory=list)
    parent_observation: Optional[Observation] = None
    
    @property
    def total_duration_minutes(self) -> float:
        """Total duration of all sequences in this visit."""
        return sum(seq.duration_minutes for seq in self.observation_sequences)
    
    @property
    def total_science_minutes(self) -> float:
        """Total science time (excluding overhead)."""
        return sum(seq.science_duration_minutes or 0.0 for seq in self.observation_sequences)
    
    @property
    def start_time(self) -> Optional[datetime]:
        """Start time of the first sequence."""
        if self.observation_sequences:
            return min(seq.start for seq in self.observation_sequences)
        return None
    
    @property
    def end_time(self) -> Optional[datetime]:
        """End time of the last sequence."""
        if self.observation_sequences:
            return max(seq.stop for seq in self.observation_sequences)
        return None

@dataclass
class SchedulerConfig:
    """Configuration for the scheduler."""
    # Orbital parameters
    tle_line1: str
    tle_line2: str
    
    # Timing
    commissioning_start: datetime
    commissioning_end: datetime
    
    # Pointing
    cvz_coords: Tuple[float, float]  # (RA, DEC)
    
    # Data rates (configurable)
    nir_data_rate_mbps: float = 2.74
    vis_data_rate_mbps: float = 0.88
    downlink_rate_bps: float = 4e6
    max_data_volume_gb: float = 150.0
    
    # Constraints
    keep_out_angles: Tuple[float, float, float] = (90.0, 25.0, 63.0)  # sun, earth, moon
    max_sequence_duration_minutes: float = 90.0  # Maximum continuous observation time
    slew_rate_deg_per_min: float = 1.0
    
    # Overhead times (minutes)
    slew_overhead_minutes: float = 1.0
    setup_overhead_minutes: float = 1.0
    
    # Optional files
    constraints_json: Optional[str] = None
    dependency_json: Optional[str] = None
    progress_json: Optional[str] = None
    extra_cvz_json: Optional[str] = None
    pointing_ephem_file: Optional[str] = None
    
    # Behavior flags
    use_visibility_cache: bool = True
    cache_dir: Optional[str] = None  # If None, will use default hidden directory
    keep_raw_xml_trees: bool = True  # Keep XML trees for cloning
    verify_cvz_visibility: bool = False  # If False, skip CVZ visibility check (faster)
    enable_gap_filling: bool = True  # If True, try to schedule science observations in gaps
    
    def __post_init__(self):
        """Validate configuration and set up cache directory."""
        if self.commissioning_start >= self.commissioning_end:
            raise ValueError("commissioning_start must be before commissioning_end")
        if len(self.keep_out_angles) != 3:
            raise ValueError("keep_out_angles must have 3 values (sun, earth, moon)")
        if self.max_data_volume_gb <= 0:
            raise ValueError("max_data_volume_gb must be positive")
        if self.max_sequence_duration_minutes <= 0:
            raise ValueError("max_sequence_duration_minutes must be positive")
        
        # Set up cache directory if not specified
        if self.cache_dir is None:
            import os
            # Use user's home directory for cache
            home = os.path.expanduser("~")
            self.cache_dir = os.path.join(home, ".pandora_scheduler_cache")
        
        # Create cache directory if it doesn't exist
        if self.use_visibility_cache:
            os.makedirs(self.cache_dir, exist_ok=True)


@dataclass
class SchedulingResult:
    """Result of a scheduling operation."""
    success: bool
    message: str
    visits: List[Visit] = field(default_factory=list)
    scheduled_sequences: List[ObservationSequence] = field(default_factory=list)  # ADD THIS
    total_duration_minutes: float = 0.0
    total_science_minutes: float = 0.0
    total_data_volume_gb: float = 0.0
    warnings: List[str] = field(default_factory=list)
    unscheduled_observations: List[Observation] = field(default_factory=list)
    
    @property
    def total_sequences(self) -> int:
        """Total number of observation sequences across all visits."""
        return sum(len(visit.observation_sequences) for visit in self.visits)
    
    @property
    def scheduling_efficiency(self) -> float:
        """Ratio of science time to total time."""
        if self.total_duration_minutes > 0:
            return self.total_science_minutes / self.total_duration_minutes
        return 0.0

@dataclass
class ScheduleDiagnostics:
    """Complete diagnostics for a schedule."""
    # Timing
    schedule_start: datetime
    schedule_end: datetime
    total_duration_minutes: float
    total_science_minutes: float
    total_overhead_minutes: float
    
    # Efficiency
    scheduling_efficiency: float  # science_time / total_time
    visibility_utilization: float  # scheduled_time / available_visibility
    
    # Per-task metrics
    task_durations: Dict[str, float]  # task_id -> total minutes
    task_science_durations: Dict[str, float]  # task_id -> science minutes only
    task_observation_counts: Dict[str, int]  # task_id -> count
    task_sequence_counts: Dict[str, int]  # task_id -> count
    
    # Per-observation metrics
    observation_info: Dict[str, Dict]  # obs_id -> info dict
    
    # Data volume
    estimated_data_volume_gb: float
    data_volume_by_task: Dict[str, float]
    
    # CVZ metrics
    cvz_time_minutes: float
    cvz_sequence_count: int
    
    # Validation
    validation_issues: List[Any] = field(default_factory=list)
    
    # Timeline
    timeline: List[Dict] = field(default_factory=list)  # Chronological sequence list
    
    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(issue.severity == "error" for issue in self.validation_issues)
    
    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(issue.severity == "warning" for issue in self.validation_issues)


@dataclass
class BlockedTimeWindow:
    """Represents a blocked time window in the schedule."""
    window_type: str  # "downlink", "analysis_buffer", etc.
    start_time: datetime
    end_time: datetime
    description: str = ""
    trigger_obs_id: Optional[str] = None  # If triggered by an observation
    trigger_task_id: Optional[str] = None  # If triggered by a task
    
    @property
    def duration_minutes(self) -> float:
        return (self.end_time - self.start_time).total_seconds() / 60.0


@dataclass
class BlockedTimeConstraint:
    """Definition of a blocked time constraint from constraints file."""
    constraint_type: str  # "after_observation", "after_task", "fixed"
    window_type: str  # "downlink", "analysis_buffer", etc.
    description: str = ""
    
    # For triggered constraints
    observation_id: Optional[str] = None
    task_id: Optional[str] = None
    duration_minutes: Optional[float] = None
    
    # For fixed constraints
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate constraint definition."""
        if self.constraint_type == "fixed":
            if self.start is None or self.end is None:
                raise ValueError("Fixed blocked time requires start and end times")
        elif self.constraint_type == "after_observation":
            if self.observation_id is None or self.duration_minutes is None:
                raise ValueError("after_observation requires observation_id and duration_minutes")
        elif self.constraint_type == "after_task":
            if self.task_id is None or self.duration_minutes is None:
                raise ValueError("after_task requires task_id and duration_minutes")


class Task0312SequenceType(Enum):
    """Types of sequences for task 0312."""
    DATA_COLLECTION = "data"
    STARING = "staring"


@dataclass
class ContinuousObservationConstraint:
    """Constraint requiring observations to be scheduled continuously."""
    tasks: List[str] = field(default_factory=list)  # Task numbers
    observations: List[str] = field(default_factory=list)  # Specific obs IDs
    description: str = ""

    def applies_to(self, obs: Observation) -> bool:
        """Check if constraint applies to observation."""
        return (
            (obs.task_number in self.tasks) or
            (obs.obs_id in self.observations)
        )