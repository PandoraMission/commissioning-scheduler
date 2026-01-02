# diagnostics.py
"""
Schedule diagnostics and validation.

This module provides:
- Schedule validation (dependencies, overlaps, visibility)
- Performance metrics (efficiency, utilization)
- Detailed reporting (per-task, per-observation)
- Timeline analysis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict
import xml.etree.ElementTree as ET

from .models import (
    Observation,
    ObservationSequence,
    Visit,
    ContinuousObservationConstraint,
)
from .utils import calculate_angular_separation, compute_data_volume_gb

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """A validation issue found in the schedule."""

    severity: str  # "error", "warning", "info"
    category: str  # "dependency", "overlap", "visibility", "data_volume", etc.
    message: str
    details: Dict = field(default_factory=dict)


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
    task_science_durations: Dict[str, float] = field(
        default_factory=dict
    )  # task_id -> science minutes only
    task_observation_counts: Dict[str, int] = field(
        default_factory=dict
    )  # task_id -> count
    task_sequence_counts: Dict[str, int] = field(
        default_factory=dict
    )  # task_id -> count

    # Per-observation metrics
    observation_info: Dict[str, Dict] = field(
        default_factory=dict
    )  # obs_id -> info dict

    # Data volume
    estimated_data_volume_gb: float = 0.0
    data_volume_by_task: Dict[str, float] = field(default_factory=dict)

    # CVZ metrics
    cvz_time_minutes: float = 0.0
    cvz_sequence_count: int = 0

    # Validation - using Any to avoid circular import
    validation_issues: List[Any] = field(default_factory=list)

    # Timeline
    timeline: List[Dict] = field(
        default_factory=list
    )  # Chronological sequence list

    @property
    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(
            hasattr(issue, "severity") and issue.severity == "error"
            for issue in self.validation_issues
        )

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(
            hasattr(issue, "severity") and issue.severity == "warning"
            for issue in self.validation_issues
        )


class DiagnosticsAnalyzer:
    """Analyzes and validates a schedule."""

    def __init__(
        self,
        nir_data_rate_mbps: float = 2.74,
        vis_data_rate_mbps: float = 0.88,
        max_data_volume_gb: float = 150.0,
    ):
        """
        Initialize analyzer.

        Args:
            nir_data_rate_mbps: NIR camera data rate
            vis_data_rate_mbps: Visible camera data rate
            max_data_volume_gb: Maximum data volume budget
        """
        self.nir_rate = nir_data_rate_mbps
        self.vis_rate = vis_data_rate_mbps
        self.max_data_volume = max_data_volume_gb

    def analyze_schedule(
        self,
        visits: List[Visit],
        input_observations: Optional[List[Observation]] = None,
        dependency_json: Optional[str] = None,
    ) -> ScheduleDiagnostics:
        """
        Perform complete analysis of a schedule.

        Args:
            visits: List of scheduled visits
            input_observations: Original observation requests (for comparison)
            dependency_json: Path to dependency file (for validation)

        Returns:
            ScheduleDiagnostics
        """
        logger.info("Analyzing schedule...")

        # Extract all sequences
        all_sequences = []
        for visit in visits:
            all_sequences.extend(visit.observation_sequences)

        if not all_sequences:
            logger.warning("No sequences to analyze")
            return ScheduleDiagnostics(
                schedule_start=datetime.now(),
                schedule_end=datetime.now(),
                total_duration_minutes=0.0,
                total_science_minutes=0.0,
                total_overhead_minutes=0.0,
                scheduling_efficiency=0.0,
                visibility_utilization=0.0,
                task_durations={},
                task_observation_counts={},
                task_sequence_counts={},
                observation_info={},
                estimated_data_volume_gb=0.0,
                data_volume_by_task={},
                cvz_time_minutes=0.0,
                cvz_sequence_count=0,
            )

        # Sort sequences chronologically
        all_sequences.sort(key=lambda s: s.start)

        # Calculate timing metrics
        schedule_start = all_sequences[0].start
        schedule_end = all_sequences[-1].stop
        total_duration = (schedule_end - schedule_start).total_seconds() / 60.0

        # Calculate science time and overhead separately
        total_science = 0.0
        total_overhead = 0.0

        for seq in all_sequences:
            science_minutes = (
                seq.science_duration_minutes
                if seq.science_duration_minutes is not None
                else seq.duration_minutes
            )
            overhead_minutes = (
                seq.duration_minutes - science_minutes
                if science_minutes is not None
                else 0.0
            )

            total_science += science_minutes
            total_overhead += overhead_minutes

        scheduling_efficiency = (
            total_science / total_duration if total_duration > 0 else 0.0
        )

        # Calculate per-task metrics
        task_durations = defaultdict(float)
        task_science_durations = defaultdict(float)
        task_obs_counts = defaultdict(int)
        task_seq_counts = defaultdict(int)
        data_volume_by_task = defaultdict(float)

        cvz_time = 0.0
        cvz_count = 0

        for seq in all_sequences:
            task_id = seq.visit_id

            if task_id == "CVZ":
                cvz_time += seq.duration_minutes
                cvz_count += 1
            else:
                task_durations[task_id] += seq.duration_minutes

                science_minutes = (
                    seq.science_duration_minutes
                    if seq.science_duration_minutes is not None
                    else seq.duration_minutes
                )
                task_science_durations[task_id] += science_minutes

                task_seq_counts[task_id] += 1

                # Estimate data volume
                nir_min = seq.nir_duration_minutes or 0.0
                vis_min = seq.vis_duration_minutes or 0.0
                volume = compute_data_volume_gb(
                    nir_min, vis_min, self.nir_rate, self.vis_rate
                )
                data_volume_by_task[task_id] += volume

        # Count unique observations per task
        obs_by_task = defaultdict(set)
        for seq in all_sequences:
            if seq.visit_id != "CVZ":
                obs_by_task[seq.visit_id].add(seq.obs_id)

        task_obs_counts = {
            task: len(obs_set) for task, obs_set in obs_by_task.items()
        }

        # Calculate total data volume
        total_data_volume = sum(data_volume_by_task.values())

        # Build observation info
        observation_info = self._build_observation_info(all_sequences)

        # Build timeline
        timeline = self._build_timeline(all_sequences)

        # Create diagnostics object
        diagnostics = ScheduleDiagnostics(
            schedule_start=schedule_start,
            schedule_end=schedule_end,
            total_duration_minutes=total_duration,
            total_science_minutes=total_science,
            total_overhead_minutes=total_overhead,
            scheduling_efficiency=scheduling_efficiency,
            visibility_utilization=0.0,  # Will calculate if input_observations provided
            task_durations=dict(task_durations),
            task_science_durations=dict(task_science_durations),
            task_observation_counts=task_obs_counts,
            task_sequence_counts=dict(task_seq_counts),
            observation_info=observation_info,
            estimated_data_volume_gb=total_data_volume,
            data_volume_by_task=dict(data_volume_by_task),
            cvz_time_minutes=cvz_time,
            cvz_sequence_count=cvz_count,
            timeline=timeline,
        )

        if input_observations:
            self._validate_completeness(
                visits, input_observations, diagnostics
            )
            self._calculate_visibility_utilization(
                all_sequences, input_observations, diagnostics
            )
            self._compare_scheduled_vs_requested(
                all_sequences, input_observations, diagnostics
            )

        # Check for missed scheduling opportunities during CVZ times
        self._check_cvz_gaps(all_sequences, input_observations, diagnostics)

        # Perform validations
        self._validate_no_overlaps(all_sequences, diagnostics)
        self._validate_time_ordering(all_sequences, diagnostics)
        self._validate_sequence_durations(all_sequences, diagnostics)
        self._validate_data_budget(total_data_volume, diagnostics)

        if dependency_json:
            self._validate_dependencies(visits, dependency_json, diagnostics)

        if input_observations:
            self._validate_completeness(
                visits, input_observations, diagnostics
            )
            self._calculate_visibility_utilization(
                all_sequences, input_observations, diagnostics
            )
            self._compare_scheduled_vs_requested(
                all_sequences, input_observations, diagnostics
            )

        # Check for missed scheduling opportunities during CVZ times
        self._check_cvz_gaps(all_sequences, input_observations, diagnostics)

        logger.info(
            f"Analysis complete: {len(diagnostics.validation_issues)} issues found"
        )

        return diagnostics

    def _compare_scheduled_vs_requested(
        self,
        sequences: List[ObservationSequence],
        input_observations: List[Observation],
        diagnostics: ScheduleDiagnostics,
    ):
        """
        Compare scheduled science time vs requested time for each observation.

        Args:
            sequences: Scheduled sequences
            input_observations: Original observation requests
            diagnostics: Diagnostics object to update
        """
        # Build map of scheduled time per observation
        scheduled_time = defaultdict(float)
        for seq in sequences:
            if seq.visit_id != "CVZ":
                science_minutes = (
                    seq.science_duration_minutes
                    if seq.science_duration_minutes is not None
                    else seq.duration_minutes
                )
                scheduled_time[seq.obs_id] += science_minutes

        # Compare with requested time
        for obs in input_observations:
            obs_id = obs.obs_id
            requested_seconds = obs.duration * 60.0 if obs.duration else 0.0
            scheduled_seconds = scheduled_time.get(obs_id, 0.0) * 60.0

            difference_seconds = scheduled_seconds - requested_seconds

            # Check if scheduled time is less than requested (undertime)
            # Allow small tolerance for rounding (1 second)
            if difference_seconds < -1.0:
                # Scheduled less than requested - this is a problem
                diagnostics.validation_issues.append(
                    ValidationIssue(
                        severity="error",
                        category="science_time",
                        message=f"Observation {obs_id} scheduled less time than requested",
                        details={
                            "obs_id": obs_id,
                            "requested_seconds": requested_seconds,
                            "scheduled_seconds": scheduled_seconds,
                            "difference_seconds": difference_seconds,
                            "requested_minutes": requested_seconds / 60.0,
                            "scheduled_minutes": scheduled_seconds / 60.0,
                        },
                    )
                )
            elif difference_seconds > 60.0:
                # Scheduled significantly more than requested (> 1 minute extra)
                # This is just info, not necessarily a problem
                diagnostics.validation_issues.append(
                    ValidationIssue(
                        severity="info",
                        category="science_time",
                        message=f"Observation {obs_id} scheduled more time than requested",
                        details={
                            "obs_id": obs_id,
                            "requested_seconds": requested_seconds,
                            "scheduled_seconds": scheduled_seconds,
                            "difference_seconds": difference_seconds,
                            "requested_minutes": requested_seconds / 60.0,
                            "scheduled_minutes": scheduled_seconds / 60.0,
                        },
                    )
                )

    def _check_cvz_gaps(
        self,
        sequences: List[ObservationSequence],
        input_observations: Optional[List[Observation]],
        diagnostics: ScheduleDiagnostics,
    ):
        """
        Check if there are schedulable observations during CVZ times.

        Args:
            sequences: All scheduled sequences
            input_observations: Original observations (with visibility windows)
            diagnostics: Diagnostics object to update
        """
        if not input_observations:
            return

        # Find all CVZ sequences
        cvz_sequences = [seq for seq in sequences if seq.visit_id == "CVZ"]

        if not cvz_sequences:
            return

        # Build map of when each observation was scheduled
        scheduled_times = defaultdict(list)
        for seq in sequences:
            if seq.visit_id != "CVZ":
                scheduled_times[seq.obs_id].append((seq.start, seq.stop))

        # Check each CVZ sequence
        for cvz_seq in cvz_sequences:
            cvz_start = cvz_seq.start
            cvz_end = cvz_seq.stop

            # Find observations that:
            # 1. Are visible during this CVZ time
            # 2. Are scheduled later (or not at all)

            visible_later_obs = []

            for obs in input_observations:
                # Check if observation is visible during CVZ time
                is_visible_during_cvz = False
                for vis_start, vis_end in obs.visibility_windows:
                    # Check for overlap with CVZ window
                    overlap_start = max(cvz_start, vis_start)
                    overlap_end = min(cvz_end, vis_end)
                    if overlap_start < overlap_end:
                        overlap_minutes = (
                            overlap_end - overlap_start
                        ).total_seconds() / 60.0
                        if overlap_minutes >= 2.0:  # At least 2 minutes
                            is_visible_during_cvz = True
                            break

                if not is_visible_during_cvz:
                    continue

                # Check if observation is scheduled later
                obs_scheduled_later = False
                if obs.obs_id in scheduled_times:
                    for sched_start, sched_end in scheduled_times[obs.obs_id]:
                        if sched_start > cvz_end:
                            obs_scheduled_later = True
                            break
                else:
                    # Not scheduled at all
                    obs_scheduled_later = True

                if obs_scheduled_later:
                    visible_later_obs.append(obs.obs_id)

            if visible_later_obs:
                diagnostics.validation_issues.append(
                    ValidationIssue(
                        severity="info",
                        category="cvz_gap",
                        message=f"CVZ sequence has {len(visible_later_obs)} visible observations scheduled later",
                        details={
                            "cvz_start": cvz_start.isoformat(),
                            "cvz_end": cvz_end.isoformat(),
                            "cvz_duration_minutes": (
                                cvz_end - cvz_start
                            ).total_seconds()
                            / 60.0,
                            "visible_observations": visible_later_obs[
                                :5
                            ],  # Limit to first 5
                            "total_count": len(visible_later_obs),
                        },
                    )
                )

    def _build_observation_info(
        self, sequences: List[ObservationSequence]
    ) -> Dict[str, Dict]:
        """Build detailed info for each observation."""
        obs_info = defaultdict(
            lambda: {
                "target": None,
                "ra": None,
                "dec": None,
                "sequence_count": 0,
                "total_duration_minutes": 0.0,
                "total_science_minutes": 0.0,
                "first_start": None,
                "last_end": None,
                "task_id": None,
            }
        )

        for seq in sequences:
            # Get task from visit_id or metadata
            task_id = seq.visit_id
            if task_id == "CVZ":
                continue

            info = obs_info[seq.obs_id]
            info["target"] = seq.target_name
            info["ra"] = seq.boresight_ra
            info["dec"] = seq.boresight_dec
            info["task_id"] = task_id
            info["sequence_count"] += 1
            info["total_duration_minutes"] += seq.duration_minutes

            # Use science_duration_minutes if available, otherwise use duration_minutes
            science_minutes = (
                seq.science_duration_minutes
                if seq.science_duration_minutes is not None
                else seq.duration_minutes
            )
            info["total_science_minutes"] += science_minutes

            if info["first_start"] is None or seq.start < info["first_start"]:
                info["first_start"] = seq.start

            if info["last_end"] is None or seq.stop > info["last_end"]:
                info["last_end"] = seq.stop

        return dict(obs_info)

    def _build_timeline(
        self, sequences: List[ObservationSequence]
    ) -> List[Dict]:
        """Build chronological timeline."""
        timeline = []

        for seq in sequences:
            # Calculate science minutes properly
            science_minutes = (
                seq.science_duration_minutes
                if seq.science_duration_minutes is not None
                else seq.duration_minutes
            )

            # Calculate overhead
            overhead_minutes = (
                seq.duration_minutes - science_minutes
                if science_minutes is not None
                else 0.0
            )

            entry = {
                "obs_id": seq.obs_id,
                "sequence_id": seq.sequence_id,
                "visit_id": seq.visit_id,
                "target": seq.target_name,
                "start": seq.start,
                "stop": seq.stop,
                "duration_minutes": seq.duration_minutes,
                "science_minutes": science_minutes,
                "overhead_minutes": overhead_minutes,  # ADD THIS
                "ra": seq.boresight_ra,
                "dec": seq.boresight_dec,
                "priority": seq.priority,
            }
            timeline.append(entry)

        return timeline

    def _validate_no_overlaps(
        self,
        sequences: List[ObservationSequence],
        diagnostics: ScheduleDiagnostics,
    ):
        """Check for overlapping sequences."""
        for i in range(len(sequences) - 1):
            seq1 = sequences[i]
            seq2 = sequences[i + 1]

            if seq1.stop > seq2.start:
                overlap_seconds = (seq1.stop - seq2.start).total_seconds()
                diagnostics.validation_issues.append(
                    ValidationIssue(
                        severity="error",
                        category="overlap",
                        message=f"Sequences overlap by {overlap_seconds:.1f} seconds",
                        details={
                            "sequence_1": f"{seq1.obs_id}_{seq1.sequence_id}",
                            "sequence_2": f"{seq2.obs_id}_{seq2.sequence_id}",
                            "overlap_seconds": overlap_seconds,
                        },
                    )
                )

    def _validate_time_ordering(
        self,
        sequences: List[ObservationSequence],
        diagnostics: ScheduleDiagnostics,
    ):
        """Validate sequences are properly time-ordered."""
        for seq in sequences:
            if seq.start >= seq.stop:
                diagnostics.validation_issues.append(
                    ValidationIssue(
                        severity="error",
                        category="timing",
                        message=f"Sequence {seq.obs_id}_{seq.sequence_id} has start >= stop",
                        details={
                            "sequence": f"{seq.obs_id}_{seq.sequence_id}",
                            "start": seq.start.isoformat(),
                            "stop": seq.stop.isoformat(),
                        },
                    )
                )

    def _validate_data_budget(
        self, total_volume: float, diagnostics: ScheduleDiagnostics
    ):
        """Validate data volume against budget."""
        if total_volume > self.max_data_volume:
            diagnostics.validation_issues.append(
                ValidationIssue(
                    severity="warning",
                    category="data_volume",
                    message=f"Estimated data volume ({total_volume:.2f} GB) exceeds budget ({self.max_data_volume:.2f} GB)",
                    details={
                        "estimated_volume_gb": total_volume,
                        "budget_gb": self.max_data_volume,
                        "excess_gb": total_volume - self.max_data_volume,
                    },
                )
            )
        else:
            percent_used = (total_volume / self.max_data_volume) * 100
            diagnostics.validation_issues.append(
                ValidationIssue(
                    severity="info",
                    category="data_volume",
                    message=f"Data volume within budget: {percent_used:.1f}% used",
                    details={
                        "estimated_volume_gb": total_volume,
                        "budget_gb": self.max_data_volume,
                        "percent_used": percent_used,
                    },
                )
            )

    def _validate_dependencies(
        self,
        visits: List[Visit],
        dependency_json: str,
        diagnostics: ScheduleDiagnostics,
    ):
        """Validate task dependencies were respected."""
        import json

        try:
            with open(dependency_json, "r") as f:
                dependencies = json.load(f)
        except Exception as e:
            diagnostics.validation_issues.append(
                ValidationIssue(
                    severity="warning",
                    category="dependency",
                    message=f"Could not load dependencies: {e}",
                    details={},
                )
            )
            return

        # Build task completion times
        task_completion = {}
        for visit in visits:
            if visit.visit_id == "CVZ":
                continue

            if visit.end_time:
                if (
                    visit.visit_id not in task_completion
                    or visit.end_time > task_completion[visit.visit_id]
                ):
                    task_completion[visit.visit_id] = visit.end_time

        # Check each dependency
        for task_id, prereqs in dependencies.items():
            if task_id not in task_completion:
                continue  # Task not scheduled

            task_start = None
            for visit in visits:
                if visit.visit_id == task_id and visit.start_time:
                    if task_start is None or visit.start_time < task_start:
                        task_start = visit.start_time

            if not task_start:
                continue

            for prereq in prereqs:
                if prereq not in task_completion:
                    diagnostics.validation_issues.append(
                        ValidationIssue(
                            severity="error",
                            category="dependency",
                            message=f"Task {task_id} depends on {prereq}, but {prereq} was not scheduled",
                            details={
                                "task": task_id,
                                "missing_prerequisite": prereq,
                            },
                        )
                    )
                elif task_completion[prereq] > task_start:
                    diagnostics.validation_issues.append(
                        ValidationIssue(
                            severity="error",
                            category="dependency",
                            message=f"Task {task_id} started before prerequisite {prereq} completed",
                            details={
                                "task": task_id,
                                "task_start": task_start.isoformat(),
                                "prerequisite": prereq,
                                "prerequisite_end": task_completion[
                                    prereq
                                ].isoformat(),
                            },
                        )
                    )

    def _validate_completeness(
        self,
        visits: List[Visit],
        input_observations: List[Observation],
        diagnostics: ScheduleDiagnostics,
    ):
        """Check if all requested observations were scheduled."""
        scheduled_obs_ids = set()
        for visit in visits:
            for seq in visit.observation_sequences:
                if seq.visit_id != "CVZ":
                    scheduled_obs_ids.add(seq.obs_id)

        requested_obs_ids = {obs.obs_id for obs in input_observations}

        unscheduled = requested_obs_ids - scheduled_obs_ids

        if unscheduled:
            diagnostics.validation_issues.append(
                ValidationIssue(
                    severity="warning",
                    category="completeness",
                    message=f"{len(unscheduled)} observations were not scheduled",
                    details={
                        "unscheduled_count": len(unscheduled),
                        "unscheduled_ids": sorted(list(unscheduled)),
                    },
                )
            )

    def _calculate_visibility_utilization(
        self,
        sequences: List[ObservationSequence],
        input_observations: List[Observation],
        diagnostics: ScheduleDiagnostics,
    ):
        """Calculate how efficiently visibility windows were used."""
        # Calculate total available visibility time
        total_visibility_minutes = 0.0
        for obs in input_observations:
            for start, end in obs.visibility_windows:
                total_visibility_minutes += (
                    end - start
                ).total_seconds() / 60.0

        # Calculate scheduled time (excluding CVZ)
        scheduled_minutes = sum(
            seq.duration_minutes for seq in sequences if seq.visit_id != "CVZ"
        )

        if total_visibility_minutes > 0:
            diagnostics.visibility_utilization = (
                scheduled_minutes / total_visibility_minutes
            )
        else:
            diagnostics.visibility_utilization = 0.0

    def _validate_sequence_durations(
        self,
        sequences: List[ObservationSequence],
        diagnostics: ScheduleDiagnostics,
        max_duration_minutes: float = 90.0,
    ):
        """
        Validate that all sequences respect the maximum duration constraint.

        Args:
            sequences: All scheduled sequences
            diagnostics: Diagnostics object to update
            max_duration_minutes: Maximum allowed sequence duration (default: 90)
        """
        for seq in sequences:
            total_duration = seq.duration_minutes

            if total_duration > max_duration_minutes:
                # Determine if this is science + overhead or just science
                science_minutes = (
                    seq.science_duration_minutes
                    if seq.science_duration_minutes is not None
                    else total_duration
                )
                overhead_minutes = total_duration - science_minutes

                diagnostics.validation_issues.append(
                    ValidationIssue(
                        severity="error",
                        category="sequence_duration",
                        message=f"Sequence {seq.obs_id}_{seq.sequence_id} exceeds 90-minute limit",
                        details={
                            "obs_id": seq.obs_id,
                            "sequence_id": seq.sequence_id,
                            "total_duration_minutes": total_duration,
                            "science_minutes": science_minutes,
                            "overhead_minutes": overhead_minutes,
                            "max_allowed_minutes": max_duration_minutes,
                            "excess_minutes": total_duration
                            - max_duration_minutes,
                        },
                    )
                )

    def _validate_continuous_constraints(
        self,
        sequences: List[ObservationSequence],
        continuous_constraints: List[ContinuousObservationConstraint],
        diagnostics: ScheduleDiagnostics,
    ):
        """Validate that continuous observation constraints were respected."""

        # Group sequences by observation ID
        sequences_by_obs = defaultdict(list)
        for seq in sequences:
            sequences_by_obs[seq.obs_id].append(seq)

        # Check each observation that should be continuous
        for obs_id, obs_sequences in sequences_by_obs.items():
            # Skip if observation doesn't require continuous scheduling
            if not any(
                c.applies_to(obs_sequences[0].parent_observation)
                for c in continuous_constraints
            ):
                continue

            # Sort sequences by start time
            obs_sequences.sort(key=lambda s: s.start)

            if len(obs_sequences) > 1:
                diagnostics.validation_issues.append(
                    ValidationIssue(
                        severity="error",
                        category="continuous_constraint",
                        message=f"Observation {obs_id} was split but requires continuous scheduling",
                        details={
                            "obs_id": obs_id,
                            "sequence_count": len(obs_sequences),
                            "start_time": obs_sequences[0].start.isoformat(),
                            "end_time": obs_sequences[-1].stop.isoformat(),
                        },
                    )
                )


def analyze_schedule_from_xml(
    schedule_xml_path: str,
    input_xml_dir: Optional[str] = None,
    dependency_json: Optional[str] = None,
    nir_rate: float = 2.74,
    vis_rate: float = 0.88,
    max_data_gb: float = 150.0,
) -> ScheduleDiagnostics:
    """
    Analyze a schedule from an XML file.

    Args:
        schedule_xml_path: Path to output schedule XML
        input_xml_dir: Directory of input observation XMLs (for comparison)
        dependency_json: Path to dependency JSON file
        nir_rate: NIR data rate (Mbps)
        vis_rate: Visible data rate (Mbps)
        max_data_gb: Maximum data budget (GB)

    Returns:
        ScheduleDiagnostics
    """
    from .xml_io import ObservationParser

    # Parse schedule XML
    visits = _parse_schedule_xml(schedule_xml_path)

    # Parse input observations if provided
    input_observations = None
    if input_xml_dir:
        parser = ObservationParser()
        input_observations = parser.parse_directory(input_xml_dir)

    # Analyze
    analyzer = DiagnosticsAnalyzer(nir_rate, vis_rate, max_data_gb)
    diagnostics = analyzer.analyze_schedule(
        visits, input_observations, dependency_json
    )

    return diagnostics


def _parse_schedule_xml(xml_path: str) -> List[Visit]:
    """Parse schedule XML into Visit objects."""
    from .models import Visit, ObservationSequence
    from .utils import parse_utc_time

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Strip namespace
    for elem in root.iter():
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]

    visits = []

    for visit_elem in root.findall("Visit"):
        visit_id_elem = visit_elem.find("ID")
        visit_id = (
            visit_id_elem.text if visit_id_elem is not None else "Unknown"
        )

        visit = Visit(visit_id=visit_id)

        for seq_elem in visit_elem.findall("Observation_Sequence"):
            seq_id_elem = seq_elem.find("ID")
            seq_id = seq_id_elem.text if seq_id_elem is not None else "000"

            # Parse observational parameters
            target_elem = seq_elem.find(".//Target")
            target = target_elem.text if target_elem is not None else "Unknown"

            priority_elem = seq_elem.find(".//Priority")
            priority = (
                float(priority_elem.text) if priority_elem is not None else 1.0
            )

            start_elem = seq_elem.find(".//Timing/Start")
            stop_elem = seq_elem.find(".//Timing/Stop")
            start = (
                parse_utc_time(start_elem.text)
                if start_elem is not None
                else datetime.now()
            )
            stop = (
                parse_utc_time(stop_elem.text)
                if stop_elem is not None
                else datetime.now()
            )

            ra_elem = seq_elem.find(".//Boresight/RA")
            dec_elem = seq_elem.find(".//Boresight/DEC")
            ra = float(ra_elem.text) if ra_elem is not None else 0.0
            dec = float(dec_elem.text) if dec_elem is not None else 0.0

            seq = ObservationSequence(
                obs_id=f"{visit_id}_{seq_id}",
                sequence_id=seq_id,
                start=start,
                stop=stop,
                target_name=target,
                boresight_ra=ra,
                boresight_dec=dec,
                priority=priority,
            )

            visit.observation_sequences.append(seq)

        visits.append(visit)

    return visits


def print_diagnostics_report(
    diagnostics: ScheduleDiagnostics,
    save_to_file: Optional[str] = None,
    detailed: bool = True,
):
    """
    Print diagnostics report to console and optionally save to file.

    Args:
        diagnostics: ScheduleDiagnostics object
        save_to_file: Optional path to save report
        detailed: Include detailed breakdowns
    """
    lines = []

    def add_line(text=""):
        lines.append(text)
        print(text)

    # Header
    add_line("=" * 80)
    add_line("SCHEDULE DIAGNOSTICS REPORT")
    add_line("=" * 80)

    # Schedule Overview
    add_line("\nSCHEDULE OVERVIEW:")
    add_line(
        f"  Start Time:           {diagnostics.schedule_start.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    add_line(
        f"  End Time:             {diagnostics.schedule_end.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    add_line(
        f"  Total Duration:       {diagnostics.total_duration_minutes:.1f} minutes ({diagnostics.total_duration_minutes/60:.1f} hours)"
    )
    add_line(
        f"  Science Time:         {diagnostics.total_science_minutes:.1f} minutes ({diagnostics.total_science_minutes/60:.1f} hours)"
    )
    add_line(
        f"  Overhead Time:        {diagnostics.total_overhead_minutes:.1f} minutes ({diagnostics.total_overhead_minutes/60:.1f} hours)"
    )
    add_line(
        f"  Scheduling Efficiency: {diagnostics.scheduling_efficiency:.1%}"
    )

    if diagnostics.visibility_utilization > 0:
        add_line(
            f"  Visibility Utilization: {diagnostics.visibility_utilization:.1%}"
        )

    # Data Volume
    add_line(f"\nDATA VOLUME (ESTIMATED):")
    add_line(
        f"  Total:                {diagnostics.estimated_data_volume_gb:.2f} GB"
    )
    add_line(
        f"  Budget:               {diagnostics.estimated_data_volume_gb:.2f} / 150.00 GB ({diagnostics.estimated_data_volume_gb/150*100:.1f}%)"
    )

    # CVZ Metrics
    add_line(f"\nCVZ IDLE TIME:")
    add_line(
        f"  Total CVZ Time:       {diagnostics.cvz_time_minutes:.1f} minutes ({diagnostics.cvz_time_minutes/60:.1f} hours)"
    )
    add_line(f"  CVZ Sequences:        {diagnostics.cvz_sequence_count}")

    # Task Summary
    add_line(f"\nTASK SUMMARY:")
    add_line(
        f"  {'Task':<10} {'Observations':<15} {'Sequences':<12} {'Science (min)':<15} {'Total (min)':<15} {'Data (GB)':<10}"
    )
    add_line(f"  {'-'*10} {'-'*15} {'-'*12} {'-'*15} {'-'*15} {'-'*10}")

    for task_id in sorted(diagnostics.task_durations.keys()):
        obs_count = diagnostics.task_observation_counts.get(task_id, 0)
        seq_count = diagnostics.task_sequence_counts.get(task_id, 0)
        science_duration = diagnostics.task_science_durations.get(
            task_id, 0.0
        )  # Use science_durations
        total_duration = diagnostics.task_durations[task_id]
        data = diagnostics.data_volume_by_task.get(task_id, 0.0)
        add_line(
            f"  {task_id:<10} {obs_count:<15} {seq_count:<12} {science_duration:<15.1f} {total_duration:<15.1f} {data:<10.2f}"
        )

    # Science time comparison
    science_time_issues = [
        i
        for i in diagnostics.validation_issues
        if i.category == "science_time"
    ]
    if science_time_issues:
        add_line(f"\nSCIENCE TIME COMPARISON (Scheduled vs Requested):")
        add_line(
            f"  {'Obs ID':<15} {'Requested (s)':<15} {'Scheduled (s)':<15} {'Difference (s)':<15}"
        )
        add_line(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*15}")

        for issue in science_time_issues:
            details = issue.details
            add_line(
                f"  {details['obs_id']:<15} {details['requested_seconds']:<15.1f} {details['scheduled_seconds']:<15.1f} {details['difference_seconds']:<15.1f}"
            )

    # Sequence duration violations
    duration_issues = [
        i
        for i in diagnostics.validation_issues
        if i.category == "sequence_duration"
    ]
    if duration_issues:
        add_line(f"\nSEQUENCE DURATION VIOLATIONS:")
        add_line(
            f"  Found {len(duration_issues)} sequences exceeding 90-minute limit"
        )
        if detailed:
            add_line(
                f"  {'Obs ID':<15} {'Seq ID':<10} {'Duration (min)':<15} {'Excess (min)':<15}"
            )
            add_line(f"  {'-'*15} {'-'*10} {'-'*15} {'-'*15}")
            for issue in duration_issues[:20]:  # Show first 20
                details = issue.details
                add_line(
                    f"  {details['obs_id']:<15} {details['sequence_id']:<10} {details['total_duration_minutes']:<15.1f} {details['excess_minutes']:<15.1f}"
                )

    # CVZ gap analysis
    cvz_gap_issues = [
        i for i in diagnostics.validation_issues if i.category == "cvz_gap"
    ]
    if cvz_gap_issues:
        add_line(f"\nCVZ GAP ANALYSIS:")
        add_line(
            f"  Found {len(cvz_gap_issues)} CVZ sequences with potentially schedulable observations"
        )
        if detailed:
            for idx, issue in enumerate(
                cvz_gap_issues[:10], 1
            ):  # Show first 10
                details = issue.details
                add_line(f"\n  CVZ Sequence {idx}:")
                add_line(
                    f"    Time: {details['cvz_start']} to {details['cvz_end']}"
                )
                add_line(
                    f"    Duration: {details['cvz_duration_minutes']:.1f} minutes"
                )
                add_line(
                    f"    Visible later observations ({details['total_count']}): {', '.join(details['visible_observations'])}"
                )

    # Validation Issues
    if diagnostics.validation_issues:
        add_line(
            f"\nVALIDATION ISSUES ({len(diagnostics.validation_issues)}):"
        )

        errors = [
            i for i in diagnostics.validation_issues if i.severity == "error"
        ]
        warnings = [
            i for i in diagnostics.validation_issues if i.severity == "warning"
        ]
        infos = [
            i for i in diagnostics.validation_issues if i.severity == "info"
        ]

        if errors:
            add_line(f"\n  ERRORS ({len(errors)}):")
            for issue in errors:
                add_line(f"    - [{issue.category}] {issue.message}")

        if warnings:
            add_line(f"\n  WARNINGS ({len(warnings)}):")
            for issue in warnings:
                add_line(f"    - [{issue.category}] {issue.message}")

        if detailed and infos:
            add_line(f"\n  INFO ({len(infos)}):")
            for issue in infos:
                add_line(f"    - [{issue.category}] {issue.message}")
    else:
        add_line(f"\nVALIDATION: No issues found ✓")

    # Detailed observation info
    if detailed and diagnostics.observation_info:
        add_line(f"\nOBSERVATION DETAILS:")
        add_line(
            f"  {'Obs ID':<15} {'Target':<20} {'Sequences':<10} {'Duration (min)':<15} {'First Start':<20}"
        )
        add_line(f"  {'-'*15} {'-'*20} {'-'*10} {'-'*15} {'-'*20}")

        # Sort by task and observation
        sorted_obs = sorted(
            diagnostics.observation_info.items(),
            key=lambda x: (x[1]["task_id"] or "", x[0]),
        )

        for obs_id, info in sorted_obs:
            target = (info["target"] or "Unknown")[:20]
            seq_count = info["sequence_count"]
            duration = info["total_duration_minutes"]
            first_start = (
                info["first_start"].strftime("%Y-%m-%d %H:%M")
                if info["first_start"]
                else "N/A"
            )
            add_line(
                f"  {obs_id:<15} {target:<20} {seq_count:<10} {duration:<15.1f} {first_start:<20}"
            )

    # Timeline preview
    if detailed and len(diagnostics.timeline) <= 50:
        add_line(f"\nTIMELINE ({len(diagnostics.timeline)} sequences):")
        add_line(
            f"  {'#':<4} {'Obs ID':<15} {'Target':<15} {'Start':<20} {'Duration':<12}"
        )
        add_line(f"  {'-'*4} {'-'*15} {'-'*15} {'-'*20} {'-'*12}")

        for i, entry in enumerate(diagnostics.timeline, 1):
            obs_id = entry["obs_id"]
            target = (entry["target"] or "Unknown")[:15]
            start = entry["start"].strftime("%Y-%m-%d %H:%M")
            duration = f"{entry['duration_minutes']:.1f} min"
            add_line(
                f"  {i:<4} {obs_id:<15} {target:<15} {start:<20} {duration:<12}"
            )
    elif len(diagnostics.timeline) > 50:
        add_line(
            f"\nTIMELINE: {len(diagnostics.timeline)} sequences (too many to display)"
        )

    add_line("\n" + "=" * 80)

    # Save to file if requested
    if save_to_file:
        try:
            with open(save_to_file, "w") as f:
                f.write("\n".join(lines))
            print(f"\nReport saved to: {save_to_file}")
        except Exception as e:
            print(f"\nError saving report to file: {e}")


# Convenience function
def analyze_schedule_diagnostics(
    schedule_xml_path: Optional[str] = None,
    visits: Optional[List[Visit]] = None,
    sequences: Optional[List[ObservationSequence]] = None,  # NEW
    input_xml_dir: Optional[str] = None,
    dependency_json: Optional[str] = None,
    print_report: bool = True,
    save_report: Optional[str] = None,
    detailed: bool = True,
    nir_rate: float = 2.74,
    vis_rate: float = 0.88,
    max_data_gb: float = 150.0,
) -> ScheduleDiagnostics:
    """
    Analyze schedule from visits, sequences, or XML file.

    Priority: sequences > visits > XML file
    """
    from .xml_io import ObservationParser
    from .diagnostics import DiagnosticsAnalyzer, print_diagnostics_report

    # Get sequences
    if sequences is not None:
        # Create temporary visits from sequences for analysis
        from .xml_io import ScheduleWriter

        temp_writer = ScheduleWriter(None)
        visit_groups = temp_writer._group_sequences_into_visits(sequences)
        visits_to_analyze = []
        for group in visit_groups:
            visit = Visit(visit_id="TEMP")
            visit.observation_sequences = group
            visits_to_analyze.append(visit)
    elif visits is not None:
        visits_to_analyze = visits
    elif schedule_xml_path is not None:
        from .diagnostics import _parse_schedule_xml

        visits_to_analyze = _parse_schedule_xml(schedule_xml_path)
    else:
        raise ValueError(
            "Must provide sequences, visits, or schedule_xml_path"
        )

    # Parse input observations
    input_observations = None
    if input_xml_dir:
        parser = ObservationParser()
        input_observations = parser.parse_directory(input_xml_dir)

    # Analyze
    analyzer = DiagnosticsAnalyzer(nir_rate, vis_rate, max_data_gb)
    diagnostics = analyzer.analyze_schedule(
        visits_to_analyze, input_observations, dependency_json
    )

    if print_report:
        print_diagnostics_report(diagnostics, save_report, detailed)

    return diagnostics


def print_diagnostics_report(
    diagnostics: ScheduleDiagnostics,
    save_to_file: Optional[str] = None,
    detailed: bool = True,
):
    """
    Print diagnostics report to console and optionally save to file.

    Args:
        diagnostics: ScheduleDiagnostics object
        save_to_file: Optional path to save report
        detailed: Include detailed breakdowns
    """
    lines = []

    def add_line(text=""):
        lines.append(text)
        print(text)

    # Header
    add_line("=" * 80)
    add_line("SCHEDULE DIAGNOSTICS REPORT")
    add_line("=" * 80)

    # Schedule Overview
    add_line("\nSCHEDULE OVERVIEW:")
    add_line(
        f"  Start Time:           {diagnostics.schedule_start.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    add_line(
        f"  End Time:             {diagnostics.schedule_end.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    add_line(
        f"  Total Duration:       {diagnostics.total_duration_minutes:.1f} minutes ({diagnostics.total_duration_minutes/60:.1f} hours)"
    )
    add_line(
        f"  Science Time:         {diagnostics.total_science_minutes:.1f} minutes ({diagnostics.total_science_minutes/60:.1f} hours)"
    )
    add_line(
        f"  Overhead Time:        {diagnostics.total_overhead_minutes:.1f} minutes ({diagnostics.total_overhead_minutes/60:.1f} hours)"
    )
    add_line(
        f"  Scheduling Efficiency: {diagnostics.scheduling_efficiency:.1%}"
    )

    if diagnostics.visibility_utilization > 0:
        add_line(
            f"  Visibility Utilization: {diagnostics.visibility_utilization:.1%}"
        )

    # Data Volume
    add_line(f"\nDATA VOLUME (ESTIMATED):")
    add_line(
        f"  Total:                {diagnostics.estimated_data_volume_gb:.2f} GB"
    )
    add_line(
        f"  Budget:               {diagnostics.estimated_data_volume_gb:.2f} / 150.00 GB ({diagnostics.estimated_data_volume_gb/150*100:.1f}%)"
    )

    # CVZ Metrics
    add_line(f"\nCVZ IDLE TIME:")
    add_line(
        f"  Total CVZ Time:       {diagnostics.cvz_time_minutes:.1f} minutes ({diagnostics.cvz_time_minutes/60:.1f} hours)"
    )
    add_line(f"  CVZ Sequences:        {diagnostics.cvz_sequence_count}")

    # Task Summary with science time comparison
    add_line(f"\nTASK SUMMARY:")
    add_line(
        f"  {'Task':<10} {'Observations':<15} {'Sequences':<12} {'Science (min)':<15} {'Total (min)':<15} {'Data (GB)':<10}"
    )
    add_line(f"  {'-'*10} {'-'*15} {'-'*12} {'-'*15} {'-'*15} {'-'*10}")

    for task_id in sorted(diagnostics.task_durations.keys()):
        obs_count = diagnostics.task_observation_counts.get(task_id, 0)
        seq_count = diagnostics.task_sequence_counts.get(task_id, 0)
        science_duration = diagnostics.task_science_durations.get(task_id, 0.0)
        total_duration = diagnostics.task_durations[task_id]
        data = diagnostics.data_volume_by_task.get(task_id, 0.0)
        add_line(
            f"  {task_id:<10} {obs_count:<15} {seq_count:<12} {science_duration:<15.1f} {total_duration:<15.1f} {data:<10.2f}"
        )

    # Science time comparison
    science_time_issues = [
        i
        for i in diagnostics.validation_issues
        if i.category == "science_time"
    ]
    if science_time_issues:
        add_line(f"\nSCIENCE TIME COMPARISON (Scheduled vs Requested):")
        add_line(
            f"  {'Obs ID':<15} {'Requested (s)':<15} {'Scheduled (s)':<15} {'Difference (s)':<15}"
        )
        add_line(f"  {'-'*15} {'-'*15} {'-'*15} {'-'*15}")

        for issue in science_time_issues:
            details = issue.details
            add_line(
                f"  {details['obs_id']:<15} {details['requested_seconds']:<15.1f} {details['scheduled_seconds']:<15.1f} {details['difference_seconds']:<15.1f}"
            )

    # CVZ gap analysis
    cvz_gap_issues = [
        i for i in diagnostics.validation_issues if i.category == "cvz_gap"
    ]
    if cvz_gap_issues:
        add_line(f"\nCVZ GAP ANALYSIS:")
        add_line(
            f"  Found {len(cvz_gap_issues)} CVZ sequences with potentially schedulable observations"
        )
        if detailed:
            for idx, issue in enumerate(
                cvz_gap_issues[:10], 1
            ):  # Show first 10
                details = issue.details
                add_line(f"\n  CVZ Sequence {idx}:")
                add_line(
                    f"    Time: {details['cvz_start']} to {details['cvz_end']}"
                )
                add_line(
                    f"    Duration: {details['cvz_duration_minutes']:.1f} minutes"
                )
                add_line(
                    f"    Visible later observations ({details['total_count']}): {', '.join(details['visible_observations'])}"
                )

    # Validation Issues
    if diagnostics.validation_issues:
        add_line(
            f"\nVALIDATION ISSUES ({len(diagnostics.validation_issues)}):"
        )

        errors = [
            i for i in diagnostics.validation_issues if i.severity == "error"
        ]
        warnings = [
            i for i in diagnostics.validation_issues if i.severity == "warning"
        ]
        infos = [
            i for i in diagnostics.validation_issues if i.severity == "info"
        ]

        if errors:
            add_line(f"\n  ERRORS ({len(errors)}):")
            for issue in errors:
                add_line(f"    - [{issue.category}] {issue.message}")

        if warnings:
            add_line(f"\n  WARNINGS ({len(warnings)}):")
            for issue in warnings:
                add_line(f"    - [{issue.category}] {issue.message}")

        if detailed and infos:
            add_line(f"\n  INFO ({len(infos)}):")
            for issue in infos[:20]:  # Limit to first 20 info messages
                add_line(f"    - [{issue.category}] {issue.message}")
    else:
        add_line(f"\nVALIDATION: No issues found ✓")

    # Detailed observation info
    if detailed and diagnostics.observation_info:
        add_line(f"\nOBSERVATION DETAILS:")
        add_line(
            f"  {'Obs ID':<15} {'Target':<20} {'Sequences':<10} {'Science (min)':<15} {'Total (min)':<15}"
        )
        add_line(f"  {'-'*15} {'-'*20} {'-'*10} {'-'*15} {'-'*15}")

        # Sort by task and observation
        sorted_obs = sorted(
            diagnostics.observation_info.items(),
            key=lambda x: (x[1]["task_id"] or "", x[0]),
        )

        for obs_id, info in sorted_obs:
            target = (info["target"] or "Unknown")[:20]
            seq_count = info["sequence_count"]
            science_duration = info["total_science_minutes"]
            total_duration = info["total_duration_minutes"]
            add_line(
                f"  {obs_id:<15} {target:<20} {seq_count:<10} {science_duration:<15.1f} {total_duration:<15.1f}"
            )

    add_line("\n" + "=" * 80)

    # Save to file if requested
    if save_to_file:
        try:
            with open(save_to_file, "w") as f:
                f.write("\n".join(lines))
            print(f"\nReport saved to: {save_to_file}")
        except Exception as e:
            print(f"\nError saving report to file: {e}")
