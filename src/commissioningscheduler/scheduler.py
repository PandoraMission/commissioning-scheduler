# scheduling.py
"""
Core scheduling logic for telescope observations.

This module orchestrates:
- Loading and parsing observations
- Computing visibility windows
- Scheduling observations respecting dependencies and constraints
- Handling CVZ idle pointings
- Creating the final schedule
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict, deque
import uuid
import os

from astropy.time import Time

from .models import (
    Observation,
    ObservationSequence,
    SchedulerConfig,
    SchedulingResult,
    BlockedTimeConstraint,
    BlockedTimeWindow,
    ContinuousObservationConstraint,
)
from .xml_io import ObservationParser, ScheduleWriter
from .visibility import (
    VisibilityCalculator,
    find_visible_cvz_pointing,
    compute_antisolar_coordinates,
)
from .constraints import (
    ConstraintChecker,
    SpecialConstraintHandler,
    Task0312Handler,
)
from .utils import (
    # align_to_minute_boundary,
    format_utc_time,
    parse_utc_time,
    compute_data_volume_gb,
)

logger = logging.getLogger(__name__)


class Scheduler:
    """
    Main scheduler class that orchestrates observation scheduling.

    Usage:
        config = SchedulerConfig(...)
        scheduler = Scheduler(config)
        result = scheduler.schedule(xml_paths, output_path)
    """

    def __init__(self, config: SchedulerConfig):
        """Initialize scheduler."""
        self.config = config
        self.parser = ObservationParser(
            keep_raw_tree=config.keep_raw_xml_trees
        )
        self.writer = ScheduleWriter(config)
        self.constraint_checker = ConstraintChecker(config)
        self.special_constraints = SpecialConstraintHandler()

        # Tracking - SIMPLIFIED
        self.current_time = config.commissioning_start
        self.scheduled_sequences: List[ObservationSequence] = (
            []
        )  # Just a flat list

        # Dependency tracking
        self.dependencies: Dict[str, List[str]] = {}
        self.completed_tasks: Set[str] = set()
        self.completed_observations: Set[str] = set()

        # Blocked time tracking
        self.blocked_time_constraints: List[BlockedTimeConstraint] = []
        self.blocked_time_windows: List[BlockedTimeWindow] = []

        # Last pointing for overhead calculation
        self.last_ra: Optional[float] = None
        self.last_dec: Optional[float] = None

        logger.info(
            f"Initialized Scheduler: {config.commissioning_start} to {config.commissioning_end}"
        )

    def schedule(
        self, xml_paths: List[str], output_path: str
    ) -> SchedulingResult:
        """Main scheduling entry point."""
        logger.info("=" * 80)
        logger.info("STARTING SCHEDULING PROCESS")
        logger.info("=" * 80)

        # Step 1: Parse observations
        logger.info("\n[STEP 1/7] Parsing observation XML files...")
        observations = self._parse_observations(xml_paths)
        if not observations:
            return SchedulingResult(
                success=False, message="No observations to schedule"
            )
        logger.info(f"✓ Parsed {len(observations)} observations")

        # Step 2: Load constraints
        logger.info("\n[STEP 2/7] Loading constraints...")
        self._load_constraints()
        logger.info("✓ Constraints loaded")

        # Step 3: Schedule fixed blocked time first
        logger.info("\n[STEP 3/7] Scheduling fixed blocked time windows...")
        self._schedule_fixed_blocked_time()
        logger.info("✓ Fixed blocked time scheduled")

        # Step 3.5: Initialize Moonshine/Earthshine now that constraints are loaded
        logger.info(
            "\n[STEP 4.5/7] Initializing Moonshine/Earthshine components..."
        )
        self._initialize_shine_components()

        # Step 3.6: Generate Moonshine/Earthshine observations (NEW)
        if self.shine_generator:
            moonshine_template = self._find_template_xml(xml_paths, "0342")
            earthshine_template = self._find_template_xml(xml_paths, "0341")

            if (
                moonshine_template
                and self.config.moonshine_config
                and self.config.moonshine_config.enabled
            ):
                moonshine_obs = (
                    self.shine_generator.generate_moonshine_observations(
                        moonshine_template,
                        Time(self.config.commissioning_start),
                        Time(self.config.commissioning_end),
                    )
                )
                observations.extend(moonshine_obs)
                logger.info(
                    f"✓ Generated {len(moonshine_obs)} Moonshine observations"
                )

            if (
                earthshine_template
                and self.config.earthshine_config
                and self.config.earthshine_config.enabled
            ):
                earthshine_obs = (
                    self.shine_generator.generate_earthshine_observations(
                        earthshine_template,
                        Time(self.config.commissioning_start),
                        Time(self.config.commissioning_end),
                    )
                )
                observations.extend(earthshine_obs)
                logger.info(
                    f"✓ Generated {len(earthshine_obs)} Earthshine observations"
                )

        # Step 4: Compute visibility
        logger.info("\n[STEP 4/7] Computing visibility windows...")
        self._compute_visibility(observations)
        logger.info("✓ Visibility computation complete")

        # Step 4.5: Compute Earthshine orbital visibility
        earthshine_obs = [
            o for o in observations if getattr(o, "is_earthshine", False)
        ]
        if earthshine_obs:
            logger.info(
                f"\n[STEP 4.5/7] Computing orbital position visibility for {len(earthshine_obs)} Earthshine observations..."
            )
            self.vis_calc.compute_earthshine_visibility(earthshine_obs)
            logger.info("✓ Earthshine orbital visibility computed")

        # Debug: Print visibility for 0312_000
        obs_0312 = next(
            (o for o in observations if o.obs_id == "0312_000"), None
        )
        if obs_0312:
            print_visibility_windows_for_observation(obs_0312)

        # Step 5: Schedule observations
        logger.info("\n[STEP 5/7] Scheduling observations...")
        unscheduled = self._schedule_observations(observations)
        logger.info(
            f"✓ Scheduled {len(observations) - len(unscheduled)}/{len(observations)} observations"
        )

        # Step 5.5: Schedule Moonshine block
        if self.moonshine_scheduler:
            moonshine_obs = [
                o for o in observations if getattr(o, "is_moonshine", False)
            ]
            unscheduled_moonshine = [
                o for o in moonshine_obs if o in unscheduled
            ]

            if unscheduled_moonshine:
                logger.info(
                    f"Scheduling {len(unscheduled_moonshine)} Moonshine observations in block"
                )
                moonshine_sequences, still_unscheduled = (
                    self.moonshine_scheduler.schedule_block(
                        unscheduled_moonshine,
                        Time(self.config.commissioning_start),
                        Time(self.config.commissioning_end),
                        self.scheduled_sequences,
                    )
                )

                # Add scheduled sequences
                self.scheduled_sequences.extend(moonshine_sequences)

                # Update unscheduled list
                scheduled_ids = {seq.obs_id for seq in moonshine_sequences}
                unscheduled = [
                    o
                    for o in unscheduled
                    if not (
                        getattr(o, "is_moonshine", False)
                        and o.obs_id in scheduled_ids
                    )
                ]
                unscheduled.extend(still_unscheduled)

                logger.info(
                    f"Moonshine: scheduled {len(moonshine_sequences)}, unscheduled {len(still_unscheduled)}"
                )

        # Step 5.6: Schedule Earthshine block if in block mode (NEW)
        if (
            self.earthshine_scheduler
            and self.config.earthshine_config.scheduling_mode == "block"
        ):
            earthshine_obs = [
                o for o in observations if getattr(o, "is_earthshine", False)
            ]
            unscheduled_earthshine = [
                o for o in earthshine_obs if o in unscheduled
            ]

            if unscheduled_earthshine:
                logger.info(
                    f"\n[STEP 5.6/7] Scheduling {len(unscheduled_earthshine)} "
                    f"Earthshine observations in block mode..."
                )

                earthshine_sequences, still_unscheduled = (
                    self.earthshine_scheduler.schedule_block(
                        unscheduled_earthshine,
                        Time(self.config.commissioning_start),
                        Time(self.config.commissioning_end),
                        self.scheduled_sequences,
                    )
                )

                # Add scheduled sequences
                self.scheduled_sequences.extend(earthshine_sequences)

                # Update unscheduled list
                scheduled_ids = {seq.obs_id for seq in earthshine_sequences}
                unscheduled = [
                    o
                    for o in unscheduled
                    if not (
                        getattr(o, "is_earthshine", False)
                        and o.obs_id in scheduled_ids
                    )
                ]
                unscheduled.extend(still_unscheduled)

                logger.info(
                    f"Earthshine block: scheduled {len(earthshine_sequences)}, "
                    f"unscheduled {len(still_unscheduled)}"
                )
        else:
            logger.info("Earthshine using flexible scheduling mode")

        # Step 6: Fill gaps with CVZ pointings
        logger.info("\n[STEP 6/7] Filling gaps with CVZ idle pointings...")
        self._fill_cvz_gaps()
        logger.info("✓ CVZ gap filling complete")

        # Step 7: Sort sequences chronologically and create visits
        logger.info(
            "\n[STEP 7/7] Organizing sequences into visits and writing output..."
        )
        self.scheduled_sequences.sort(key=lambda s: s.start)

        # Group sequences into visits (done by XML writer)
        metadata = self._generate_metadata()
        self.writer.write_schedule_from_sequences(
            self.scheduled_sequences, output_path, metadata
        )
        logger.info(f"✓ Schedule written to {output_path}")

        # Generate result
        result = self._generate_result(self.scheduled_sequences, unscheduled)

        logger.info("\n" + "=" * 80)
        logger.info("SCHEDULING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total sequences: {len(self.scheduled_sequences)}")
        logger.info(f"Unscheduled observations: {len(unscheduled)}")
        logger.info("=" * 80 + "\n")

        # Generate result and attach sequences for diagnostics
        result = self._generate_result(self.scheduled_sequences, unscheduled)
        result.scheduled_sequences = self.scheduled_sequences

        return result

    def _parse_observations(self, xml_paths: List[str]) -> List[Observation]:
        """Parse all observation XML files with diagnostic logging."""
        observations = []
        parse_errors = []

        for xml_path in xml_paths:
            # Skip template files - they're only used for generating synthetic observations
            if "_template_" in os.path.basename(xml_path):
                logger.debug(f"Skipping template file: {xml_path}")
                continue

            try:
                obs = self.parser.parse_file(
                    xml_path
                )  # Uses ObservationParser from xml_io.py
                observations.append(obs)
                logger.debug(f"Parsed {obs.obs_id} from {xml_path}")

                # Check for issues
                if obs.duration is None:
                    logger.warning(
                        f"⚠ Observation {obs.obs_id} has no duration calculated"
                    )
                if obs.boresight_ra is None or obs.boresight_dec is None:
                    logger.warning(
                        f"⚠ Observation {obs.obs_id} has no coordinates"
                    )

            except Exception as e:
                logger.error(f"Failed to parse {xml_path}: {e}")
                parse_errors.append((xml_path, str(e)))

        logger.info(f"Parsed {len(observations)} observations")

        # Summary of parsed observations
        if observations:
            logger.info("\nParsed observations summary:")
            no_duration = [o for o in observations if o.duration is None]
            no_coords = [
                o
                for o in observations
                if o.boresight_ra is None or o.boresight_dec is None
            ]

            if no_duration:
                logger.warning(
                    f"  {len(no_duration)} observations without duration: {[o.obs_id for o in no_duration]}"
                )
            if no_coords:
                logger.warning(
                    f"  {len(no_coords)} observations without coordinates: {[o.obs_id for o in no_coords]}"
                )

            logger.info(
                f"  {len(observations) - len(no_duration)} observations with valid duration"
            )
            logger.info(
                f"  {len(observations) - len(no_coords)} observations with valid coordinates"
            )

        if parse_errors:
            logger.error(f"\n{len(parse_errors)} files failed to parse:")
            for path, error in parse_errors:
                logger.error(f"  {path}: {error}")

        # Store for later gap filling
        self._all_observations = observations

        return observations

    def _load_constraints(self):
        """Load constraints from JSON file (dependencies and blocked time)."""
        # Support both new constraints.json and old dependency.json
        constraints_file = (
            self.config.constraints_json or self.config.dependency_json
        )

        if not constraints_file:
            logger.info("No constraints file specified")
            return

        try:
            with open(constraints_file, "r") as f:
                constraints_data = json.load(f)

            # Load dependencies
            if "dependencies" in constraints_data:
                self.dependencies = constraints_data["dependencies"]
                logger.info(
                    f"Loaded dependencies for {len(self.dependencies)} tasks"
                )

            # Load blocked time constraints
            if "blocked_time" in constraints_data:
                for bt_def in constraints_data["blocked_time"]:
                    constraint = self._parse_blocked_time_constraint(bt_def)
                    self.blocked_time_constraints.append(constraint)
                logger.info(
                    f"Loaded {len(self.blocked_time_constraints)} blocked time constraints"
                )

                # Log each constraint
                for bt in self.blocked_time_constraints:
                    if bt.constraint_type == "fixed":
                        logger.info(
                            f"  Fixed: {bt.start} to {bt.end} ({bt.window_type})"
                        )
                    elif bt.constraint_type == "after_task":
                        logger.info(
                            f"  After task {bt.task_id}: {bt.duration_minutes} min ({bt.window_type})"
                        )
                    elif bt.constraint_type == "after_observation":
                        logger.info(
                            f"  After obs {bt.observation_id}: {bt.duration_minutes} min ({bt.window_type})"
                        )

            # Load continuous observation constraints
            if "continuous_observations" in constraints_data:
                cont_data = constraints_data["continuous_observations"]
                constraint = ContinuousObservationConstraint(
                    tasks=cont_data.get("tasks", []),
                    observations=cont_data.get("observations", []),
                    description=cont_data.get("description", ""),
                )
                self.constraint_checker.continuous_constraints.append(
                    constraint
                )
                logger.info(
                    f"Loaded continuous observation constraint: "
                    f"{len(constraint.tasks)} tasks, {len(constraint.observations)} observations"
                )

            # Load Moonshine/Earthshine constraints
            self.constraint_checker._load_shine_constraints(constraints_data)

        except Exception as e:
            logger.error(
                f"Failed to load constraints from {constraints_file}: {e}"
            )

    def _compute_visibility(self, observations: List[Observation]):
        """Compute visibility windows for all observations."""
        start_time = Time(self.config.commissioning_start)
        end_time = Time(self.config.commissioning_end)

        # Create visibility calculator
        self.vis_calc = VisibilityCalculator(
            config=self.config,
            start=start_time,
            stop=end_time,
            timestep_seconds=60,  # TODO: Make configurable
        )

        # Compute visibility for all observations
        self.vis_calc.compute_for_observations(
            observations,
            force_recompute=False,
            parallel=True,
            max_workers=4,
            attach_to_observations=True,
        )

        # Log visibility statistics
        total_windows = sum(
            len(obs.visibility_windows) for obs in observations
        )
        logger.info(
            f"Computed {total_windows} visibility windows across {len(observations)} observations"
        )

    def _schedule_observations(
        self, observations: List[Observation]
    ) -> List[Observation]:  # NEEDS CHANGE
        """
        Schedule observations respecting dependencies and constraints.

        Returns:
            List of unscheduled observations
        """
        # Store all observations for later gap filling
        self._all_observations = observations

        # Group observations by task
        tasks = self._group_by_task(observations)

        logger.info(f"Grouped into {len(tasks)} tasks")
        for task_id, task_obs in sorted(tasks.items()):
            logger.info(f"  Task {task_id}: {len(task_obs)} observations")

        # Determine scheduling order using topological sort
        schedule_order = self._topological_sort(tasks.keys())

        logger.info(
            f"\nScheduling order determined: {' -> '.join(schedule_order)}"
        )

        unscheduled = []

        for idx, task_id in enumerate(schedule_order, 1):
            logger.info(
                f"\n--- Processing task {task_id} ({idx}/{len(schedule_order)}) ---"
            )

            # if not self._can_schedule_task(task_id):
            #     prereqs = self.dependencies.get(task_id, [])
            #     missing = [p for p in prereqs if p not in self.completed_tasks]
            #     logger.warning(f"Dependencies not met for task {task_id} (missing: {missing}), deferring")
            #     unscheduled.extend(tasks[task_id])
            #     continue

            # GOOD HERE TO BOTTOM

            task_obs = tasks[task_id]
            logger.info(
                f"Task {task_id} has {len(task_obs)} observations to schedule"
            )

            # Schedule each observation in the task
            task_scheduled = 0
            for obs_idx, obs in enumerate(task_obs, 1):
                logger.info(
                    f"  Scheduling observation {obs.obs_id} ({obs_idx}/{len(task_obs)})..."
                )
                logger.info(f"    Target: {obs.target_name or 'Unknown'}")

                # Normal scheduling (will handle pairs if this is first in pair)
                logger.info(
                    f"  Scheduling observation {obs.obs_id} ({obs_idx}/{len(task_obs)})..."
                )

                # Safe duration logging
                if obs.duration is not None:
                    logger.info(
                        f"    Duration: {obs.duration:.3f} minutes ({obs.duration*60:.1f} seconds)"
                    )
                else:
                    logger.info("    Duration: Not computed")

                # Safe coordinate logging
                if (
                    obs.boresight_ra is not None
                    and obs.boresight_dec is not None
                ):
                    logger.info(
                        f"    Coordinates: RA={obs.boresight_ra:.4f}, DEC={obs.boresight_dec:.4f}"
                    )
                else:
                    logger.info("    Coordinates: Not specified")

                logger.info(
                    f"    Visibility windows: {len(obs.visibility_windows)}"
                )

                if self._schedule_single_observation(obs):
                    logger.info(f"    ✓ Successfully scheduled {obs.obs_id}")
                    task_scheduled += 1

                    # Check if this observation triggers blocked time
                    self._check_and_add_triggered_blocked_time(
                        obs, task_complete=False
                    )
                else:
                    logger.warning(f"    ✗ Could not schedule {obs.obs_id}")
                    unscheduled.append(obs)

            # Mark task as complete if all observations scheduled
            if all(obs not in unscheduled for obs in task_obs):
                self.completed_tasks.add(task_id)

                # Check if task completion triggers blocked time
                # Use the last observation of the task as reference
                if task_obs:
                    self._check_and_add_triggered_blocked_time(
                        task_obs[-1], task_complete=True
                    )

                logger.info(
                    f"✓ Task {task_id} complete ({task_scheduled}/{len(task_obs)} observations scheduled)"
                )
            else:
                failed = len(task_obs) - task_scheduled
                logger.warning(
                    f"⚠ Task {task_id} partially complete ({task_scheduled}/{len(task_obs)} scheduled, {failed} failed)"
                )

        return unscheduled

    def _group_by_task(
        self, observations: List[Observation]
    ) -> Dict[str, List[Observation]]:
        """Group observations by task number."""
        tasks = defaultdict(list)

        for obs in observations:
            task_id = obs.task_number or "0000"
            tasks[task_id].append(obs)

        return dict(tasks)

    def _topological_sort(self, task_ids: Set[str]) -> List[str]:
        """
        Topological sort of tasks based on dependencies.

        Uses dependency depth (how many tasks depend on this task) as primary sort key,
        then numeric/alphabetic ordering as secondary key.

        Args:
            task_ids: Set of task IDs to sort

        Returns:
            List of task IDs in dependency order
        """
        # Build adjacency list and in-degree count
        in_degree = {task: 0 for task in task_ids}
        adj_list = defaultdict(list)
        reverse_deps = defaultdict(set)  # Track what depends on each task

        for task in task_ids:
            prereqs = self.dependencies.get(task, [])
            for prereq in prereqs:
                if prereq in task_ids:
                    adj_list[prereq].append(task)
                    in_degree[task] += 1
                    reverse_deps[prereq].add(task)

        # Calculate dependency depth (how many tasks transitively depend on this task)
        def count_dependents(task_id: str, visited: Set[str] = None) -> int:
            if visited is None:
                visited = set()
            if task_id in visited:
                return 0
            visited.add(task_id)

            count = len(reverse_deps[task_id])
            for dependent in reverse_deps[task_id]:
                count += count_dependents(dependent, visited)
            return count

        # Kahn's algorithm with priority
        queue = []
        for task in task_ids:
            if in_degree[task] == 0:
                dep_count = count_dependents(task)
                queue.append((dep_count, task))

        # Sort queue by dependency count (descending), then by task ID
        queue.sort(key=lambda x: (-x[0], x[1]))
        queue = deque([task for _, task in queue])

        result = []

        while queue:
            task = queue.popleft()
            result.append(task)

            # Process dependents
            next_tasks = []
            for dependent in adj_list[task]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    dep_count = count_dependents(dependent)
                    next_tasks.append((dep_count, dependent))

            # Sort by dependency count (descending), then by task ID
            next_tasks.sort(key=lambda x: (-x[0], x[1]))

            # Add to front of queue (higher priority tasks first)
            for _, next_task in next_tasks:
                queue.appendleft(next_task)

        # Check for cycles
        if len(result) != len(task_ids):
            logger.warning("Dependency cycle detected, using partial order")
            remaining = task_ids - set(result)
            # Sort remaining by dependency count
            remaining_sorted = sorted(
                remaining, key=lambda t: (-count_dependents(t), t)
            )
            result.extend(remaining_sorted)

        return result

    def _can_schedule_task(self, task_id: str) -> bool:
        """Check if task dependencies are satisfied."""
        prereqs = self.dependencies.get(task_id, [])
        return all(prereq in self.completed_tasks for prereq in prereqs)

    def _schedule_single_observation(self, obs: Observation) -> bool:
        """
        Schedule a single observation.

        Args:
            obs: Observation to schedule

        Returns:
            True if successfully scheduled, False otherwise
        """
        # Log the observation details
        logger.info(f"  Scheduling observation {obs.obs_id}...")
        logger.info(f"    Target: {obs.target_name or 'Unknown'}")

        if obs.duration is not None:
            logger.info(
                f"    Requested duration: {obs.duration:.4f} minutes ({obs.duration*60:.1f} seconds)"
            )

        # Special handling for Earthshine observations
        if getattr(obs, "is_earthshine", False):
            # Only schedule in flexible mode here
            # Block mode is handled separately in Step 5.6
            if (
                self.config.earthshine_config
                and self.config.earthshine_config.scheduling_mode == "block"
            ):
                logger.debug(
                    f"    Earthshine {obs.obs_id} deferred to block scheduling"
                )
                return False  # Will be handled in Step 5.6
            else:
                # Flexible mode - schedule now
                return self._schedule_earthshine_observation(obs)

        # Special handling for task 0312_000
        if Task0312Handler.is_task_0312(obs):
            return self._schedule_task_0312(obs)

        # Check if observation is schedulable
        constraint_result = (
            self.constraint_checker.check_observation_schedulable(
                obs, self.current_time
            )
        )

        if not constraint_result.valid:
            logger.warning("    Constraint check failed:")
            for msg in constraint_result.messages:
                logger.warning(f"      - {msg}")
            return False

        # Get visibility windows that:
        # 1. Are after current time
        # 2. Don't overlap with blocked time
        available_windows = []
        for vis_start, vis_end in obs.visibility_windows:
            if vis_end <= self.current_time:
                continue

            # Adjust window start if needed
            if vis_start < self.current_time:
                vis_start = self.current_time

            # Check if window overlaps with blocked time
            if self._is_time_blocked(vis_start, vis_end):
                logger.debug(
                    f"    Visibility window {vis_start} to {vis_end} overlaps blocked time, skipping"
                )
                continue

            available_windows.append((vis_start, vis_end))

        if not available_windows:
            logger.warning(
                "    No available visibility windows (may be blocked)"
            )
            return False

        logger.info(
            f"    Found {len(available_windows)} available visibility windows"
        )

        # Determine if we need overhead for first sequence
        needs_initial_overhead = self._needs_slew_overhead(obs)
        logger.info(f"    Needs overhead: {needs_initial_overhead}")

        # Split observation into sequences
        sequences = self.constraint_checker.split_observation_into_sequences(
            obs, available_windows, self.current_time, needs_initial_overhead
        )

        if not sequences:
            logger.warning("    Could not split into valid sequences")
            return False

        logger.info(
            f"    Constraint checker returned {len(sequences)} sequence(s)"
        )

        # Log what constraint checker returned
        for idx, (start, win_end, science_dur) in enumerate(sequences):
            logger.info(
                f"      Returned seq {idx+1}: start={start}, science={science_dur:.4f} min"
            )

        # Create sequences
        existing_seqs = [
            s for s in self.scheduled_sequences if s.obs_id == obs.obs_id
        ]
        seq_num = len(existing_seqs) + 1

        for seq_idx, (start, window_end, science_duration) in enumerate(
            sequences
        ):
            logger.info(f"      Processing sequence {seq_num}:")
            logger.info(
                f"        Input science_duration: {science_duration:.4f} min"
            )

            # Only first sequence has overhead
            overhead = (
                self.config.slew_overhead_minutes
                if (seq_idx == 0 and needs_initial_overhead)
                else 0.0
            )
            logger.info(
                f"        Overhead for this sequence: {overhead:.4f} min"
            )

            # Round science duration
            science_duration_rounded = self._round_duration_to_minute(
                science_duration
            )
            logger.info(
                f"        Rounded science_duration: {science_duration_rounded:.4f} min"
            )

            # Total duration
            total_duration = science_duration_rounded + overhead
            logger.info(f"        Total duration: {total_duration:.4f} min")

            sequence_end = start + timedelta(minutes=total_duration)
            logger.info(f"        Calculated end: {sequence_end}")

            logger.info(f"      Sequence {seq_num}:")
            logger.info(f"        Time: {start} to {sequence_end}")
            logger.info(
                f"        Science: {science_duration:.1f} min, Overhead: {overhead:.1f} min"
            )

            seq = ObservationSequence(
                obs_id=obs.obs_id,
                sequence_id=f"{seq_num:03d}",
                start=start,
                stop=sequence_end,
                parent_observation=obs,
                target_name=obs.target_name,
                boresight_ra=obs.boresight_ra,
                boresight_dec=obs.boresight_dec,
                priority=obs.priority,
                science_duration_minutes=science_duration,
                nir_duration_minutes=obs.nir_duration,
                vis_duration_minutes=obs.visible_duration,
                raw_xml_tree=obs.raw_xml_tree,
            )

            self._adjust_sequence_camera_parameters(seq, science_duration, obs)
            self.scheduled_sequences.append(seq)

            # Update tracking ONLY after first sequence
            if seq_idx == 0:
                self.last_ra = obs.boresight_ra
                self.last_dec = obs.boresight_dec

            # Update current time to end of this sequence
            self.current_time = sequence_end

            seq_num += 1
            logger.info(
                f"        ✓ Created sequence {seq.obs_id}_{seq.sequence_id}"
            )

        self.completed_observations.add(obs.obs_id)
        return True

    def _find_overlapping_visibility_window(  # NEEDS CHANGE
        self,
        first_obs: Observation,
        second_obs: Observation,
        required_duration: float,
        allow_truncation: bool = True,
    ) -> Optional[Tuple[datetime, datetime]]:
        """Find visibility window for paired observations."""
        best_window = None
        best_duration = 0.0

        for win1_start, win1_end in first_obs.visibility_windows:
            if win1_start < self.current_time:
                win1_start = self.current_time

            if win1_start >= win1_end:
                continue

            for win2_start, win2_end in second_obs.visibility_windows:
                # Find overlap
                overlap_start = max(win1_start, win2_start)
                overlap_end = min(win1_end, win2_end)

                if overlap_start >= overlap_end:
                    continue

                # Check if not blocked
                if self._is_time_blocked(overlap_start, overlap_end):
                    continue

                overlap_duration = (
                    overlap_end - overlap_start
                ).total_seconds() / 60.0

                # If truncation allowed, keep track of best window
                if allow_truncation:
                    if overlap_duration > best_duration:
                        best_duration = overlap_duration
                        best_window = (overlap_start, overlap_end)
                else:
                    # Without truncation, need full duration
                    if overlap_duration >= required_duration:
                        return (overlap_start, overlap_end)

        # If allowing truncation, return best window found
        if allow_truncation and best_window:
            logger.info(
                f"      Using truncated window: {best_duration:.1f} min "
                f"available (needed {required_duration:.1f} min)"
            )
            return best_window

        return None

    def _create_sequences_for_duration(  # NEEDS CHANGE
        self,
        obs: Observation,
        start_time: datetime,
        end_time: datetime,
        science_duration: float,
        needs_overhead: bool,
    ) -> List[ObservationSequence]:
        """
        Create observation sequences to fill a time span.

        Handles splitting if duration exceeds 90 minutes.

        Args:
            obs: Observation
            start_time: Start time
            end_time: End time
            science_duration: Total science time to schedule
            needs_overhead: Whether first sequence needs overhead

        Returns:
            List of ObservationSequence objects
        """
        sequences = []
        current_pos = start_time
        remaining_science = science_duration

        # Get existing sequence count for this observation
        existing_seqs = [
            s for s in self.scheduled_sequences if s.obs_id == obs.obs_id
        ]
        seq_num = len(existing_seqs) + 1

        first_sequence = True

        while remaining_science > 0 and current_pos < end_time:
            # Determine science duration for this sequence
            max_science = self.config.max_sequence_duration_minutes

            # Account for overhead on first sequence
            if first_sequence and needs_overhead:
                max_science -= 1.0  # Reserve space for overhead

            seq_science = min(remaining_science, max_science)

            # Add overhead to first sequence
            if first_sequence and needs_overhead:
                seq_total = seq_science + 1.0
            else:
                seq_total = seq_science

            seq_end = current_pos + timedelta(minutes=seq_total)

            # Make sure we don't exceed the allocated end time
            if seq_end > end_time:
                seq_end = end_time
                seq_total = (seq_end - current_pos).total_seconds() / 60.0
                seq_science = seq_total - (
                    1.0 if first_sequence and needs_overhead else 0.0
                )

            if seq_science <= 0:
                break

            seq = ObservationSequence(
                obs_id=obs.obs_id,
                sequence_id=f"{seq_num:03d}",
                start=current_pos,
                stop=seq_end,
                parent_observation=obs,
                target_name=obs.target_name,
                boresight_ra=obs.boresight_ra,
                boresight_dec=obs.boresight_dec,
                priority=obs.priority,
                science_duration_minutes=seq_science,
                nir_duration_minutes=obs.nir_duration,
                vis_duration_minutes=obs.visible_duration,
                raw_xml_tree=obs.raw_xml_tree,
            )

            # Adjust camera parameters
            if seq_science < obs.duration:
                # Partial observation - scale parameters
                scale_factor = seq_science / obs.duration
                scaled_params = Task0312Handler.calculate_scaled_parameters(
                    obs, scale_factor
                )
                seq.metadata["adjusted_params"] = scaled_params
            else:
                self._adjust_sequence_camera_parameters(seq, seq_science, obs)

            sequences.append(seq)

            current_pos = seq_end
            remaining_science -= seq_science
            first_sequence = False
            seq_num += 1

        return sequences

    def _needs_slew_overhead(self, obs: Observation) -> bool:
        """
        Check if observation needs slew overhead.

        Returns:
            True if 1 minute overhead should be included
        """
        # Check for no-overhead special constraint
        if self.special_constraints.requires_no_overhead(obs.obs_id):
            return False

        # Check if pointing changed
        if self.last_ra is None or self.last_dec is None:
            return True

        # Check if pointing is same (within small tolerance)
        if (
            abs(self.last_ra - (obs.boresight_ra or 0)) < 0.001
            and abs(self.last_dec - (obs.boresight_dec or 0)) < 0.001
        ):
            return False

        return True

    def _adjust_sequence_camera_parameters(
        self,
        seq: ObservationSequence,
        science_duration_minutes: float,
        parent_obs: Observation,
    ):
        """
        Adjust camera parameters (frames, integrations) based on actual science duration.

        This is critical for split observations where a sequence may have less time
        than the full observation request.

        Args:
            seq: ObservationSequence to adjust
            science_duration_minutes: Actual science time available (after overhead)
            parent_obs: Parent Observation with original parameters
        """
        # Store adjusted parameters in sequence metadata
        if "adjusted_params" not in seq.metadata:
            seq.metadata["adjusted_params"] = {}

        # Calculate visible camera parameters
        if (
            parent_obs.num_total_frames_requested
            and parent_obs.exposure_time_us
        ):
            # Calculate frame time in seconds
            frame_time_s = parent_obs.exposure_time_us / 1e6

            # Calculate how many frames fit in the science duration
            science_duration_s = science_duration_minutes * 60.0
            max_frames = int(science_duration_s / frame_time_s)

            # If there's a FramesPerCoadd requirement, round down to nearest multiple
            if parent_obs.frames_per_coadd:
                max_frames = (
                    max_frames // parent_obs.frames_per_coadd
                ) * parent_obs.frames_per_coadd

            # Don't exceed the original request
            adjusted_frames = min(
                max_frames, parent_obs.num_total_frames_requested
            )

            seq.metadata["adjusted_params"][
                "NumTotalFramesRequested"
            ] = adjusted_frames

            logger.debug(
                f"        Adjusted visible frames: {parent_obs.num_total_frames_requested} -> {adjusted_frames} "
                f"({science_duration_minutes:.3f} min available)"
            )

        # Calculate NIR camera parameters
        if (
            parent_obs.sc_integrations
            and parent_obs.roi_sizex
            and parent_obs.roi_sizey
        ):
            # Calculate single integration time
            frame_count_term = (
                (parent_obs.sc_resets1 or 0)
                + (parent_obs.sc_resets2 or 0)
                + (parent_obs.sc_dropframes1 or 0)
                + (parent_obs.sc_dropframes2 or 0)
                + (parent_obs.sc_dropframes3 or 0)
                + (parent_obs.sc_readframes or 0)
                + 1
            )
            pixel_term = (parent_obs.roi_sizex * parent_obs.roi_sizey) + (
                parent_obs.roi_sizey * 12
            )
            integration_time_s = frame_count_term * pixel_term * 0.00001

            # Calculate how many integrations fit in the science duration
            science_duration_s = science_duration_minutes * 60.0
            max_integrations = int(science_duration_s / integration_time_s)

            # Don't exceed the original request
            adjusted_integrations = min(
                max_integrations, parent_obs.sc_integrations
            )

            seq.metadata["adjusted_params"][
                "SC_Integrations"
            ] = adjusted_integrations

            logger.debug(
                f"        Adjusted NIR integrations: {parent_obs.sc_integrations} -> {adjusted_integrations} "
                f"({science_duration_minutes:.3f} min available)"
            )

    def _fill_cvz_gaps(self):
        """Fill scheduling gaps with CVZ idle pointings."""
        logger.info("Analyzing schedule for gaps...")

        # Find gaps in the schedule
        gaps = self._find_schedule_gaps()

        if not gaps:
            logger.info("No gaps found in schedule")

        # Try to fill gaps with science observations if enabled
        if self.config.enable_gap_filling and gaps:
            logger.info("Attempting to fill gaps with science observations...")
            gaps = self._try_fill_gaps_with_science(gaps)

        # Fill blocked time windows with CVZ sequences
        if self.blocked_time_windows:
            logger.info(
                f"Filling {len(self.blocked_time_windows)} blocked time windows with CVZ sequences..."
            )
            self._fill_blocked_time_with_cvz()

        # Fill remaining gaps with CVZ pointings
        if gaps:
            logger.info("Filling remaining gaps with CVZ idle pointings...")
            self._fill_gaps_with_cvz(gaps)

    def _find_schedule_gaps(self) -> List[Tuple[datetime, datetime]]:
        """
        Find gaps between scheduled observations.
        Does NOT include blocked time windows.
        """
        # Get all non-blocked, non-CVZ sequences
        science_sequences = [
            s
            for s in self.scheduled_sequences
            if s.obs_id not in ["CVZ", "CVZ_BLOCKED"]
        ]

        if not science_sequences:
            # Check if entire schedule is available or blocked
            gaps = [
                (
                    self.config.commissioning_start,
                    self.config.commissioning_end,
                )
            ]
        else:
            science_sequences.sort(key=lambda s: s.start)
            gaps = []

            # Gap before first sequence
            if science_sequences[0].start > self.config.commissioning_start:
                gaps.append(
                    (
                        self.config.commissioning_start,
                        science_sequences[0].start,
                    )
                )

            # Gaps between sequences
            for i in range(len(science_sequences) - 1):
                gap_start = science_sequences[i].stop
                gap_end = science_sequences[i + 1].start
                gap_duration = (gap_end - gap_start).total_seconds() / 60.0

                if gap_duration >= 2.0:
                    gaps.append((gap_start, gap_end))

            # Gap after last sequence
            if science_sequences[-1].stop < self.config.commissioning_end:
                gaps.append(
                    (science_sequences[-1].stop, self.config.commissioning_end)
                )

        # Remove any parts of gaps that overlap with blocked time
        filtered_gaps = []
        for gap_start, gap_end in gaps:
            # Split gap around blocked time windows
            current_start = gap_start

            for blocked in sorted(
                self.blocked_time_windows, key=lambda b: b.start_time
            ):
                if (
                    blocked.start_time >= gap_end
                    or blocked.end_time <= current_start
                ):
                    continue

                # Add gap before blocked time
                if current_start < blocked.start_time:
                    gap_duration = (
                        blocked.start_time - current_start
                    ).total_seconds() / 60.0
                    if gap_duration >= 2.0:
                        filtered_gaps.append(
                            (current_start, blocked.start_time)
                        )

                # Move start past blocked time
                current_start = blocked.end_time

            # Add remaining gap after all blocked times
            if current_start < gap_end:
                gap_duration = (gap_end - current_start).total_seconds() / 60.0
                if gap_duration >= 2.0:
                    filtered_gaps.append((current_start, gap_end))

        return filtered_gaps

    def _try_fill_gaps_with_science(
        self, gaps: List[Tuple[datetime, datetime]]
    ) -> List[Tuple[datetime, datetime]]:
        """
        Try to fill gaps with science observations from tasks that are ready.

        Args:
            gaps: List of (start, end) gaps

        Returns:
            List of remaining unfilled gaps
        """
        remaining_gaps = []

        # Get list of schedulable tasks (dependencies met)
        schedulable_tasks = [
            task_id
            for task_id in self.dependencies.keys()
            if self._can_schedule_task(task_id)
        ]

        if not schedulable_tasks:
            logger.info("  No schedulable tasks available for gap filling")
            return gaps

        logger.info(
            f"  Checking {len(schedulable_tasks)} schedulable tasks for gap opportunities"
        )

        # Get all observations from schedulable tasks that haven't been fully scheduled
        gap_fill_candidates = []
        for obs in self._all_observations:
            if obs.task_number in schedulable_tasks:
                # Check if observation has been scheduled (check in flat sequence list)
                is_scheduled = any(
                    seq.obs_id == obs.obs_id
                    for seq in self.scheduled_sequences
                    if seq.obs_id not in ["CVZ", "CVZ_BLOCKED"]
                )

                if not is_scheduled:
                    gap_fill_candidates.append(obs)

        if not gap_fill_candidates:
            logger.info(
                "  No unscheduled observations available for gap filling"
            )
            return gaps

        logger.info(
            f"  Found {len(gap_fill_candidates)} candidate observations for gap filling"
        )

        filled_count = 0
        for gap_start, gap_end in gaps:
            gap_duration = (gap_end - gap_start).total_seconds() / 60.0

            # Skip very short gaps
            if gap_duration < 2.0:
                remaining_gaps.append((gap_start, gap_end))
                continue

            # Try to find an observation that's visible during this gap
            filled = False
            for obs in gap_fill_candidates:
                # Check if observation is visible during this gap
                obs_visible_in_gap = False
                best_overlap_duration = 0.0

                for vis_start, vis_end in obs.visibility_windows:
                    overlap_start = max(gap_start, vis_start)
                    overlap_end = min(gap_end, vis_end)
                    if overlap_start < overlap_end:
                        overlap_duration = (
                            overlap_end - overlap_start
                        ).total_seconds() / 60.0
                        if overlap_duration >= 2.0:
                            obs_visible_in_gap = True
                            best_overlap_duration = max(
                                best_overlap_duration, overlap_duration
                            )

                if obs_visible_in_gap:
                    logger.info(
                        f"    Found {obs.obs_id} visible in gap ({best_overlap_duration:.1f} min overlap)"
                    )

                    # Try to schedule this observation in the gap
                    gap_windows = [(gap_start, gap_end)]
                    sequences = self.constraint_checker.split_observation_into_sequences(
                        obs, gap_windows, gap_start
                    )

                    if sequences:
                        # Determine sequence number for this observation
                        existing_seqs = [
                            s
                            for s in self.scheduled_sequences
                            if s.obs_id == obs.obs_id
                        ]
                        seq_num = len(existing_seqs) + 1

                        for start, end, science_duration in sequences:
                            # Check overhead
                            needs_overhead = self._needs_slew_overhead(obs)
                            overhead_minutes = (
                                self.config.slew_overhead_minutes
                                if needs_overhead
                                else 0.0
                            )

                            science_duration = self._round_duration_to_minute(
                                science_duration
                            )
                            total_duration = (
                                science_duration + overhead_minutes
                            )
                            actual_end = start + timedelta(
                                minutes=total_duration
                            )

                            # Make sure it fits in gap
                            if actual_end > gap_end:
                                science_duration = (
                                    gap_end - start
                                ).total_seconds() / 60.0 - overhead_minutes
                                if science_duration < 1.0:
                                    continue
                                actual_end = gap_end

                            seq = ObservationSequence(
                                obs_id=obs.obs_id,
                                sequence_id=f"{seq_num:03d}",
                                start=start,
                                stop=actual_end,
                                parent_observation=obs,
                                target_name=obs.target_name,
                                boresight_ra=obs.boresight_ra,
                                boresight_dec=obs.boresight_dec,
                                priority=obs.priority,
                                science_duration_minutes=science_duration,
                                nir_duration_minutes=obs.nir_duration,
                                vis_duration_minutes=obs.visible_duration,
                                raw_xml_tree=obs.raw_xml_tree,
                            )

                            self._adjust_sequence_camera_parameters(
                                seq, science_duration, obs
                            )

                            # Add to flat sequence list
                            self.scheduled_sequences.append(seq)
                            seq_num += 1

                            self.last_ra = obs.boresight_ra
                            self.last_dec = obs.boresight_dec

                            logger.info(
                                f"      ✓ Scheduled {obs.obs_id} in gap: {start} to {actual_end} ({science_duration:.1f} min science)"
                            )
                            filled = True
                            filled_count += 1

                            # Update gap
                            if actual_end < gap_end:
                                remaining_duration = (
                                    gap_end - actual_end
                                ).total_seconds() / 60.0
                                if remaining_duration >= 2.0:
                                    remaining_gaps.append(
                                        (actual_end, gap_end)
                                    )

                            break  # Only one sequence per attempt

                    if filled:
                        # Remove this observation from candidates
                        gap_fill_candidates.remove(obs)
                        break

            if not filled:
                remaining_gaps.append((gap_start, gap_end))

        if filled_count > 0:
            logger.info(
                f"  ✓ Filled {filled_count} gaps with science observations"
            )

        return remaining_gaps

    def _fill_gaps_with_cvz(self, gaps: List[Tuple[datetime, datetime]]):
        """Fill gaps with CVZ idle pointings."""
        if not gaps:
            return

        logger.info(f"Filling {len(gaps)} gaps with CVZ idle pointings...")

        skip_visibility = not self.config.verify_cvz_visibility

        for gap_idx, (gap_start, gap_end) in enumerate(gaps):
            gap_duration = (gap_end - gap_start).total_seconds() / 60.0

            # Find CVZ pointing
            midpoint = gap_start + (gap_end - gap_start) / 2
            antisolar_ra, dec = compute_antisolar_coordinates(
                Time(midpoint), self.config.cvz_coords[1]
            )

            if skip_visibility:
                ra, dec = antisolar_ra, dec
            else:
                coords = find_visible_cvz_pointing(
                    self.vis_calc,
                    Time(gap_start),
                    Time(gap_end),
                    self.config.cvz_coords[1],
                    skip_visibility_check=False,
                )
                if coords:
                    ra, dec = coords
                else:
                    ra, dec = antisolar_ra, dec

            # Split into chunks
            cvz_chunks = self._split_cvz_gap(gap_start, gap_end)

            logger.info(
                f"  Gap {gap_idx + 1}: {gap_start} to {gap_end} ({gap_duration:.1f} min)"
            )
            logger.info(
                f"    CVZ pointing at RA={ra:.2f}, DEC={dec:.2f}, {len(cvz_chunks)} chunk(s)"
            )

            for chunk_idx, (chunk_start, chunk_end) in enumerate(
                cvz_chunks, 1
            ):
                chunk_duration = (
                    chunk_end - chunk_start
                ).total_seconds() / 60.0

                seq = ObservationSequence(
                    obs_id="CVZ",
                    sequence_id=f"{chunk_idx:03d}",
                    start=chunk_start,
                    stop=chunk_end,
                    target_name="CVZ",
                    boresight_ra=ra,
                    boresight_dec=dec,
                    priority=0,
                    science_duration_minutes=chunk_duration,
                )

                seq.metadata["needs_vis_block"] = True

                # Add to flat sequence list
                self.scheduled_sequences.append(seq)

                logger.debug(
                    f"      Added CVZ sequence: {chunk_start} to {chunk_end}"
                )

    def _get_scheduled_duration(self, obs: Observation) -> float:
        """Get total duration already scheduled for an observation."""
        total = 0.0
        visit_id = obs.task_number or obs.obs_id.split("_")[0]
        visit = self.visits.get(visit_id)

        if visit:
            for seq in visit.observation_sequences:
                if seq.obs_id == obs.obs_id:
                    total += (
                        seq.science_duration_minutes or seq.duration_minutes
                    )

        return total

    def _round_duration_to_minute(self, duration_minutes: float) -> float:
        """
        Round duration to nearest minute for short observations.

        Args:
            duration_minutes: Duration in minutes

        Returns:
            Rounded duration in minutes
        """
        import math

        # If duration is less than 1 minute, round up to 1 minute
        if duration_minutes < 1.0:
            return 1.0

        # If duration has fractional seconds, round up to next minute
        if duration_minutes != int(duration_minutes):
            return math.ceil(duration_minutes)

        return duration_minutes

    def _split_cvz_gap(
        self, start: datetime, end: datetime, max_chunk_minutes: float = 90.0
    ) -> List[Tuple[datetime, datetime]]:
        """Split CVZ gap into chunks of maximum duration."""
        chunks = []
        current = start

        while current < end:
            remaining = (end - current).total_seconds() / 60.0
            chunk_duration = min(remaining, max_chunk_minutes)
            chunk_end = current + timedelta(minutes=chunk_duration)
            chunks.append((current, chunk_end))
            current = chunk_end

        return chunks

    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate metadata for output XML."""
        # Find actual start and end times from scheduled sequences
        if self.scheduled_sequences:
            actual_start = min(seq.start for seq in self.scheduled_sequences)
            actual_end = max(seq.stop for seq in self.scheduled_sequences)
        else:
            actual_start = self.config.commissioning_start
            actual_end = self.config.commissioning_end

        return {
            "Valid_From": format_utc_time(actual_start),
            "Expires": format_utc_time(actual_end),
            "Created": format_utc_time(datetime.now()),
            "Delivery_Id": str(uuid.uuid4()),
            "Total_Visits": "",
            "Total_Sequences": str(len(self.scheduled_sequences)),
            "Calendar_Status": "VALID",
            "TLE_Line1": str(self.config.tle_line1),
            "TLE_Line2": str(self.config.tle_line2),
        }

    def _generate_result(
        self,
        sequences: List[ObservationSequence],
        unscheduled: List[Observation],
    ) -> SchedulingResult:
        """Generate final scheduling result."""

        # Calculate metrics from sequences
        science_sequences = [
            s for s in sequences if s.obs_id not in ["CVZ", "CVZ_BLOCKED"]
        ]

        total_science = sum(
            seq.science_duration_minutes or seq.duration_minutes
            for seq in science_sequences
        )

        total_duration = sum(seq.duration_minutes for seq in sequences)

        # Estimate data volume
        total_data_volume = 0.0
        for seq in science_sequences:
            volume = compute_data_volume_gb(
                nir_minutes=seq.nir_duration_minutes or 0.0,
                vis_minutes=seq.vis_duration_minutes or 0.0,
                nir_rate_mbps=self.config.nir_data_rate_mbps,
                vis_rate_mbps=self.config.vis_data_rate_mbps,
            )
            total_data_volume += volume

        # Note: visits will be created during XML writing, so we don't have them here
        result = SchedulingResult(
            success=len(unscheduled) == 0,
            message=f"Scheduled {len(sequences)} sequences",
            visits=[],  # Will be populated by XML writer
            total_duration_minutes=total_duration,
            total_science_minutes=total_science,
            total_data_volume_gb=total_data_volume,
            unscheduled_observations=unscheduled,
        )

        if unscheduled:
            result.warnings.append(
                f"{len(unscheduled)} observations could not be scheduled"
            )

        return result

    def _parse_blocked_time_constraint(
        self, bt_def: Dict
    ) -> BlockedTimeConstraint:
        """Parse a blocked time constraint definition."""
        constraint_type = bt_def.get("type")
        window_type = bt_def.get("window_type", "blocked")
        description = bt_def.get("description", "")

        if constraint_type == "fixed":
            start = parse_utc_time(bt_def["start"])
            end = parse_utc_time(bt_def["end"])
            return BlockedTimeConstraint(
                constraint_type=constraint_type,
                window_type=window_type,
                description=description,
                start=start,
                end=end,
            )
        elif constraint_type == "after_task":
            return BlockedTimeConstraint(
                constraint_type=constraint_type,
                window_type=window_type,
                description=description,
                task_id=bt_def["task_id"],
                duration_minutes=float(bt_def["duration_minutes"]),
            )
        elif constraint_type == "after_observation":
            return BlockedTimeConstraint(
                constraint_type=constraint_type,
                window_type=window_type,
                description=description,
                observation_id=bt_def["observation_id"],
                duration_minutes=float(bt_def["duration_minutes"]),
            )
        else:
            raise ValueError(
                f"Unknown blocked time constraint type: {constraint_type}"
            )

    def _schedule_fixed_blocked_time(self):
        """Schedule all fixed blocked time windows."""
        fixed_constraints = [
            bt
            for bt in self.blocked_time_constraints
            if bt.constraint_type == "fixed"
        ]

        for constraint in fixed_constraints:
            window = BlockedTimeWindow(
                window_type=constraint.window_type,
                start_time=constraint.start,
                end_time=constraint.end,
                description=constraint.description,
            )

            self.blocked_time_windows.append(window)

            # Update current_time if this blocked time is in the future
            if constraint.start > self.current_time:
                logger.info(
                    f"  Scheduled fixed blocked time: {constraint.start} to {constraint.end}"
                )

            logger.debug(
                f"  Added fixed blocked time: {window.duration_minutes:.1f} minutes"
            )

    def _check_and_add_triggered_blocked_time(
        self, obs: Observation, task_complete: bool = False
    ):
        """
        Check if an observation or task completion triggers blocked time.

        Args:
            obs: Observation that just completed
            task_complete: True if the entire task is complete
        """
        # Check for observation-triggered constraints
        obs_constraints = [
            bt
            for bt in self.blocked_time_constraints
            if bt.constraint_type == "after_observation"
            and bt.observation_id == obs.obs_id
        ]

        for constraint in obs_constraints:
            start_time = self.current_time
            end_time = start_time + timedelta(
                minutes=constraint.duration_minutes
            )

            window = BlockedTimeWindow(
                window_type=constraint.window_type,
                start_time=start_time,
                end_time=end_time,
                description=constraint.description,
                trigger_obs_id=obs.obs_id,
            )

            self.blocked_time_windows.append(window)

            # Advance current_time past the blocked window
            self.current_time = end_time

            logger.info(
                f"  Added blocked time after {obs.obs_id}: {start_time} to {end_time} ({constraint.duration_minutes} min)"
            )

        # Check for task-triggered constraints if task is complete
        if task_complete and obs.task_number:
            task_constraints = [
                bt
                for bt in self.blocked_time_constraints
                if bt.constraint_type == "after_task"
                and bt.task_id == obs.task_number
            ]

            for constraint in task_constraints:
                start_time = self.current_time
                end_time = start_time + timedelta(
                    minutes=constraint.duration_minutes
                )

                window = BlockedTimeWindow(
                    window_type=constraint.window_type,
                    start_time=start_time,
                    end_time=end_time,
                    description=constraint.description,
                    trigger_task_id=obs.task_number,
                )

                self.blocked_time_windows.append(window)

                # Advance current_time past the blocked window
                self.current_time = end_time

                logger.info(
                    f"  Added blocked time after task {obs.task_number}: {start_time} to {end_time} ({constraint.duration_minutes} min)"
                )

    def _is_time_available(
        self, start_time: datetime, end_time: datetime
    ) -> bool:
        """
        Check if a time range is available (not blocked and not already scheduled).

        Args:
            start_time: Start of time range
            end_time: End of time range

        Returns:
            True if time range is available for scheduling
        """
        # Check against blocked time windows
        for blocked in self.blocked_time_windows:
            # Check for ANY overlap
            if start_time < blocked.end_time and end_time > blocked.start_time:
                logger.debug(
                    f"Time {start_time} to {end_time} overlaps with blocked time"
                )
                return False

        # Check against already scheduled sequences
        for seq in self.scheduled_sequences:
            if start_time < seq.stop and end_time > seq.start:
                logger.debug(
                    f"Time {start_time} to {end_time} overlaps with {seq.obs_id}_{seq.sequence_id}"
                )
                return False

        return True

    def _is_time_blocked(
        self, start_time: datetime, end_time: datetime
    ) -> bool:
        """Check if time range overlaps any blocked time window."""
        for blocked in self.blocked_time_windows:
            if start_time < blocked.end_time and end_time > blocked.start_time:
                return True
        return False

    def _fill_blocked_time_with_cvz(self):
        """Fill blocked time windows with CVZ observation sequences."""
        for blocked_idx, blocked_window in enumerate(
            self.blocked_time_windows
        ):
            # Split into 90-minute chunks first
            cvz_chunks = self._split_cvz_gap(
                blocked_window.start_time, blocked_window.end_time
            )

            logger.info(
                f"  Blocked time {blocked_idx + 1} ({blocked_window.window_type}): {blocked_window.start_time} to {blocked_window.end_time}"
            )
            logger.info(
                f"    Creating {len(cvz_chunks)} CVZ sequences with dynamic pointing"
            )

            skip_visibility = not self.config.verify_cvz_visibility

            for chunk_idx, (chunk_start, chunk_end) in enumerate(
                cvz_chunks, 1
            ):
                chunk_duration = (
                    chunk_end - chunk_start
                ).total_seconds() / 60.0

                # Calculate pointing for THIS chunk (not the whole blocked window)
                chunk_midpoint = chunk_start + (chunk_end - chunk_start) / 2
                antisolar_ra, dec = compute_antisolar_coordinates(
                    Time(chunk_midpoint), self.config.cvz_coords[1]
                )

                if skip_visibility:
                    ra, dec = antisolar_ra, dec
                else:
                    coords = find_visible_cvz_pointing(
                        self.vis_calc,
                        Time(chunk_start),
                        Time(chunk_end),
                        self.config.cvz_coords[1],
                        skip_visibility_check=False,
                    )
                    if coords:
                        ra, dec = coords
                    else:
                        ra, dec = antisolar_ra, dec

                seq = ObservationSequence(
                    obs_id="CVZ_BLOCKED",
                    sequence_id=f"{chunk_idx:03d}",
                    start=chunk_start,
                    stop=chunk_end,
                    target_name=f"CVZ_{blocked_window.window_type}",
                    boresight_ra=ra,
                    boresight_dec=dec,
                    priority=0,
                    science_duration_minutes=chunk_duration,
                )

                seq.metadata["needs_vis_block"] = True
                seq.metadata["blocked_type"] = blocked_window.window_type
                seq.metadata["blocked_description"] = (
                    blocked_window.description
                )

                self.scheduled_sequences.append(seq)

                logger.debug(
                    f"      Chunk {chunk_idx}: {chunk_start} to {chunk_end}, RA={ra:.2f}"
                )

    def _schedule_task_0312(self, obs: Observation) -> bool:
        """
        Special scheduling for task 0312_000.

        Creates 14 consecutive sequences following the specific pattern.

        Args:
            obs: Observation object for 0312_000

        Returns:
            True if successfully scheduled
        """
        logger.info("    Using special Task 0312 scheduling pattern")

        # Check if we have enough contiguous time
        required_minutes = 101.0  # 100 minutes + 1 minute overhead

        # Find a visibility window that can accommodate all 101 minutes
        suitable_window = None
        for vis_start, vis_end in obs.visibility_windows:
            if vis_start < self.current_time:
                vis_start = self.current_time

            if vis_start >= vis_end:
                continue

            # Check if window is large enough
            available_minutes = (vis_end - vis_start).total_seconds() / 60.0

            # Check if this time range is not blocked
            test_end = vis_start + timedelta(minutes=required_minutes)
            if not self._is_time_blocked(vis_start, test_end):
                if available_minutes >= required_minutes:
                    suitable_window = (vis_start, vis_end)
                    break

        if not suitable_window:
            logger.warning(
                "    No visibility window can accommodate 101 minutes for 0312_000"
            )
            return False

        win_start, win_end = suitable_window
        logger.info(f"    Found suitable window: {win_start} to {win_end}")

        # Determine if we need overhead
        needs_overhead = self._needs_slew_overhead(obs)

        # Create the 14 sequences
        task_sequences = Task0312Handler.create_sequences(
            obs, win_start, needs_overhead
        )

        logger.info(
            f"    Creating {len(task_sequences)} sequences for 0312 pattern"
        )

        # Calculate scaled parameters for data sequences (5 min / 100 min = 0.05)
        scaled_params = Task0312Handler.calculate_scaled_parameters(obs, 0.05)

        logger.info(f"    Scaled parameters: {scaled_params}")

        for idx, (seq_start, seq_end, science_duration, seq_type) in enumerate(
            task_sequences, 1
        ):
            seq = ObservationSequence(
                obs_id=obs.obs_id,
                sequence_id=f"{idx:03d}",
                start=seq_start,
                stop=seq_end,
                parent_observation=obs,
                target_name=obs.target_name,
                boresight_ra=obs.boresight_ra,
                boresight_dec=obs.boresight_dec,
                priority=int(obs.priority),
                science_duration_minutes=science_duration,
                nir_duration_minutes=(
                    obs.nir_duration if seq_type == "data" else None
                ),
                vis_duration_minutes=(
                    obs.visible_duration if seq_type == "data" else None
                ),
                raw_xml_tree=obs.raw_xml_tree if seq_type == "data" else None,
            )

            # Store sequence type in metadata
            seq.metadata["task_0312_type"] = seq_type

            if seq_type == "data":
                # Apply scaled parameters for data collection sequences
                seq.metadata["adjusted_params"] = scaled_params.copy()
            else:
                # Staring sequence - needs visible camera block only
                seq.metadata["needs_vis_block"] = True
                seq.metadata["is_staring"] = True

            self.scheduled_sequences.append(seq)

            logger.info(
                f"      Seq {idx}: {seq_start} to {seq_end} ({science_duration:.0f} min {seq_type})"
            )

        # Update tracking
        final_end = task_sequences[-1][1]  # End time of last sequence
        self.current_time = final_end
        self.last_ra = obs.boresight_ra
        self.last_dec = obs.boresight_dec

        # Mark observation as complete
        self.completed_observations.add(obs.obs_id)

        logger.info(
            "    ✓ Completed 0312_000 special scheduling (101 minutes total)"
        )

        return True

    def _validate_sequence_scheduling(
        self, start: datetime, end: datetime
    ) -> bool:  # LIKELY REMOVAL
        """
        Validate that a sequence can be scheduled in this time range.

        Args:
            start: Proposed start time
            end: Proposed end time

        Returns:
            True if time slot is available, False if conflicts exist
        """
        # Check for blocked time
        if self._is_time_blocked(start, end):
            logger.warning(
                f"      Time slot {start} to {end} overlaps with blocked time"
            )
            return False

        # Check for overlap with existing sequences
        for seq in self.scheduled_sequences:
            if seq.start < end and seq.stop > start:
                logger.warning(
                    f"      Time slot {start} to {end} overlaps with "
                    f"existing sequence {seq.obs_id}_{seq.sequence_id}"
                )
                return False

        return True

    def _scale_camera_parameters(
        self, obs: Observation, scale_factor: float
    ) -> dict:
        """
        Scale camera parameters (exposure counts) proportionally.

        Args:
            obs: Observation being scaled
            scale_factor: Fraction to scale by (e.g., 0.5 for half duration)

        Returns:
            Dictionary of scaled parameters
        """
        scaled = {}

        # Scale NIR exposures if present
        if obs.nir_duration and obs.nir_duration > 0:
            original_nir = obs.nir_duration
            scaled_nir = original_nir * scale_factor
            scaled["nir_duration_minutes"] = scaled_nir

            # If we have exposure info, scale count (round to nearest integer)
            if hasattr(obs, "nir_exposure_count"):
                scaled["nir_exposure_count"] = max(
                    1, round(obs.nir_exposure_count * scale_factor)
                )

        # Scale visible exposures if present
        if obs.visible_duration and obs.visible_duration > 0:
            original_vis = obs.visible_duration
            scaled_vis = original_vis * scale_factor
            scaled["vis_duration_minutes"] = scaled_vis

            if hasattr(obs, "vis_exposure_count"):
                scaled["vis_exposure_count"] = max(
                    1, round(obs.vis_exposure_count * scale_factor)
                )

        logger.info(f"      Scaled camera parameters by {scale_factor:.2f}:")
        for key, value in scaled.items():
            logger.info(f"        {key}: {value}")

        return scaled

    def _find_template_xml(
        self, xml_paths: List[str], task_number: str
    ) -> Optional[str]:
        """
        Find template XML file for given task number.

        Args:
            xml_paths: List of XML file paths
            task_number: Task number to search for (e.g., "0341", "0342")

        Returns:
            Path to template XML, or None if not found
        """
        for path in xml_paths:
            filename = os.path.basename(path)
            if filename.startswith(
                f"{task_number}_000_template"
            ) and filename.endswith(".xml"):
                logger.info(f"Found template for task {task_number}: {path}")
                return path

        logger.warning(f"Template XML not found for task {task_number}")
        return None

    def _initialize_shine_components(self):
        """
        Initialize Moonshine/Earthshine components after constraints are loaded.

        This must be called after _load_constraints() so that config.moonshine_config
        and config.earthshine_config are populated.
        """
        self.shine_generator = None
        self.moonshine_scheduler = None
        self.earthshine_scheduler = None

        # Create EphemerisProvider for shine calculations (uses same TLE as scheduler)
        shine_ephemeris = None
        if (
            hasattr(self.config, "moonshine_config")
            and self.config.moonshine_config
            and self.config.moonshine_config.enabled
        ) or (
            hasattr(self.config, "earthshine_config")
            and self.config.earthshine_config
            and self.config.earthshine_config.enabled
        ):

            try:
                from .shine_scheduler import EphemerisProvider

                # Create ephemeris provider using same TLE as scheduler
                shine_ephemeris = EphemerisProvider(
                    tle_line1=self.config.tle_line1,
                    tle_line2=self.config.tle_line2,
                    gmat_file=None,  # Can add GMAT file support later if needed
                )
                logger.info(
                    "✓ Created EphemerisProvider for Moonshine/Earthshine"
                )

            except ImportError as e:
                logger.warning(
                    f"Could not import shine_scheduler components: {e}"
                )
                return

        # Check if Moonshine is enabled
        if (
            hasattr(self.config, "moonshine_config")
            and self.config.moonshine_config
            and self.config.moonshine_config.enabled
        ):
            try:
                from .shine_scheduler import (
                    ShineObservationGenerator,
                    MoonshineScheduler,
                )

                # Use visibility calculator's ephemeris (already initialized)
                self.shine_generator = ShineObservationGenerator(
                    self.config, shine_ephemeris
                )
                self.moonshine_scheduler = MoonshineScheduler(
                    self.config, shine_ephemeris
                )
                logger.info("✓ Moonshine scheduling enabled")
            except ImportError as e:
                logger.warning(
                    f"Moonshine scheduling requested but shine_scheduler not available: {e}"
                )

        # Check if Earthshine is enabled
        if (
            hasattr(self.config, "earthshine_config")
            and self.config.earthshine_config
            and self.config.earthshine_config.enabled
        ):
            try:
                from .shine_scheduler import (
                    ShineObservationGenerator,
                    EarthshineScheduler,
                )

                if not self.shine_generator:
                    self.shine_generator = ShineObservationGenerator(
                        self.config, shine_ephemeris
                    )

                # Create EarthshineScheduler
                self.earthshine_scheduler = EarthshineScheduler(
                    self.config, shine_ephemeris, scheduler_ref=self
                )

                logger.info("✓ Earthshine scheduling enabled")
            except ImportError as e:
                logger.warning(
                    f"Earthshine scheduling requested but shine_scheduler not available: {e}"
                )

    def _schedule_earthshine_observation(self, obs: Observation) -> bool:
        """
        Schedule an Earthshine observation.

        Earthshine observations are orbital-position-dependent and get RA/Dec
        calculated at schedule time based on spacecraft position.

        Args:
            obs: Earthshine observation

        Returns:
            True if successfully scheduled, False otherwise
        """
        from astropy.time import Time
        from datetime import timedelta
        from .shine_scheduler import EarthshinePointing

        logger.info(f"    Scheduling Earthshine observation {obs.obs_id}")
        logger.info(f"    Orbital position: {obs.orbital_position_deg}°")
        logger.info(f"    Limb separation: {obs.limb_separation_deg}°")

        # Check dependencies FIRST
        task_id = obs.task_number

        if task_id and task_id in self.dependencies:
            prereqs = self.dependencies[task_id]
            missing_prereqs = [
                p for p in prereqs if p not in self.completed_tasks
            ]

            if missing_prereqs:
                logger.warning(
                    f"    Cannot schedule {obs.obs_id}: missing prerequisites {missing_prereqs}"
                )
                return False

        # Check if shine_generator is available
        if (
            not hasattr(self, "shine_generator")
            or self.shine_generator is None
        ):
            logger.error(
                "    Shine generator not initialized, cannot schedule Earthshine"
            )
            return False

        # Find next visibility window after current time
        next_window = None
        for window_start, window_end in obs.visibility_windows:
            if window_end > self.current_time:
                # Found a future window
                next_window = (
                    max(
                        window_start, self.current_time
                    ),  # Start no earlier than current time
                    window_end,
                )
                break

        if not next_window:
            logger.warning("    No future orbital position windows available")
            return False

        window_start, window_end = next_window

        # Check if there's enough time in the window
        duration_minutes = obs.calculated_duration_minutes or obs.duration
        available_minutes = (window_end - window_start).total_seconds() / 60.0

        if available_minutes < duration_minutes:
            logger.warning(
                f"    Not enough time in window: need {duration_minutes:.1f} min, "
                f"have {available_minutes:.1f} min"
            )
            return False

        # Calculate pointing at window start time
        try:
            earthshine_calc = EarthshinePointing(
                self.shine_generator.ephemeris
            )

            result = earthshine_calc.calculate_pointing(
                Time(window_start),
                obs.orbital_position_deg,
                obs.limb_separation_deg,
                max_search_orbits=1,  # Already at the right position
                position_tolerance_deg=obs.max_orbital_drift_deg,
            )

            # Verify Sun constraints
            if not result.pointing_in_antisolar:
                logger.warning(
                    f"    Pointing not in antisolar hemisphere "
                    f"(Sun angle={result.sun_angle_deg:.1f}°)"
                )
                return False

            if result.sun_angle_deg < 91.0:
                logger.warning(
                    f"    Sun avoidance violated ({result.sun_angle_deg:.1f}° < 91°)"
                )
                return False

            # Check orbital drift during observation
            end_time = window_start + timedelta(minutes=duration_minutes)
            sc_end = earthshine_calc.ephemeris.get_spacecraft_state(
                Time(end_time)
            )
            pos_end = earthshine_calc._get_orbital_position(sc_end)

            drift = abs(pos_end - result.orbital_position_deg)
            if drift > 180:
                drift = 360 - drift

            if drift > obs.max_orbital_drift_deg:
                logger.warning(
                    f"    Orbital drift {drift:.1f}° exceeds max {obs.max_orbital_drift_deg}° "
                    f"(scheduling anyway with warning)"
                )

            # Create the observation sequence
            from .models import ObservationSequence

            # Determine sequence number
            existing_seqs = [
                s for s in self.scheduled_sequences if s.obs_id == obs.obs_id
            ]
            seq_num = len(existing_seqs) + 1

            # Add overhead if needed (first sequence for this observation)
            needs_overhead = seq_num == 1 and self._needs_slew_overhead(obs)
            overhead_minutes = (
                self.config.slew_overhead_minutes if needs_overhead else 0.0
            )

            total_duration = duration_minutes + overhead_minutes
            final_end_time = window_start + timedelta(minutes=total_duration)

            sequence = ObservationSequence(
                obs_id=obs.obs_id,
                sequence_id=f"{seq_num:03d}",
                start=window_start,
                stop=final_end_time,
                parent_observation=obs,
                target_name=obs.target_name,
                boresight_ra=result.ra_deg,  # Calculated RA/Dec
                boresight_dec=result.dec_deg,
                priority=obs.priority,
                science_duration_minutes=duration_minutes,
                nir_duration_minutes=obs.nir_duration,
                vis_duration_minutes=obs.visible_duration,
                raw_xml_tree=obs.raw_xml_tree,
            )

            # Add to schedule
            self.scheduled_sequences.append(sequence)
            self.completed_observations.add(obs.obs_id)

            # Update tracking
            self.current_time = final_end_time
            self.last_ra = result.ra_deg
            self.last_dec = result.dec_deg

            logger.info(
                f"    ✓ Scheduled Earthshine {obs.obs_id} at {window_start.isoformat()}"
            )
            logger.info(
                f"      Orbital pos: {result.orbital_position_deg:.1f}°"
            )
            logger.info(
                f"      RA/Dec: {result.ra_deg:.2f}°, {result.dec_deg:.2f}°"
            )
            logger.info(f"      Sun angle: {result.sun_angle_deg:.1f}°")
            logger.info(
                f"      Duration: {total_duration:.1f} min (science: {duration_minutes:.1f}, overhead: {overhead_minutes:.1f})"
            )

            return True

        except Exception as e:
            logger.error(f"    Error calculating Earthshine pointing: {e}")
            import traceback

            traceback.print_exc()
            return False


def print_visibility_windows_for_observation(obs: Observation):
    """Print all visibility windows for an observation."""
    logger.info(f"\nVisibility windows for {obs.obs_id}:")
    logger.info(f"  Total windows: {len(obs.visibility_windows)}")

    for idx, (start, end) in enumerate(obs.visibility_windows, 1):
        duration = (end - start).total_seconds() / 60.0
        logger.info(f"  Window {idx}: {start} to {end} ({duration:.1f} min)")

    if obs.visibility_windows:
        total_visible = sum(
            (end - start).total_seconds() / 60.0
            for start, end in obs.visibility_windows
        )
        logger.info(f"  Total visible time: {total_visible:.1f} minutes")
