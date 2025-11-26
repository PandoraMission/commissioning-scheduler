# __init__.py
"""
Commissioning Scheduler Package

A modular scheduling system for space telescope observations.
"""

__version__ = "1.0.0"

from .models import (
    Observation,
    ObservationSequence,
    Visit,
    SchedulerConfig,
    SchedulingResult
)
from .scheduler import Scheduler
from .xml_io import ObservationParser, ScheduleWriter, gather_task_xmls
from .visibility import (
    VisibilityCalculator,
    compute_target_visibility,
    get_default_cache_dir,
    clear_visibility_cache,
    get_cache_info
)
from .constraints import ConstraintChecker, SpecialConstraintHandler
from .diagnostics import analyze_schedule_diagnostics, print_diagnostics_report

__all__ = [
    'Observation',
    'ObservationSequence',
    'Visit',
    'SchedulerConfig',
    'SchedulingResult',
    'Scheduler',
    'ObservationParser',
    'ScheduleWriter',
    'gather_task_xmls',
    'VisibilityCalculator',
    'compute_target_visibility',
    'get_default_cache_dir',
    'clear_visibility_cache',
    'get_cache_info',
    'ConstraintChecker',
    'SpecialConstraintHandler',
    'analyze_schedule_diagnostics',
    'print_diagnostics_report',
]


# Convenience functions

def schedule_observations(
    xml_dir: str,
    output_path: str,
    tle_line1: str,
    tle_line2: str,
    commissioning_start,
    commissioning_end,
    cvz_coords=(120.0, 8.5),
    **kwargs
):
    """
    Convenience function to schedule observations.

    Args:
        xml_dir: Directory containing input XML files
        output_path: Path for output schedule
        tle_line1: TLE line 1
        tle_line2: TLE line 2
        commissioning_start: Start datetime
        commissioning_end: End datetime
        cvz_coords: CVZ coordinates (RA, DEC)
        **kwargs: Additional SchedulerConfig parameters
            - constraints_json: Path to constraints file (dependencies and blocked time)
            - dependency_json: Deprecated, use constraints_json instead
            - max_data_volume_gb: Maximum data volume budget
            - verify_cvz_visibility: Check CVZ pointing visibility
            - enable_gap_filling: Try to fill gaps with science
            - etc.

    Returns:
        SchedulingResult
    """
    # Gather XML files
    xml_paths = gather_task_xmls(xml_dir)

    # Support both old and new parameter names
    if 'dependency_json' in kwargs and 'constraints_json' not in kwargs:
        kwargs['constraints_json'] = kwargs.pop('dependency_json')

    # Create config
    config = SchedulerConfig(
        tle_line1=tle_line1,
        tle_line2=tle_line2,
        commissioning_start=commissioning_start,
        commissioning_end=commissioning_end,
        cvz_coords=cvz_coords,
        **kwargs
    )

    # Create scheduler and run
    scheduler = Scheduler(config)
    result = scheduler.schedule(xml_paths, output_path)

    return result
