import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from pandoravisibility import Visibility


@dataclass
class ObservationCheck:
    """Results of visibility check for a single observation."""

    visit_id: str
    sequence_id: str
    task: str
    target: str
    ra: float
    dec: float
    start: str
    stop: str
    duration_minutes: float
    is_visible: bool
    visibility_fraction: float
    first_violation_time: Optional[str] = None
    num_violations: int = 0

    def __str__(self):
        status = "✓ PASS" if self.is_visible else "✗ FAIL"
        base = (
            f"{status} | Visit {self.visit_id} Seq {self.sequence_id} | "
            f"{self.task} | {self.target} | "
            f"RA={self.ra:.4f}° Dec={self.dec:.4f}° | "
            f"{self.start} to {self.stop} ({self.duration_minutes:.1f} min)"
        )

        if not self.is_visible:
            base += (
                f"\n        Visibility: {self.visibility_fraction:.1%} "
                f"({self.num_violations} violations, "
                f"first at {self.first_violation_time})"
            )

        return base


class CalendarVisibilityChecker:
    """
    Parse science calendar XML and verify visibility constraints for all observations.
    """

    def __init__(self, xml_path: str, time_step: u.Quantity = 1 * u.min):
        """
        Initialize the checker.

        Parameters
        ----------
        xml_path : str
            Path to the XML calendar file
        time_step : u.Quantity, optional
            Time resolution for visibility checking (default: 1 minute)
        """
        self.xml_path = Path(xml_path)
        self.time_step = time_step
        self.tree = None
        self.root = None
        self.tle_line1 = None
        self.tle_line2 = None
        self.vis = None
        self.checks: List[ObservationCheck] = []
        self.namespace = None  # Store namespace for later use

    def parse_calendar(self):
        """Parse the XML calendar and extract TLE information."""
        self.tree = ET.parse(self.xml_path)
        self.root = self.tree.getroot()

        # Detect and store namespace
        if self.root.tag.startswith("{"):
            self.namespace = self.root.tag.split("}")[0].strip("{")
        else:
            self.namespace = None

        # Extract TLE from Meta element
        meta = self.root.find("Meta")
        if meta is None and self.namespace:
            meta = self.root.find(f"{{{self.namespace}}}Meta")

        if meta is not None:
            self.tle_line1 = meta.get("TLE_Line1")
            self.tle_line2 = meta.get("TLE_Line2")
        else:
            raise ValueError(
                "Could not find Meta element with TLE data in XML"
            )

        if not self.tle_line1 or not self.tle_line2:
            raise ValueError("TLE data not found in Meta element")

        # Initialize Visibility object
        self.vis = Visibility(self.tle_line1, self.tle_line2)

        print(f"Loaded TLE for satellite: {self.vis.tle.satnum}")
        print(f"Orbital period: {self.vis.get_period():.2f}")

    def _find_element(self, parent, tag):
        """Helper to find element with or without namespace."""
        elem = parent.find(tag)
        if elem is None and self.namespace:
            elem = parent.find(f"{{{self.namespace}}}{tag}")
        return elem

    def _findall_elements(self, parent, tag):
        """Helper to findall elements with or without namespace."""
        elems = parent.findall(tag)
        if not elems and self.namespace:
            elems = parent.findall(f"{{{self.namespace}}}{tag}")
        return elems

    def _extract_observations(self) -> List[Dict]:
        """Extract all observations from the XML."""
        observations = []

        # Find all Visit elements
        visits = self._findall_elements(self.root, "Visit")

        if not visits:
            raise ValueError("No Visit elements found in XML")

        for visit in visits:
            visit_id_elem = self._find_element(visit, "ID")
            if visit_id_elem is None:
                print(f"Warning: Visit without ID found, skipping...")
                continue
            visit_id = visit_id_elem.text

            sequences = self._findall_elements(visit, "Observation_Sequence")

            for seq in sequences:
                seq_id_elem = self._find_element(seq, "ID")
                task_elem = self._find_element(seq, "Task")

                if seq_id_elem is None or task_elem is None:
                    print(
                        f"Warning: Incomplete sequence in Visit {visit_id}, skipping..."
                    )
                    continue

                seq_id = seq_id_elem.text
                task = task_elem.text

                obs_params = self._find_element(
                    seq, "Observational_Parameters"
                )
                if obs_params is None:
                    print(
                        f"Warning: No Observational_Parameters in Visit {visit_id} Seq {seq_id}, skipping..."
                    )
                    continue

                target_elem = self._find_element(obs_params, "Target")
                timing = self._find_element(obs_params, "Timing")
                boresight = self._find_element(obs_params, "Boresight")

                if target_elem is None or timing is None or boresight is None:
                    print(
                        f"Warning: Incomplete observation parameters in Visit {visit_id} Seq {seq_id}, skipping..."
                    )
                    continue

                target = target_elem.text

                start_elem = self._find_element(timing, "Start")
                stop_elem = self._find_element(timing, "Stop")

                if start_elem is None or stop_elem is None:
                    print(
                        f"Warning: Incomplete timing in Visit {visit_id} Seq {seq_id}, skipping..."
                    )
                    continue

                start = start_elem.text
                stop = stop_elem.text

                ra_elem = self._find_element(boresight, "RA")
                dec_elem = self._find_element(boresight, "DEC")

                if ra_elem is None or dec_elem is None:
                    print(
                        f"Warning: Incomplete boresight in Visit {visit_id} Seq {seq_id}, skipping..."
                    )
                    continue

                ra = float(ra_elem.text)
                dec = float(dec_elem.text)

                observations.append(
                    {
                        "visit_id": visit_id,
                        "sequence_id": seq_id,
                        "task": task,
                        "target": target,
                        "ra": ra,
                        "dec": dec,
                        "start": start,
                        "stop": stop,
                    }
                )

        if not observations:
            raise ValueError("No valid observations found in XML")

        return observations

    def check_observation_visibility(self, obs: Dict) -> ObservationCheck:
        """
        Check visibility for a single observation.

        Parameters
        ----------
        obs : Dict
            Observation dictionary with ra, dec, start, stop

        Returns
        -------
        ObservationCheck
            Results of the visibility check
        """
        # Parse times
        tstart = Time(obs["start"])
        tstop = Time(obs["stop"])
        duration = (tstop - tstart).to(u.min)

        # Create time array
        deltas = np.arange(0, duration.value, self.time_step.value) * u.min
        times = tstart + TimeDelta(deltas)

        # Create target coordinate
        target_coord = SkyCoord(ra=obs["ra"] * u.deg, dec=obs["dec"] * u.deg)

        # Check visibility
        visibility = self.vis.get_visibility(target_coord, times)

        # Analyze results
        is_fully_visible = np.all(visibility)
        visibility_fraction = np.sum(visibility) / len(visibility)
        num_violations = np.sum(~visibility)

        first_violation_time = None
        if not is_fully_visible:
            first_violation_idx = np.where(~visibility)[0][0]
            first_violation_time = times[first_violation_idx].iso

        return ObservationCheck(
            visit_id=obs["visit_id"],
            sequence_id=obs["sequence_id"],
            task=obs["task"],
            target=obs["target"],
            ra=obs["ra"],
            dec=obs["dec"],
            start=obs["start"],
            stop=obs["stop"],
            duration_minutes=duration.value,
            is_visible=is_fully_visible,
            visibility_fraction=visibility_fraction,
            first_violation_time=first_violation_time,
            num_violations=num_violations,
        )

    def check_all_observations(self, verbose: bool = True):
        """
        Check visibility for all observations in the calendar.

        Parameters
        ----------
        verbose : bool, optional
            Print progress during checking (default: True)
        """
        if self.vis is None:
            self.parse_calendar()

        observations = self._extract_observations()

        if verbose:
            print(
                f"\nChecking visibility for {len(observations)} observations..."
            )
            print("=" * 80)

        self.checks = []
        for i, obs in enumerate(observations, 1):
            check = self.check_observation_visibility(obs)
            self.checks.append(check)

            if verbose:
                print(f"{i:3d}. {check}")

        if verbose:
            print("=" * 80)
            self.print_summary()

    def print_summary(self):
        """Print summary statistics of visibility checks."""
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.is_visible)
        failed = total - passed

        print(f"\nSummary:")
        print(f"  Total observations: {total}")
        print(f"  Passed (fully visible): {passed} ({100*passed/total:.1f}%)")
        print(
            f"  Failed (visibility violations): {failed} ({100*failed/total:.1f}%)"
        )

        if failed > 0:
            print(f"\nFailed observations:")
            for check in self.checks:
                if not check.is_visible:
                    print(
                        f"  - Visit {check.visit_id} Seq {check.sequence_id}: "
                        f"{check.task} | {check.target} | "
                        f"Visibility: {check.visibility_fraction:.1%}"
                    )

    def get_failed_observations(self) -> List[ObservationCheck]:
        """Return list of observations that failed visibility check."""
        return [c for c in self.checks if not c.is_visible]

    def get_passed_observations(self) -> List[ObservationCheck]:
        """Return list of observations that passed visibility check."""
        return [c for c in self.checks if c.is_visible]

    def export_results(self, output_path: str):
        """
        Export results to a CSV file.

        Parameters
        ----------
        output_path : str
            Path for output CSV file
        """
        import csv

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Visit_ID",
                    "Sequence_ID",
                    "Task",
                    "Target",
                    "RA",
                    "Dec",
                    "Start",
                    "Stop",
                    "Duration_min",
                    "Passed",
                    "Visibility_Fraction",
                    "Num_Violations",
                    "First_Violation_Time",
                ]
            )

            for check in self.checks:
                writer.writerow(
                    [
                        check.visit_id,
                        check.sequence_id,
                        check.task,
                        check.target,
                        check.ra,
                        check.dec,
                        check.start,
                        check.stop,
                        check.duration_minutes,
                        check.is_visible,
                        check.visibility_fraction,
                        check.num_violations,
                        check.first_violation_time or "",
                    ]
                )

        print(f"\nResults exported to {output_path}")

    def plot_observation_visibility(
        self, visit_id: str, sequence_id: str, ax=None
    ):
        """
        Plot detailed visibility timeline for a specific observation.

        Parameters
        ----------
        visit_id : str
            Visit ID
        sequence_id : str
            Sequence ID within the visit
        ax : matplotlib.axes.Axes, optional
            Axes to plot on (creates new figure if None)
        """
        import matplotlib.pyplot as plt
        from astropy.visualization import time_support

        time_support()

        # Find the observation
        obs = None
        for check in self.checks:
            if check.visit_id == visit_id and check.sequence_id == sequence_id:
                obs = check
                break

        if obs is None:
            raise ValueError(
                f"Observation not found: Visit {visit_id} Seq {sequence_id}"
            )

        # Recalculate with finer resolution for plotting
        tstart = Time(obs.start)
        tstop = Time(obs.stop)
        duration = (tstop - tstart).to(u.min)
        deltas = np.arange(0, duration.value, 0.1) * u.min
        times = tstart + TimeDelta(deltas)

        target_coord = SkyCoord(ra=obs.ra * u.deg, dec=obs.dec * u.deg)
        visibility = self.vis.get_visibility(target_coord, times)

        # Get individual constraint information at start time
        constraints = self.vis.get_all_constraints(target_coord, tstart)
        separations = self.vis.get_separations(target_coord, tstart)

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(times.datetime, visibility, "o-", markersize=2)
        ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel("Time (UTC)")
        ax.set_ylabel("Visible")
        ax.set_title(
            f"Visit {visit_id} Seq {sequence_id}: {obs.task} | {obs.target}\n"
            f"RA={obs.ra:.4f}° Dec={obs.dec:.4f}°"
        )
        ax.grid(True, alpha=0.3)

        # Add constraint info as text
        constraint_text = "Constraints at start:\n"
        for body, passed in constraints.items():
            sep = separations[body]
            min_sep = getattr(self.vis, f"{body}_min")
            status = "✓" if passed else "✗"
            constraint_text += (
                f"{status} {body}: {sep:.1f} (min: {min_sep:.1f})\n"
            )

        ax.text(
            0.02,
            0.98,
            constraint_text,
            transform=ax.transAxes,
            verticalalignment="top",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        return ax


# Convenience function
def check_calendar_visibility(
    xml_path: str,
    time_step: u.Quantity = 1 * u.min,
    export_csv: Optional[str] = None,
) -> CalendarVisibilityChecker:
    """
    Convenience function to check visibility for an entire calendar.

    Parameters
    ----------
    xml_path : str
        Path to XML calendar file
    time_step : u.Quantity, optional
        Time resolution for checking (default: 1 minute)
    export_csv : str, optional
        If provided, export results to this CSV file

    Returns
    -------
    CalendarVisibilityChecker
        Checker object with all results

    Examples
    --------
    >>> checker = check_calendar_visibility('calendar.xml')
    >>> failed = checker.get_failed_observations()
    >>> if failed:
    >>>     checker.plot_observation_visibility(failed[0].visit_id,
    >>>                                         failed[0].sequence_id)
    """
    checker = CalendarVisibilityChecker(xml_path, time_step=time_step)
    checker.check_all_observations()

    if export_csv:
        checker.export_results(export_csv)

    return checker


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        xml_file = sys.argv[1]
    else:
        xml_file = "../src/commissioningscheduler/master_schedule.xml"

    # Check the calendar
    checker = check_calendar_visibility(
        xml_file, export_csv="visibility_check.csv"
    )

    # Plot any failed observations
    # failed = checker.get_failed_observations()
    # if failed:
    #     print(f"\nPlotting {len(failed)} failed observations...")
    #     import matplotlib.pyplot as plt

    #     for obs in failed[:5]:  # Plot first 5 failures
    #         fig, ax = plt.subplots(figsize=(12, 4))
    #         checker.plot_observation_visibility(
    #             obs.visit_id, obs.sequence_id, ax=ax
    #         )
    #         plt.savefig(
    #             f"visibility_fail_V{obs.visit_id}_S{obs.sequence_id}.png",
    #             dpi=150,
    #             bbox_inches="tight",
    #         )
    #         print(
    #             f"  Saved plot: visibility_fail_V{obs.visit_id}_S{obs.sequence_id}.png"
    #         )

    #     plt.show()
