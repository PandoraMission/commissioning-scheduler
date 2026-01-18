# shine.py
"""
Moonshine and Earthshine observation planning for Pandora spacecraft.

This module computes telescope pointing directions for shine observations:
- Moonshine: Point at specified angles around Moon's disk at specified distances from limb
- Earthshine: Point at specified distances from Earth's limb when spacecraft is at
  specified orbital positions
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union, List
from pathlib import Path

import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, get_sun, get_body
import astropy.units as u
from astropy import constants as const
from sgp4.api import Satrec, jday

logger = logging.getLogger(__name__)

# Constants
EARTH_RADIUS_KM = const.R_earth.to(u.km).value
MOON_RADIUS_KM = 1737.4  # km
SUN_AVOIDANCE_ANGLE = 91.0  # degrees from Sun center


@dataclass
class SpacecraftState:
    """Spacecraft state at a given time."""

    time: Time
    position_km: np.ndarray  # ECI J2000, shape (3,)
    velocity_km_s: np.ndarray  # ECI J2000, shape (3,)

    @property
    def altitude_km(self) -> float:
        """Altitude above Earth surface."""
        return np.linalg.norm(self.position_km) - EARTH_RADIUS_KM


@dataclass
class MoonshineResult:
    """Result from Moonshine pointing calculation."""

    ra_deg: float
    dec_deg: float
    time: Time
    angular_position_deg: float
    limb_separation_deg: float
    moon_visible: bool
    pointing_vector_eci: np.ndarray
    moon_position_eci: np.ndarray
    spacecraft_position_eci: np.ndarray


@dataclass
class EarthshineResult:
    """Result from Earthshine pointing calculation."""

    ra_deg: float
    dec_deg: float
    time: Time
    orbital_position_deg: float
    limb_separation_deg: float
    pointing_in_antisolar: bool
    sun_angle_deg: float
    pointing_vector_eci: np.ndarray
    spacecraft_position_eci: np.ndarray


class EphemerisProvider:
    """
    Provide spacecraft, Moon, Sun, and Earth ephemeris from multiple sources.

    Supports:
    - GMAT report files
    - SGP4 propagation from TLE
    - Astropy for Moon and Sun positions
    """

    def __init__(
        self,
        tle_line1: Optional[str] = None,
        tle_line2: Optional[str] = None,
        gmat_file: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize ephemeris provider.

        Args:
            tle_line1: First line of TLE
            tle_line2: Second line of TLE
            gmat_file: Path to GMAT report file
        """
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2
        self.gmat_file = Path(gmat_file) if gmat_file else None

        # Initialize SGP4 if TLE provided
        self.satellite = None
        if tle_line1 and tle_line2:
            self.satellite = Satrec.twoline2rv(tle_line1, tle_line2)
            logger.info("Initialized SGP4 propagator from TLE")

        # Load GMAT file if provided
        self.gmat_data = None
        if self.gmat_file and self.gmat_file.exists():
            self._load_gmat_file()

    def _load_gmat_file(self):
        """Load and parse GMAT report file."""
        logger.info(f"Loading GMAT file: {self.gmat_file}")

        # Read file, skip header
        with open(self.gmat_file, "r") as f:
            lines = f.readlines()

        # Find data start (after header line with column names)
        data_start = 0
        for i, line in enumerate(lines):
            if "Earth.UTCModJulian" in line:
                data_start = i + 1
                break

        # Parse data
        data = []
        for line in lines[data_start:]:
            if line.strip():
                values = line.split()
                data.append([float(v) for v in values])

        data = np.array(data)

        # Store columns (based on your header)
        self.gmat_data = {
            "utc_mod_julian": data[:, 0],
            "pandora_x": data[:, 3],
            "pandora_y": data[:, 4],
            "pandora_z": data[:, 5],
            "moon_x": data[:, 6],
            "moon_y": data[:, 7],
            "moon_z": data[:, 8],
            "sun_x": data[:, 9],
            "sun_y": data[:, 10],
            "sun_z": data[:, 11],
        }

        logger.info(f"Loaded GMAT data: {len(data)} time steps")

    def get_spacecraft_state(self, time: Time) -> SpacecraftState:
        """
        Get spacecraft state at given time.

        Args:
            time: Time for state

        Returns:
            SpacecraftState
        """
        # Try GMAT file first if available
        if self.gmat_data is not None:
            return self._get_spacecraft_from_gmat(time)

        # Fall back to SGP4
        if self.satellite is not None:
            return self._get_spacecraft_from_sgp4(time)

        raise ValueError(
            "No ephemeris source available (need TLE or GMAT file)"
        )

    def _get_spacecraft_from_gmat(self, time: Time) -> SpacecraftState:
        """Get spacecraft state from GMAT file interpolation."""
        # Convert to Modified Julian Date
        mjd = time.mjd

        # Find bracketing indices
        times = self.gmat_data["utc_mod_julian"]
        if mjd < times[0] or mjd > times[-1]:
            raise ValueError(f"Time {time.iso} outside GMAT file range")

        # Linear interpolation
        idx = np.searchsorted(times, mjd)
        if idx == 0:
            idx = 1

        t0, t1 = times[idx - 1], times[idx]
        alpha = (mjd - t0) / (t1 - t0)

        # Interpolate position
        pos = np.array(
            [
                np.interp(mjd, times, self.gmat_data["pandora_x"]),
                np.interp(mjd, times, self.gmat_data["pandora_y"]),
                np.interp(mjd, times, self.gmat_data["pandora_z"]),
            ]
        )

        # Estimate velocity from finite difference
        if idx < len(times) - 1:
            pos_next = np.array(
                [
                    self.gmat_data["pandora_x"][idx],
                    self.gmat_data["pandora_y"][idx],
                    self.gmat_data["pandora_z"][idx],
                ]
            )
            dt = (t1 - t0) * 86400  # Convert days to seconds
            vel = (pos_next - pos) / dt
        else:
            pos_prev = np.array(
                [
                    self.gmat_data["pandora_x"][idx - 1],
                    self.gmat_data["pandora_y"][idx - 1],
                    self.gmat_data["pandora_z"][idx - 1],
                ]
            )
            dt = (t1 - t0) * 86400
            vel = (pos - pos_prev) / dt

        return SpacecraftState(time=time, position_km=pos, velocity_km_s=vel)

    def _get_spacecraft_from_sgp4(self, time: Time) -> SpacecraftState:
        """Get spacecraft state from SGP4 propagation."""
        # Convert astropy Time to Julian date
        jd, fr = jday(
            time.datetime.year,
            time.datetime.month,
            time.datetime.day,
            time.datetime.hour,
            time.datetime.minute,
            time.datetime.second + time.datetime.microsecond / 1e6,
        )

        # Propagate
        error, pos_teme, vel_teme = self.satellite.sgp4(jd, fr)

        if error != 0:
            raise RuntimeError(f"SGP4 propagation error: {error}")

        # Convert TEME to ECI J2000 (approximate - for better accuracy use astropy)
        # For now, treat as approximately the same
        pos = np.array(pos_teme)
        vel = np.array(vel_teme)

        return SpacecraftState(time=time, position_km=pos, velocity_km_s=vel)

    def get_moon_position(self, time: Time) -> np.ndarray:
        """
        Get Moon position in ECI J2000.

        Args:
            time: Time for position

        Returns:
            Position vector in km, shape (3,)
        """
        # Try GMAT file first
        if self.gmat_data is not None:
            mjd = time.mjd
            times = self.gmat_data["utc_mod_julian"]

            if mjd >= times[0] and mjd <= times[-1]:
                pos = np.array(
                    [
                        np.interp(mjd, times, self.gmat_data["moon_x"]),
                        np.interp(mjd, times, self.gmat_data["moon_y"]),
                        np.interp(mjd, times, self.gmat_data["moon_z"]),
                    ]
                )
                return pos

        # Fall back to astropy
        moon = get_body("moon", time, location=None)
        # Convert to GCRS and extract position
        moon_gcrs = moon.transform_to("gcrs")
        pos_km = moon_gcrs.cartesian.xyz.to(u.km).value

        return pos_km

    def get_sun_position(self, time: Time) -> np.ndarray:
        """
        Get Sun position in ECI J2000.

        Args:
            time: Time for position

        Returns:
            Position vector in km, shape (3,)
        """
        # Try GMAT file first
        if self.gmat_data is not None:
            mjd = time.mjd
            times = self.gmat_data["utc_mod_julian"]

            if mjd >= times[0] and mjd <= times[-1]:
                pos = np.array(
                    [
                        np.interp(mjd, times, self.gmat_data["sun_x"]),
                        np.interp(mjd, times, self.gmat_data["sun_y"]),
                        np.interp(mjd, times, self.gmat_data["sun_z"]),
                    ]
                )
                return pos

        # Fall back to astropy
        sun = get_sun(time)
        sun_gcrs = sun.transform_to("gcrs")
        pos_km = sun_gcrs.cartesian.xyz.to(u.km).value

        return pos_km


class MoonshinePointing:
    """
    Calculate telescope pointing for Moonshine observations.

    Computes RA/Dec pointing at specified angular positions around Moon's disk
    at specified distances from the limb.
    """

    def __init__(self, ephemeris: EphemerisProvider):
        """
        Initialize Moonshine pointing calculator.

        Args:
            ephemeris: Ephemeris provider
        """
        self.ephemeris = ephemeris

    def calculate_pointing(
        self,
        time: Time,
        angular_position_deg: float,
        limb_separation_deg: float,
        check_earth_blockage: bool = True,
        earth_avoidance_deg: float = 20.0,
    ) -> MoonshineResult:
        """
        Calculate pointing for Moonshine observation.

        Args:
            time: UTC time for observation
            angular_position_deg: Angular position around Moon (0=North, 90=right, etc.)
            limb_separation_deg: Angular separation from Moon's limb (degrees)
            check_earth_blockage: Check if Moon is blocked by Earth
            earth_avoidance_deg: Minimum angle from Earth limb for Moon visibility

        Returns:
            MoonshineResult
        """
        # Get spacecraft and Moon positions
        sc_state = self.ephemeris.get_spacecraft_state(time)
        moon_pos = self.ephemeris.get_moon_position(time)

        sc_pos = sc_state.position_km

        # Vector from spacecraft to Moon
        sc_to_moon = moon_pos - sc_pos
        distance_to_moon = np.linalg.norm(sc_to_moon)
        sc_to_moon_unit = sc_to_moon / distance_to_moon

        # Calculate Moon's angular radius as seen from spacecraft
        moon_angular_radius_deg = np.degrees(
            np.arcsin(MOON_RADIUS_KM / distance_to_moon)
        )

        # Total angle from Moon center to pointing
        total_angle_deg = moon_angular_radius_deg + limb_separation_deg

        # Define reference frame around Moon
        # "North" is toward Earth's North pole
        earth_north_pole = np.array(
            [0, 0, 1]
        )  # In ECI, Z-axis points to North pole

        # Create orthonormal frame
        # Primary axis: spacecraft to Moon
        z_axis = sc_to_moon_unit

        # Project North pole onto plane perpendicular to z_axis
        north_projection = (
            earth_north_pole - np.dot(earth_north_pole, z_axis) * z_axis
        )
        north_projection_norm = np.linalg.norm(north_projection)

        if north_projection_norm < 1e-6:
            # Moon is near pole, use arbitrary reference
            if abs(z_axis[0]) < 0.9:
                x_axis = np.array([1, 0, 0])
            else:
                x_axis = np.array([0, 1, 0])
            x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
            x_axis = x_axis / np.linalg.norm(x_axis)
        else:
            # "Up" direction (toward North pole)
            x_axis = north_projection / north_projection_norm

        # Complete right-handed frame
        y_axis = np.cross(z_axis, x_axis)

        # Convert angular position to radians
        theta = np.radians(angular_position_deg)

        # Rotate around Moon in the x-y plane of the reference frame
        # This gives us a direction in the plane perpendicular to sc-to-Moon vector
        radial_direction = np.cos(theta) * x_axis + np.sin(theta) * y_axis

        # Now tilt away from Moon center by total_angle_deg
        # The pointing vector is a rotation of sc_to_moon_unit toward radial_direction
        total_angle_rad = np.radians(total_angle_deg)

        pointing_vector = (
            np.cos(total_angle_rad) * sc_to_moon_unit
            + np.sin(total_angle_rad) * radial_direction
        )
        pointing_vector = pointing_vector / np.linalg.norm(pointing_vector)

        # Convert pointing vector to RA/Dec
        ra_deg, dec_deg = self._vector_to_radec(pointing_vector)

        # Check Moon visibility if requested
        moon_visible = True
        if check_earth_blockage:
            moon_visible = self._check_moon_visibility(
                sc_pos, moon_pos, earth_avoidance_deg
            )

        return MoonshineResult(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            time=time,
            angular_position_deg=angular_position_deg,
            limb_separation_deg=limb_separation_deg,
            moon_visible=moon_visible,
            pointing_vector_eci=pointing_vector,
            moon_position_eci=moon_pos,
            spacecraft_position_eci=sc_pos,
        )

    def _vector_to_radec(self, vector: np.ndarray) -> Tuple[float, float]:
        """
        Convert ECI vector to RA/Dec.

        Args:
            vector: Unit vector in ECI J2000

        Returns:
            Tuple of (RA, Dec) in degrees
        """
        # Normalize
        v = vector / np.linalg.norm(vector)

        # RA is angle in x-y plane from x-axis
        ra = np.degrees(np.arctan2(v[1], v[0]))
        if ra < 0:
            ra += 360

        # Dec is angle from x-y plane
        dec = np.degrees(np.arcsin(v[2]))

        return ra, dec

    def _check_moon_visibility(
        self,
        sc_pos: np.ndarray,
        moon_pos: np.ndarray,
        earth_avoidance_deg: float,
    ) -> bool:
        """
        Check if Moon is visible from spacecraft (not blocked by Earth).

        Args:
            sc_pos: Spacecraft position (km)
            moon_pos: Moon position (km)
            earth_avoidance_deg: Minimum angle from Earth limb

        Returns:
            True if Moon is visible
        """
        # Vector from spacecraft to Moon
        sc_to_moon = moon_pos - sc_pos
        sc_to_moon_unit = sc_to_moon / np.linalg.norm(sc_to_moon)

        # Vector from spacecraft to Earth center (negative of position)
        sc_to_earth = -sc_pos
        sc_to_earth_unit = sc_to_earth / np.linalg.norm(sc_to_earth)

        # Angle between Moon direction and Earth direction
        angle = np.degrees(
            np.arccos(
                np.clip(np.dot(sc_to_moon_unit, sc_to_earth_unit), -1, 1)
            )
        )

        # Earth's angular radius as seen from spacecraft
        sc_altitude = np.linalg.norm(sc_pos)
        earth_angular_radius = np.degrees(
            np.arcsin(EARTH_RADIUS_KM / sc_altitude)
        )

        # Moon is blocked if angle is less than Earth radius + avoidance
        min_angle = earth_angular_radius + earth_avoidance_deg

        return angle >= min_angle


class EarthshinePointing:
    """
    Calculate telescope pointing for Earthshine observations.

    Finds orbital positions and computes RA/Dec pointing at specified distances
    from Earth's limb in the antisolar direction.
    """

    def __init__(self, ephemeris: EphemerisProvider):
        """
        Initialize Earthshine pointing calculator.

        Args:
            ephemeris: Ephemeris provider
        """
        self.ephemeris = ephemeris

        # Cache for orbital period determination
        self._orbital_period_minutes = None
        self._period_reference_time = None

    def calculate_pointing(
        self,
        start_time: Time,
        orbital_position_deg: float,
        limb_separation_deg: float,
        max_search_orbits: int = 3,
        position_tolerance_deg: float = 5.0,
        sun_avoidance_deg: float = SUN_AVOIDANCE_ANGLE,
    ) -> EarthshineResult:
        """
        Calculate pointing for Earthshine observation.

        For sun-synchronous orbit, we find when spacecraft reaches the desired
        orbital position, then point in the antisolar direction at the specified
        angle from Earth's limb.

        Args:
            start_time: Earliest time to search from
            orbital_position_deg: Desired orbital position (0=North, 90/270=equator, 180=South)
            limb_separation_deg: Angular separation from Earth's limb (degrees)
            max_search_orbits: Maximum number of orbits to search
            position_tolerance_deg: Tolerance for orbital position match
            sun_avoidance_deg: Minimum angle from Sun center

        Returns:
            EarthshineResult
        """
        # Find when spacecraft reaches desired orbital position
        target_time, actual_position = self._find_orbital_position(
            start_time,
            orbital_position_deg,
            max_search_orbits,
            position_tolerance_deg,
        )

        if target_time is None:
            raise ValueError(
                f"Could not find orbital position {orbital_position_deg}° "
                f"within {max_search_orbits} orbits"
            )

        # Get spacecraft state at target time
        sc_state = self.ephemeris.get_spacecraft_state(target_time)
        sc_pos = sc_state.position_km

        # Get Sun position
        sun_pos = self.ephemeris.get_sun_position(target_time)

        # Calculate pointing in antisolar direction at N degrees from Earth limb
        pointing_vector, sun_angle_deg = self._calculate_antisolar_pointing(
            sc_pos, sun_pos, limb_separation_deg
        )

        # Convert to RA/Dec
        ra_deg, dec_deg = self._vector_to_radec(pointing_vector)

        # Verify constraints
        in_antisolar = sun_angle_deg > 90.0
        sun_ok = sun_angle_deg >= sun_avoidance_deg

        if not sun_ok:
            logger.warning(
                f"Pointing violates Sun avoidance: {sun_angle_deg:.1f}° < {sun_avoidance_deg}°"
            )

        return EarthshineResult(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            time=target_time,
            orbital_position_deg=actual_position,
            limb_separation_deg=limb_separation_deg,
            pointing_in_antisolar=in_antisolar and sun_ok,
            sun_angle_deg=sun_angle_deg,
            pointing_vector_eci=pointing_vector,
            spacecraft_position_eci=sc_pos,
        )

    def _calculate_antisolar_pointing(
        self,
        sc_pos: np.ndarray,
        sun_pos: np.ndarray,
        limb_separation_deg: float,
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate pointing in antisolar direction at specified angle from Earth limb.

        Strategy:
        1. Find the antisolar direction (away from Sun)
        2. Calculate the cone of directions that are N degrees from Earth limb
        3. Find the intersection point closest to antisolar direction

        Args:
            sc_pos: Spacecraft position (km)
            sun_pos: Sun position (km)
            limb_separation_deg: Degrees from Earth limb

        Returns:
            Tuple of (pointing_vector, sun_angle_deg)
        """
        # Calculate Earth's angular radius as seen from spacecraft
        sc_distance = np.linalg.norm(sc_pos)
        earth_angular_radius_deg = np.degrees(
            np.arcsin(EARTH_RADIUS_KM / sc_distance)
        )

        # Total angle from Earth center to pointing
        total_angle_deg = earth_angular_radius_deg + limb_separation_deg
        total_angle_rad = np.radians(total_angle_deg)

        # Antisolar direction
        sc_to_sun = sun_pos - sc_pos
        antisolar_unit = -sc_to_sun / np.linalg.norm(sc_to_sun)

        # Radial direction (away from Earth)
        radial_unit = sc_pos / sc_distance

        # The pointing must be on a cone of half-angle total_angle_deg around radial_unit
        # We want the point on this cone closest to the antisolar direction

        # Project antisolar onto the plane perpendicular to radial
        antisolar_perp = (
            antisolar_unit - np.dot(antisolar_unit, radial_unit) * radial_unit
        )
        antisolar_perp_norm = np.linalg.norm(antisolar_perp)

        if antisolar_perp_norm < 1e-6:
            # Antisolar is aligned with radial - use arbitrary perpendicular direction
            # This shouldn't happen for realistic geometries
            if abs(radial_unit[0]) < 0.9:
                perp_dir = np.array([1, 0, 0])
            else:
                perp_dir = np.array([0, 1, 0])
            antisolar_perp = (
                perp_dir - np.dot(perp_dir, radial_unit) * radial_unit
            )
            antisolar_perp = antisolar_perp / np.linalg.norm(antisolar_perp)
        else:
            antisolar_perp = antisolar_perp / antisolar_perp_norm

        # Construct pointing on cone in direction of antisolar
        pointing_vector = (
            np.cos(total_angle_rad) * radial_unit
            + np.sin(total_angle_rad) * antisolar_perp
        )
        pointing_vector = pointing_vector / np.linalg.norm(pointing_vector)

        # Calculate actual Sun angle
        sun_angle_deg = np.degrees(
            np.arccos(np.clip(np.dot(pointing_vector, -antisolar_unit), -1, 1))
        )

        return pointing_vector, sun_angle_deg

    def _find_orbital_position(
        self,
        start_time: Time,
        target_position_deg: float,
        max_orbits: int,
        tolerance_deg: float,
    ) -> Tuple[Optional[Time], Optional[float]]:
        """
        Find next time when spacecraft reaches target orbital position.

        Args:
            start_time: Start search from this time
            target_position_deg: Target orbital position (0-360)
            max_orbits: Maximum orbits to search
            tolerance_deg: Position tolerance

        Returns:
            Tuple of (time, actual_position) or (None, None) if not found
        """
        # Determine orbital period if not cached
        if self._orbital_period_minutes is None:
            self._determine_orbital_period(start_time)

        # Search with 1-minute time steps
        time_step_minutes = 1.0
        max_steps = int(
            max_orbits * self._orbital_period_minutes / time_step_minutes
        )

        # Normalize target position
        target_position_deg = target_position_deg % 360.0

        prev_position = None
        prev_time = None

        for i in range(max_steps):
            current_time = start_time + i * time_step_minutes * u.min

            try:
                sc_state = self.ephemeris.get_spacecraft_state(current_time)
                current_position = self._get_orbital_position(sc_state)

                # Check if we're within tolerance
                delta = self._angular_difference(
                    current_position, target_position_deg
                )

                if abs(delta) <= tolerance_deg:
                    logger.info(
                        f"Found orbital position {current_position:.1f}° "
                        f"(target {target_position_deg:.1f}°) at {current_time.iso}"
                    )
                    return current_time, current_position

                # Check if we crossed the target
                if prev_position is not None:
                    prev_delta = self._angular_difference(
                        prev_position, target_position_deg
                    )

                    # Crossed if deltas have opposite signs and we're moving forward
                    if prev_delta < 0 and delta > 0:
                        # Refine with interpolation
                        refined_time = self._refine_crossing(
                            prev_time,
                            current_time,
                            prev_position,
                            current_position,
                            target_position_deg,
                        )
                        refined_state = self.ephemeris.get_spacecraft_state(
                            refined_time
                        )
                        refined_position = self._get_orbital_position(
                            refined_state
                        )

                        logger.info(
                            f"Found orbital position {refined_position:.1f}° "
                            f"(target {target_position_deg:.1f}°) at {refined_time.iso}"
                        )
                        return refined_time, refined_position

                prev_position = current_position
                prev_time = current_time

            except Exception as e:
                logger.warning(f"Error at time {current_time.iso}: {e}")
                continue

        logger.warning(
            f"Could not find orbital position {target_position_deg}° "
            f"within {max_orbits} orbits"
        )
        return None, None

    def _determine_orbital_period(self, reference_time: Time):
        """
        Determine orbital period by tracking latitude changes.

        Args:
            reference_time: Reference time for period determination
        """
        logger.info("Determining orbital period...")

        # Get initial position
        initial_state = self.ephemeris.get_spacecraft_state(reference_time)
        initial_lat = self._get_latitude(initial_state.position_km)

        # Search for return to same latitude with same velocity direction
        # This indicates one complete orbit
        time_step = 1.0  # minutes
        max_steps = 200  # ~3 hours max

        prev_lat = initial_lat
        ascending_initially = initial_state.velocity_km_s[2] > 0

        for i in range(1, max_steps):
            current_time = reference_time + i * time_step * u.min
            current_state = self.ephemeris.get_spacecraft_state(current_time)
            current_lat = self._get_latitude(current_state.position_km)
            ascending_now = current_state.velocity_km_s[2] > 0

            # Check if we crossed the initial latitude going the same direction
            if ascending_now == ascending_initially:
                if (prev_lat < initial_lat <= current_lat) or (
                    prev_lat > initial_lat >= current_lat
                ):
                    # Found approximate period
                    period_minutes = i * time_step
                    self._orbital_period_minutes = period_minutes
                    self._period_reference_time = reference_time
                    logger.info(
                        f"Orbital period: {period_minutes:.2f} minutes"
                    )
                    return

            prev_lat = current_lat

        # Fallback to nominal LEO period
        self._orbital_period_minutes = 95.0
        logger.warning("Could not determine period, using 95 minutes")

    def _get_latitude(self, position: np.ndarray) -> float:
        """Get geodetic latitude from ECI position."""
        r = np.linalg.norm(position)
        lat_rad = np.arcsin(position[2] / r)
        return np.degrees(lat_rad)

    def _get_orbital_position(self, sc_state: SpacecraftState) -> float:
        """
        Calculate orbital position in degrees (0-360).

        Position is measured as the angle around the orbit, where 0° corresponds
        to the northernmost point (maximum latitude).

        For a sun-synchronous orbit:
        - 0° = Maximum northern latitude (over North pole region)
        - 90° = Descending equator crossing
        - 180° = Maximum southern latitude (over South pole region)
        - 270° = Ascending equator crossing

        Args:
            sc_state: Spacecraft state

        Returns:
            Orbital position in degrees (0-360)
        """
        pos = sc_state.position_km
        vel = sc_state.velocity_km_s

        # Calculate orbital plane normal (angular momentum direction)
        h = np.cross(pos, vel)
        h_unit = h / np.linalg.norm(h)

        # For position in orbit, we use the argument of latitude
        # This is the angle from ascending node measured in the orbital plane

        # Ascending node is where orbit crosses equator going north
        # It's the intersection of orbital plane with equatorial plane
        z_axis = np.array([0, 0, 1])  # North pole direction

        # Node vector (perpendicular to both z and h)
        n = np.cross(z_axis, h_unit)
        n_norm = np.linalg.norm(n)

        if n_norm < 1e-6:
            # Orbit is polar - use x-axis as reference
            n = np.array([1, 0, 0])
            n_norm = 1.0

        n_unit = n / n_norm

        # Calculate argument of latitude
        # This is angle from ascending node to current position
        r_unit = pos / np.linalg.norm(pos)

        # Components in orbital frame
        cos_u = np.dot(r_unit, n_unit)

        # Third axis of orbital frame (completes right-handed system)
        n_perp = np.cross(h_unit, n_unit)
        sin_u = np.dot(r_unit, n_perp)

        # Argument of latitude
        u = np.degrees(np.arctan2(sin_u, cos_u))
        if u < 0:
            u += 360

        # Now determine offset so that 0° is at northernmost point
        # The northernmost point is where latitude is maximum
        # This occurs where position is most aligned with z-axis

        # For an inclined orbit, max latitude occurs at argument of latitude = 90°
        # (or 270° for min latitude)
        # For sun-synchronous orbit with inclination ~98°, the ascending node
        # is where we cross equator going north, so max north is at u = 90°

        # Adjust so 0° = maximum northern latitude
        orbital_position = (u - 90.0) % 360.0

        return orbital_position

    def _angular_difference(self, angle1: float, angle2: float) -> float:
        """
        Calculate signed angular difference (angle1 - angle2).

        Returns value in range [-180, 180].
        """
        diff = (angle1 - angle2) % 360.0
        if diff > 180:
            diff -= 360
        return diff

    def _refine_crossing(
        self, time1: Time, time2: Time, pos1: float, pos2: float, target: float
    ) -> Time:
        """
        Refine crossing time using linear interpolation.

        Args:
            time1: Earlier time
            time2: Later time
            pos1: Position at time1
            pos2: Position at time2
            target: Target position

        Returns:
            Refined time
        """
        # Linear interpolation
        delta1 = self._angular_difference(target, pos1)
        delta2 = self._angular_difference(target, pos2)

        # Interpolation fraction
        if abs(delta2 - delta1) < 1e-6:
            alpha = 0.5
        else:
            alpha = delta1 / (delta1 - delta2)

        # Clamp to valid range
        alpha = max(0, min(1, alpha))

        refined_time = time1 + alpha * (time2 - time1)
        return refined_time

    def _vector_to_radec(self, vector: np.ndarray) -> Tuple[float, float]:
        """Convert ECI vector to RA/Dec."""
        v = vector / np.linalg.norm(vector)
        ra = np.degrees(np.arctan2(v[1], v[0]))
        if ra < 0:
            ra += 360
        dec = np.degrees(np.arcsin(np.clip(v[2], -1, 1)))
        return ra, dec


# Convenience functions for common use cases


def compute_moonshine_pointing(
    time: Time,
    angular_position_deg: float,
    limb_separation_deg: float,
    tle_line1: Optional[str] = None,
    tle_line2: Optional[str] = None,
    gmat_file: Optional[str] = None,
    check_earth_blockage: bool = True,
) -> MoonshineResult:
    """
    Convenience function to compute Moonshine pointing.

    Args:
        time: UTC time for observation
        angular_position_deg: Angular position around Moon (0=North, 90=right, etc.)
        limb_separation_deg: Angular separation from Moon's limb
        tle_line1: First line of TLE
        tle_line2: Second line of TLE
        gmat_file: Path to GMAT file
        check_earth_blockage: Check if Moon is visible

    Returns:
        MoonshineResult
    """
    ephemeris = EphemerisProvider(tle_line1, tle_line2, gmat_file)
    calculator = MoonshinePointing(ephemeris)
    return calculator.calculate_pointing(
        time, angular_position_deg, limb_separation_deg, check_earth_blockage
    )


def compute_earthshine_pointing(
    start_time: Time,
    orbital_position_deg: float,
    limb_separation_deg: float,
    tle_line1: Optional[str] = None,
    tle_line2: Optional[str] = None,
    gmat_file: Optional[str] = None,
) -> EarthshineResult:
    """
    Convenience function to compute Earthshine pointing.

    Args:
        start_time: Earliest time to search from
        orbital_position_deg: Desired orbital position (0=North, etc.)
        limb_separation_deg: Angular separation from Earth's limb
        tle_line1: First line of TLE
        tle_line2: Second line of TLE
        gmat_file: Path to GMAT file

    Returns:
        EarthshineResult
    """
    ephemeris = EphemerisProvider(tle_line1, tle_line2, gmat_file)
    calculator = EarthshinePointing(ephemeris)
    return calculator.calculate_pointing(
        start_time, orbital_position_deg, limb_separation_deg
    )
