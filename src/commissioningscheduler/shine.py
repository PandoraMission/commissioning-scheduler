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
    from Earth's limb.
    """

    def __init__(self, ephemeris: EphemerisProvider):
        """
        Initialize Earthshine pointing calculator.

        Args:
            ephemeris: Ephemeris provider
        """
        self.ephemeris = ephemeris

        # Calculate reference orbital position (northernmost point)
        # This is done once to establish consistent reference
        self._orbital_reference = None

    def calculate_pointing(
        self,
        start_time: Time,
        orbital_position_deg: float,
        limb_separation_deg: float,
        max_search_orbits: int = 3,
        position_tolerance_deg: float = 5.0,
    ) -> EarthshineResult:
        """
        Calculate pointing for Earthshine observation.

        Finds the next time after start_time when spacecraft reaches the desired
        orbital position, then computes pointing.

        Args:
            start_time: Earliest time to search from
            orbital_position_deg: Desired orbital position (0=North, 90/270=equator, 180=South)
            limb_separation_deg: Angular separation from Earth's limb (degrees)
            max_search_orbits: Maximum number of orbits to search
            position_tolerance_deg: Tolerance for orbital position match

        Returns:
            EarthshineResult
        """
        # Find when spacecraft reaches desired orbital position
        target_time = self._find_orbital_position(
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

        # Calculate Earth's angular radius as seen from spacecraft
        sc_distance = np.linalg.norm(sc_pos)
        earth_angular_radius_deg = np.degrees(
            np.arcsin(EARTH_RADIUS_KM / sc_distance)
        )

        # Total angle from Earth center to pointing
        total_angle_deg = earth_angular_radius_deg + limb_separation_deg

        # Define pointing direction
        # Point perpendicular to Earth limb, away from Earth center
        # This is in the orbital plane, perpendicular to radial direction

        # Get orbital plane normal (angular momentum direction)
        h = np.cross(sc_pos, sc_state.velocity_km_s)
        h_unit = h / np.linalg.norm(h)

        # Radial direction (away from Earth)
        radial_unit = sc_pos / sc_distance

        # Tangential direction in orbital plane
        tangent_unit = np.cross(h_unit, radial_unit)

        # Pointing vector: tilt from radial by total_angle_deg in direction of tangent
        total_angle_rad = np.radians(total_angle_deg)
        pointing_vector = (
            np.cos(total_angle_rad) * radial_unit
            + np.sin(total_angle_rad) * tangent_unit
        )
        pointing_vector = pointing_vector / np.linalg.norm(pointing_vector)

        # Convert to RA/Dec
        ra_deg, dec_deg = self._vector_to_radec(pointing_vector)

        # Check if pointing is in antisolar hemisphere and Sun avoidance
        sun_pos = self.ephemeris.get_sun_position(target_time)
        sun_angle_deg, in_antisolar = self._check_sun_constraints(
            pointing_vector, sc_pos, sun_pos
        )

        # Get actual orbital position achieved
        actual_position_deg = self._get_orbital_position(sc_state)

        return EarthshineResult(
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            time=target_time,
            orbital_position_deg=actual_position_deg,
            limb_separation_deg=limb_separation_deg,
            pointing_in_antisolar=in_antisolar,
            sun_angle_deg=sun_angle_deg,
            pointing_vector_eci=pointing_vector,
            spacecraft_position_eci=sc_pos,
        )

    def _find_orbital_position(
        self,
        start_time: Time,
        target_position_deg: float,
        max_orbits: int,
        tolerance_deg: float,
    ) -> Optional[Time]:
        """
        Find next time when spacecraft reaches target orbital position.

        Args:
            start_time: Start search from this time
            target_position_deg: Target orbital position (0-360)
            max_orbits: Maximum orbits to search
            tolerance_deg: Position tolerance

        Returns:
            Time when position is reached, or None if not found
        """
        # Estimate orbital period (assume ~95 minutes for LEO)
        orbital_period_minutes = 95.0

        # Search with 1-minute time steps
        time_step_minutes = 1.0
        max_steps = int(
            max_orbits * orbital_period_minutes / time_step_minutes
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
                delta = abs(current_position - target_position_deg)
                if delta > 180:
                    delta = 360 - delta

                if delta <= tolerance_deg:
                    logger.info(
                        f"Found orbital position {current_position:.1f}° "
                        f"(target {target_position_deg:.1f}°) at {current_time.iso}"
                    )
                    return current_time

                # Check if we crossed the target (for better accuracy)
                if prev_position is not None:
                    crossed = self._check_crossing(
                        prev_position, current_position, target_position_deg
                    )
                    if crossed:
                        # Refine with interpolation
                        refined_time = self._refine_crossing(
                            prev_time, current_time, target_position_deg
                        )
                        return refined_time

                prev_position = current_position
                prev_time = current_time

            except Exception as e:
                logger.warning(f"Error at time {current_time.iso}: {e}")
                continue

        logger.warning(
            f"Could not find orbital position {target_position_deg}° "
            f"within {max_orbits} orbits"
        )
        return None

    def _get_orbital_position(self, sc_state: SpacecraftState) -> float:
        """
        Calculate orbital position in degrees (0-360).

        Position is defined relative to the northernmost point of the orbit.

        Args:
            sc_state: Spacecraft state

        Returns:
            Orbital position in degrees (0 = northernmost point)
        """
        # Calculate argument of latitude (angle from ascending node in orbital plane)
        pos = sc_state.position_km
        vel = sc_state.velocity_km_s

        # Orbital plane normal
        h = np.cross(pos, vel)
        h_unit = h / np.linalg.norm(h)

        # Ascending node direction (where orbit crosses equator going north)
        # This is the intersection of orbital plane with equatorial plane
        z_axis = np.array([0, 0, 1])  # Earth's pole
        n = np.cross(z_axis, h_unit)
        n_norm = np.linalg.norm(n)

        if n_norm < 1e-6:
            # Polar orbit - use arbitrary reference
            n = np.array([1, 0, 0])
        else:
            n = n / n_norm

        # Calculate angle from ascending node to spacecraft position
        # Project position into orbital plane coordinate system
        r_unit = pos / np.linalg.norm(pos)

        # Angle in orbital plane
        cos_angle = np.dot(r_unit, n)
        sin_angle = np.dot(r_unit, np.cross(h_unit, n))
        angle = np.degrees(np.arctan2(sin_angle, cos_angle))

        if angle < 0:
            angle += 360

        # Find northernmost point offset (done once for reference)
        if self._orbital_reference is None:
            self._orbital_reference = self._find_northernmost_point_offset(
                sc_state
            )

        # Adjust so 0° is at northernmost point
        position = (angle - self._orbital_reference) % 360.0

        return position

    def _find_northernmost_point_offset(
        self, sc_state: SpacecraftState
    ) -> float:
        """
        Find the angle offset to the northernmost point of the orbit.

        Args:
            sc_state: Sample spacecraft state

        Returns:
            Angle offset in degrees
        """
        # Sample positions around orbit
        pos = sc_state.position_km
        vel = sc_state.velocity_km_s

        # Orbital elements
        h = np.cross(pos, vel)
        h_unit = h / np.linalg.norm(h)

        # Find point with maximum Z coordinate (northernmost)
        # This occurs when position vector is perpendicular to h and points north
        z_axis = np.array([0, 0, 1])

        # Northernmost direction in orbital plane
        north_in_plane = z_axis - np.dot(z_axis, h_unit) * h_unit
        north_in_plane = north_in_plane / np.linalg.norm(north_in_plane)

        # Find ascending node
        n = np.cross(z_axis, h_unit)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-6:
            return 0.0
        n = n / n_norm

        # Angle from ascending node to northernmost point
        cos_angle = np.dot(north_in_plane, n)
        sin_angle = np.dot(north_in_plane, np.cross(h_unit, n))
        offset = np.degrees(np.arctan2(sin_angle, cos_angle))

        if offset < 0:
            offset += 360

        return offset

    def _check_crossing(self, pos1: float, pos2: float, target: float) -> bool:
        """Check if orbital position crossed target between two measurements."""

        # Normalize to handle 360° wraparound
        def normalize_delta(delta):
            while delta > 180:
                delta -= 360
            while delta < -180:
                delta += 360
            return delta

        delta1 = normalize_delta(target - pos1)
        delta2 = normalize_delta(target - pos2)

        # Crossed if signs differ
        return (delta1 * delta2) < 0

    def _refine_crossing(
        self, time1: Time, time2: Time, target_position: float
    ) -> Time:
        """
        Refine crossing time using binary search.

        Args:
            time1: Earlier time
            time2: Later time
            target_position: Target orbital position

        Returns:
            Refined time
        """
        # Binary search for better accuracy
        for _ in range(5):  # 5 iterations gives ~2 second accuracy
            mid_time = time1 + (time2 - time1) / 2
            sc_state = self.ephemeris.get_spacecraft_state(mid_time)
            mid_position = self._get_orbital_position(sc_state)

            if self._check_crossing(
                self._get_orbital_position(
                    self.ephemeris.get_spacecraft_state(time1)
                ),
                mid_position,
                target_position,
            ):
                time2 = mid_time
            else:
                time1 = mid_time

        return time1 + (time2 - time1) / 2

    def _vector_to_radec(self, vector: np.ndarray) -> Tuple[float, float]:
        """Convert ECI vector to RA/Dec."""
        v = vector / np.linalg.norm(vector)
        ra = np.degrees(np.arctan2(v[1], v[0]))
        if ra < 0:
            ra += 360
        dec = np.degrees(np.arcsin(v[2]))
        return ra, dec

    def _check_sun_constraints(
        self,
        pointing_vector: np.ndarray,
        sc_pos: np.ndarray,
        sun_pos: np.ndarray,
    ) -> Tuple[float, bool]:
        """
        Check Sun avoidance and antisolar hemisphere constraints.

        Args:
            pointing_vector: Pointing direction (unit vector)
            sc_pos: Spacecraft position
            sun_pos: Sun position

        Returns:
            Tuple of (sun_angle_deg, in_antisolar_hemisphere)
        """
        # Vector from spacecraft to Sun
        sc_to_sun = sun_pos - sc_pos
        sc_to_sun_unit = sc_to_sun / np.linalg.norm(sc_to_sun)

        # Angle between pointing and Sun
        sun_angle = np.degrees(
            np.arccos(np.clip(np.dot(pointing_vector, sc_to_sun_unit), -1, 1))
        )

        # Antisolar hemisphere means angle > 90°
        in_antisolar = sun_angle > 90.0

        return sun_angle, in_antisolar


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
