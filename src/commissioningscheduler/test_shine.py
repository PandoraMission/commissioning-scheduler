# test_shine_scheduler.py
"""
Test and demonstration script for shine_scheduler.py
"""

import logging
from astropy.time import Time
import astropy.units as u
from shine import (
    EphemerisProvider,
    MoonshinePointing,
    EarthshinePointing,
    compute_moonshine_pointing,
    compute_earthshine_pointing,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_moonshine():
    """Test Moonshine pointing calculations."""
    print("\n" + "=" * 60)
    print("MOONSHINE POINTING TEST")
    print("=" * 60)

    # TLE for Pandora
    tle_line1 = (
        "1 99152U 26011B   26005.66013674 .000000000  00000+0  00000-0 0    16"
    )
    tle_line2 = (
        "2 99152  97.6750  17.6690 0000000 328.8990  20.9640 14.86530781  0004"
    )

    # Test time
    time = Time("2026-02-01 00:00:00", scale="utc")

    # Test different angular positions and limb separations
    positions = [0, 45, 90, 135, 180, 225, 270, 315]  # Clock positions
    separations = [5, 10, 15, 20]  # Degrees from limb

    ephemeris = EphemerisProvider(tle_line1, tle_line2)
    calculator = MoonshinePointing(ephemeris)

    print(f"\nTime: {time.iso}")
    print(f"\n{'Position':>8} {'Sep':>5} {'RA':>8} {'Dec':>8} {'Visible':>8}")
    print("-" * 45)

    for pos in positions[:4]:  # Just show a few for brevity
        for sep in separations[:2]:
            result = calculator.calculate_pointing(
                time, pos, sep, check_earth_blockage=True
            )
            print(
                f"{pos:>8.0f}° {sep:>5.0f}° {result.ra_deg:>8.2f}° "
                f"{result.dec_deg:>8.2f}° {str(result.moon_visible):>8}"
            )


def test_earthshine():
    """Test Earthshine pointing calculations."""
    print("\n" + "=" * 60)
    print("EARTHSHINE POINTING TEST")
    print("=" * 60)

    # TLE for Pandora
    tle_line1 = (
        "1 99152U 26011B   26005.66013674 .000000000  00000+0  00000-0 0    16"
    )
    tle_line2 = (
        "2 99152  97.6750  17.6690 0000000 328.8990  20.9640 14.86530781  0004"
    )

    # Start time
    start_time = Time("2026-02-01 00:00:00", scale="utc")

    # Test different orbital positions
    orbital_positions = [0, 90, 180, 270]  # North, East, South, West
    limb_separation = 10.0  # degrees

    ephemeris = EphemerisProvider(tle_line1, tle_line2)
    calculator = EarthshinePointing(ephemeris)

    print(f"\nSearching from: {start_time.iso}")
    print(f"Limb separation: {limb_separation}°")
    print(
        f"\n{'Target Pos':>11} {'Found Time':>20} {'RA':>8} {'Dec':>8} "
        f"{'Sun Angle':>10} {'Antisolar':>9}"
    )
    print("-" * 75)

    for orbital_pos in orbital_positions:
        try:
            result = calculator.calculate_pointing(
                start_time, orbital_pos, limb_separation
            )
            print(
                f"{orbital_pos:>11.0f}° {result.time.iso:>20} "
                f"{result.ra_deg:>8.2f}° {result.dec_deg:>8.2f}° "
                f"{result.sun_angle_deg:>10.1f}° {str(result.pointing_in_antisolar):>9}"
            )
        except Exception as e:
            print(f"{orbital_pos:>11.0f}° ERROR: {e}")


def test_earthshine_detailed():
    """Detailed test showing orbital positions over time."""
    print("\n" + "=" * 60)
    print("EARTHSHINE DETAILED TEST - Orbital Position Tracking")
    print("=" * 60)

    tle_line1 = (
        "1 99152U 26011B   26005.66013674 .000000000  00000+0  00000-0 0    16"
    )
    tle_line2 = (
        "2 99152  97.6750  17.6690 0000000 328.8990  20.9640 14.86530781  0004"
    )

    ephemeris = EphemerisProvider(tle_line1, tle_line2)
    calculator = EarthshinePointing(ephemeris)

    start_time = Time("2026-02-01 00:00:00", scale="utc")

    print(f"\nTracking orbital position over one orbit:")
    print(f"Start time: {start_time.iso}")
    print(
        f"\n{'Time (min)':>10} {'Orbital Pos':>12} {'Latitude':>10} {'Altitude':>10}"
    )
    print("-" * 50)

    # Track position every 5 minutes for ~100 minutes
    for i in range(0, 105, 5):
        current_time = start_time + i * u.min
        sc_state = ephemeris.get_spacecraft_state(current_time)
        orbital_pos = calculator._get_orbital_position(sc_state)
        lat = calculator._get_latitude(sc_state.position_km)
        alt = sc_state.altitude_km

        print(f"{i:>10} {orbital_pos:>11.1f}° {lat:>9.2f}° {alt:>9.1f} km")


def test_earthshine_multiple_separations():
    """Test multiple limb separations at each orbital position."""
    print("\n" + "=" * 60)
    print("EARTHSHINE TEST - Multiple Limb Separations")
    print("=" * 60)

    tle_line1 = (
        "1 99152U 26011B   26005.66013674 .000000000  00000+0  00000-0 0    16"
    )
    tle_line2 = (
        "2 99152  97.6750  17.6690 0000000 328.8990  20.9640 14.86530781  0004"
    )

    ephemeris = EphemerisProvider(tle_line1, tle_line2)
    calculator = EarthshinePointing(ephemeris)

    start_time = Time("2026-02-01 00:00:00", scale="utc")

    orbital_positions = [0, 90, 180, 270]
    separations = [5, 10, 15, 20]

    print(f"\nSearching from: {start_time.iso}")
    print(
        f"\n{'Orb Pos':>8} {'Sep':>5} {'Time':>20} {'RA':>8} {'Dec':>8} "
        f"{'Sun∠':>6} {'OK':>4}"
    )
    print("-" * 70)

    for orb_pos in orbital_positions:
        for sep in separations:
            try:
                result = calculator.calculate_pointing(
                    start_time, orb_pos, sep
                )
                ok = "✓" if result.pointing_in_antisolar else "✗"
                print(
                    f"{orb_pos:>8.0f}° {sep:>5.0f}° {result.time.iso:>20} "
                    f"{result.ra_deg:>8.2f}° {result.dec_deg:>8.2f}° "
                    f"{result.sun_angle_deg:>6.1f}° {ok:>4}"
                )
            except Exception as e:
                print(f"{orb_pos:>8.0f}° {sep:>5.0f}° ERROR: {str(e)[:40]}")


def test_with_gmat_file():
    """Test using GMAT file for ephemeris."""
    print("\n" + "=" * 60)
    print("GMAT FILE TEST")
    print("=" * 60)

    gmat_file = "/Users/bhord/research/pandora/calendars/supporting_files/Pandora-600km-withoutdrag-20251218_tom.txt"  # Update this path

    # This would use the GMAT file if available
    ephemeris = EphemerisProvider(gmat_file=gmat_file)
    print("\nTo test with GMAT file, update the path in test_with_gmat_file()")
    print(ephemeris)


def demo_complete_observation_plan():
    """Demonstrate planning a complete observation sequence."""
    print("\n" + "=" * 60)
    print("COMPLETE OBSERVATION PLAN DEMO")
    print("=" * 60)

    tle_line1 = (
        "1 99152U 26011B   26005.66013674 .000000000  00000+0  00000-0 0    16"
    )
    tle_line2 = (
        "2 99152  97.6750  17.6690 0000000 328.8990  20.9640 14.86530781  0004"
    )

    start_time = Time("2026-02-01 00:00:00", scale="utc")

    print("\n--- Moonshine Observations ---")
    print(
        "Planning observations at 8 positions around Moon, 4 separations each"
    )

    positions = [0, 45, 90, 135, 180, 225, 270, 315]
    separations = [5, 10, 15, 20]

    moonshine_count = 0
    for pos in positions:
        for sep in separations:
            result = compute_moonshine_pointing(
                start_time, pos, sep, tle_line1, tle_line2
            )
            if result.moon_visible:
                moonshine_count += 1

    print(f"Viable Moonshine observations: {moonshine_count}")

    print("\n--- Earthshine Observations ---")
    print("Planning observations at 4 orbital positions, 4 separations each")

    orbital_positions = [0, 90, 180, 270]
    earthshine_count = 0

    for orb_pos in orbital_positions:
        for sep in separations:
            try:
                result = compute_earthshine_pointing(
                    start_time, orb_pos, sep, tle_line1, tle_line2
                )
                if (
                    result.pointing_in_antisolar
                    and result.sun_angle_deg > 91.0
                ):
                    earthshine_count += 1
            except Exception as e:
                print("!!! Exception:")
                print(e)
                pass

    print(f"Viable Earthshine observations: {earthshine_count}")
    print(
        f"\nTotal observation opportunities: {moonshine_count + earthshine_count}"
    )


if __name__ == "__main__":
    # Run individual tests
    test_moonshine()
    test_earthshine()

    # Additional detailed tests
    test_earthshine_detailed()
    test_earthshine_multiple_separations()

    # Complete demo
    demo_complete_observation_plan()
