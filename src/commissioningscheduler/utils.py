# utils.py
from datetime import datetime, timedelta

# from typing import Tuple
import math


def align_to_minute_boundary(
    dt: datetime, direction: str = "down"
) -> datetime:
    """
    Align datetime to the nearest minute boundary.

    Args:
        dt: Datetime to align
        direction: 'down' to floor, 'up' to ceil, 'nearest' for closest

    Returns:
        Aligned datetime
    """
    if direction == "down":
        return dt.replace(second=0, microsecond=0)
    elif direction == "up":
        if dt.second > 0 or dt.microsecond > 0:
            return dt.replace(second=0, microsecond=0) + timedelta(minutes=1)
        return dt.replace(second=0, microsecond=0)
    elif direction == "nearest":
        if dt.second >= 30:
            return dt.replace(second=0, microsecond=0) + timedelta(minutes=1)
        return dt.replace(second=0, microsecond=0)
    else:
        raise ValueError(f"Unknown direction: {direction}")


def ensure_utc_time(dt: datetime) -> datetime:
    """
    Ensure datetime is timezone-aware and in UTC.

    Args:
        dt: Input datetime

    Returns:
        Timezone-aware UTC datetime
    """
    if dt.tzinfo is None:
        from datetime import timezone

        return dt.replace(tzinfo=timezone.utc)
    return dt


def format_utc_time(dt: datetime) -> str:
    """Format datetime as UTC ISO string with Z suffix."""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc_time(time_str: str) -> datetime:
    """Parse UTC ISO string (with or without Z suffix) to datetime."""
    # from datetime import timezone

    time_str = time_str.rstrip("Z")
    dt = datetime.fromisoformat(time_str)
    return ensure_utc_time(dt)


def calculate_angular_separation(
    ra1: float, dec1: float, ra2: float, dec2: float
) -> float:
    """
    Calculate angular separation between two points on the sky.

    Args:
        ra1, dec1: First point (degrees)
        ra2, dec2: Second point (degrees)

    Returns:
        Angular separation in degrees
    """
    ra1_rad = math.radians(ra1)
    dec1_rad = math.radians(dec1)
    ra2_rad = math.radians(ra2)
    dec2_rad = math.radians(dec2)

    cos_sep = math.sin(dec1_rad) * math.sin(dec2_rad) + math.cos(
        dec1_rad
    ) * math.cos(dec2_rad) * math.cos(ra1_rad - ra2_rad)

    # Clamp to valid range to avoid numerical errors
    cos_sep = max(-1.0, min(1.0, cos_sep))

    return math.degrees(math.acos(cos_sep))


def compute_data_volume_gb(
    nir_minutes: float = 0.0,
    vis_minutes: float = 0.0,
    nir_rate_mbps: float = 2.74,
    vis_rate_mbps: float = 0.88,
) -> float:
    """
    Compute total data volume in GB.

    Args:
        nir_minutes: NIR observation duration in minutes
        vis_minutes: Visible observation duration in minutes
        nir_rate_mbps: NIR data rate in Mbps
        vis_rate_mbps: Visible data rate in Mbps

    Returns:
        Total data volume in GB
    """
    nir_gb = (nir_minutes * 60.0 * nir_rate_mbps) / 8000.0
    vis_gb = (vis_minutes * 60.0 * vis_rate_mbps) / 8000.0
    return nir_gb + vis_gb


def compute_slew_time_minutes(
    sep_deg: float, slew_rate_deg_per_min: float = 1.0
) -> float:
    """
    Compute slew time based on angular separation.

    Args:
        sep_deg: Angular separation in degrees
        slew_rate_deg_per_min: Slew rate in degrees per minute

    Returns:
        Slew time in minutes
    """
    return sep_deg / slew_rate_deg_per_min
