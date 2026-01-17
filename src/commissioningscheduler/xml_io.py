# xml_io.py
import os
import re
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import uuid
import datetime

from astropy.time import Time

from models import Observation, ObservationSequence, Visit, SchedulerConfig
from utils import (
    # parse_utc_time,
    format_utc_time,
    # compute_data_volume_gb
)
from roll import calculate_roll

logger = logging.getLogger(__name__)

# Namespace for Pandora XML files
NS = {"cal": "/pandora/calendar/"}


class ObservationParser:
    """
    Parses per-target XML files into Observation objects.
    Based on parser.py patterns with namespace stripping.
    """

    TASK_FILENAME_RE = re.compile(
        r"(?P<task>\d{4})_(?P<seq>\d{3})_.*\.xml$", re.IGNORECASE
    )

    def __init__(self, keep_raw_tree: bool = True):
        """
        Initialize parser.

        Args:
            keep_raw_tree: Whether to store raw XML trees for later cloning
        """
        self.keep_raw_tree = keep_raw_tree

    def parse_file(self, filename: str) -> Observation:
        """
        Parse an input XML file into an Observation object.

        Args:
            filename: Path to XML file

        Returns:
            Observation object
        """
        base = os.path.basename(filename)

        # Extract task number and sequence from filename
        task_number = None
        obs_id = os.path.splitext(base)[0]
        m = self.TASK_FILENAME_RE.search(base)
        if m:
            task_number = m.group("task")
            obs_id = f"{m.group('task')}_{m.group('seq')}"

        tree = ET.parse(filename)
        root = tree.getroot()

        # Strip namespaces for easier access
        self._strip_namespace_recursive(root)

        # Locate the Observation_Sequence element
        visit_elem = root.find("Visit")
        if visit_elem is None:
            raise RuntimeError(f"No <Visit> element found in {filename}")

        seq_elem = visit_elem.find("Observation_Sequence")
        if seq_elem is None:
            raise RuntimeError(
                f"No <Observation_Sequence> element found in {filename}"
            )

        obs = Observation(obs_id=obs_id, task_number=task_number)

        # Store raw XML tree for later cloning
        if self.keep_raw_tree:
            obs.raw_xml_tree = seq_elem

        # Parse core parameters
        obs.target_name = self._find_text(root, ".//Target")

        prio_text = self._find_text(root, ".//Priority")
        if prio_text:
            try:
                obs.priority = float(prio_text)
            except ValueError:
                logger.warning(f"Priority not numeric: {prio_text}")

        # Parse boresight
        ra_text = self._find_text(root, ".//Boresight/RA")
        dec_text = self._find_text(root, ".//Boresight/DEC")
        if ra_text:
            try:
                obs.boresight_ra = float(ra_text)
            except ValueError:
                logger.warning(f"Could not parse RA: {ra_text}")
        if dec_text:
            try:
                obs.boresight_dec = float(dec_text)
            except ValueError:
                logger.warning(f"Could not parse DEC: {dec_text}")

        # Parse visible camera parameters
        vis_block = root.find(".//AcquireVisCamScienceData")
        if vis_block is None:
            vis_block = root.find(".//AcquireVisCamImages")

        if vis_block is not None:
            ntfr = self._find_text(vis_block, "NumTotalFramesRequested")
            if not ntfr:
                ntfr = self._find_text(vis_block, "NumExposures")

            et_us = self._find_text(vis_block, "ExposureTime_us")
            fpc = self._find_text(vis_block, "FramesPerCoadd")

            if ntfr:
                try:
                    obs.num_total_frames_requested = int(float(ntfr))
                except ValueError:
                    logger.warning(
                        f"Could not parse NumTotalFramesRequested: {ntfr}"
                    )

            if et_us:
                try:
                    obs.exposure_time_us = float(et_us)
                except ValueError:
                    logger.warning(f"Could not parse ExposureTime_us: {et_us}")

            if fpc:
                try:
                    obs.frames_per_coadd = int(float(fpc))
                except ValueError:
                    logger.warning(f"Could not parse FramesPerCoadd: {fpc}")

            # Calculate visible duration
            if obs.num_total_frames_requested and obs.exposure_time_us:
                total_seconds = (
                    obs.num_total_frames_requested * obs.exposure_time_us
                ) / 1e6
                obs.visible_duration = total_seconds / 60.0
                logger.debug(
                    f"Calculated visible duration: {obs.visible_duration:.3f} minutes"
                )

        # Parse NIR camera parameters
        nir_block = root.find(".//AcquireInfCamImages")
        if nir_block is not None:
            obs.sc_integrations = self._parse_int(nir_block, "SC_Integrations")
            obs.sc_resets1 = self._parse_int(nir_block, "SC_Resets1")
            obs.sc_resets2 = self._parse_int(nir_block, "SC_Resets2")
            obs.sc_dropframes1 = self._parse_int(nir_block, "SC_DropFrames1")
            obs.sc_dropframes2 = self._parse_int(nir_block, "SC_DropFrames2")
            obs.sc_dropframes3 = self._parse_int(nir_block, "SC_DropFrames3")
            obs.sc_readframes = self._parse_int(nir_block, "SC_ReadFrames")
            obs.roi_sizex = self._parse_int(nir_block, "ROI_SizeX")
            obs.roi_sizey = self._parse_int(nir_block, "ROI_SizeY")

            # Calculate NIR duration
            if obs.sc_integrations and obs.roi_sizex and obs.roi_sizey:
                integration_time_sec = self._calculate_nir_integration_time(
                    obs
                )
                total_seconds = obs.sc_integrations * integration_time_sec
                obs.nir_duration = total_seconds / 60.0
                logger.debug(
                    f"Calculated NIR duration: {obs.nir_duration:.3f} minutes"
                )

        # Manually trigger duration calculation if __post_init__ didn't do it
        if obs.duration is None:
            nir = obs.nir_duration or 0.0
            vis = obs.visible_duration or 0.0
            if nir > 0 or vis > 0:
                obs.duration = max(nir, vis)
                logger.debug(
                    f"Manually calculated total duration: {obs.duration:.3f} minutes"
                )
            else:
                logger.warning(
                    f"Could not calculate duration for {filename} - no camera data found"
                )

        # Store filename in metadata
        obs.metadata["source_filename"] = filename

        return obs

    def parse_directory(self, xml_dir: str) -> List[Observation]:
        """
        Parse all XML files in a directory.

        Args:
            xml_dir: Directory containing XML files

        Returns:
            List of Observation objects
        """
        xml_files = gather_task_xmls(xml_dir)
        observations = []

        for xml_file in xml_files:
            try:
                obs = self.parse_file(xml_file)
                observations.append(obs)
            except Exception as e:
                logger.error(f"Error parsing {xml_file}: {e}")

        return observations

    def _find_text(self, root: ET.Element, path: str) -> Optional[str]:
        """Find element text, returning None if not found."""
        el = root.find(path)
        if el is None or el.text is None:
            return None
        return el.text.strip()

    def _parse_int(self, parent: ET.Element, tag: str) -> Optional[int]:
        """Parse integer from child element."""
        text = self._find_text(parent, tag)
        if text:
            try:
                return int(text)
            except ValueError:
                logger.warning(f"Could not parse {tag} as int: {text}")
        return None

    def _strip_namespace_recursive(self, elem: ET.Element):
        """Remove namespace prefixes from tags in-place."""
        if "}" in elem.tag:
            elem.tag = elem.tag.split("}", 1)[1]
        for child in elem:
            self._strip_namespace_recursive(child)

    def _calculate_nir_integration_time(self, obs: Observation) -> float:
        """
        Calculate NIR single integration time in seconds.

        Formula: (SC_Resets1 + SC_Resets2 + SC_DropFrames1 + SC_DropFrames2 +
                SC_DropFrames3 + SC_ReadFrames + 1) *
                (ROI_SizeX * ROI_SizeY + (ROI_SizeY * 12)) * 0.00001
        """
        if not obs.roi_sizex or not obs.roi_sizey:
            return 0.0

        frame_count_term = (
            (obs.sc_resets1 or 0)
            + (obs.sc_resets2 or 0)
            + (obs.sc_dropframes1 or 0)
            + (obs.sc_dropframes2 or 0)
            + (obs.sc_dropframes3 or 0)
            + (obs.sc_readframes or 0)
            + 1
        )

        pixel_term = (obs.roi_sizex * obs.roi_sizey) + (obs.roi_sizey * 12)

        return frame_count_term * pixel_term * 0.00001


def gather_task_xmls(
    xml_dir: str, pattern: str = r"^\d{4}_\d{3}_.*\.xml$"
) -> List[str]:
    """
    Gather all task XML files from a directory.

    Args:
        xml_dir: Directory containing XML files
        pattern: Regex pattern for matching XML files

    Returns:
        List of full paths to XML files, sorted
    """
    xml_dir_path = Path(xml_dir)
    if not xml_dir_path.exists():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")

    xml_files = []
    regex = re.compile(pattern)

    for file in xml_dir_path.iterdir():
        if file.is_file() and regex.match(file.name):
            xml_files.append(str(file))

    return sorted(xml_files)


class ScheduleWriter:
    """
    Writes scheduled observations to output XML file.
    Based on writer.py patterns.
    """

    def __init__(self, config: SchedulerConfig):
        """
        Initialize writer.

        Args:
            config: Scheduler configuration
        """
        self.config = config

    def write_schedule(
        self,
        visits: List[Visit],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write scheduled visits to output XML file.

        Args:
            visits: List of Visit objects
            output_path: Path to output XML file
            metadata: Optional metadata for ScienceCalendar header
        """
        # Create root
        root = ET.Element("ScienceCalendar")
        root.set("xmlns", "/pandora/calendar/")

        # Add metadata
        meta = ET.SubElement(root, "Meta")
        if metadata:
            for key, value in metadata.items():
                meta.set(key, str(value))
        else:
            # Default metadata
            meta.set(
                "Valid_From", format_utc_time(self.config.commissioning_start)
            )
            meta.set("Expires", format_utc_time(self.config.commissioning_end))
            meta.set("Created", format_utc_time(datetime.now()))
            meta.set("Delivery_Id", str(uuid.uuid4()))

        # Sort ALL visits by their start time (chronological order)
        sorted_visits = sorted(
            visits,
            key=lambda v: v.start_time if v.start_time else datetime.max,
        )

        logger.info(
            f"Writing {len(sorted_visits)} visits in chronological order"
        )

        # Add visits with sequential numeric IDs
        for visit_num, visit in enumerate(sorted_visits, start=1):
            visit_elem = self._create_visit_element(visit, visit_num)
            root.append(visit_elem)

        # Write with formatting
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

        logger.info(
            f"Wrote schedule with {len(visits)} visits to {output_path}"
        )

    def _create_visit_element(
        self, visit: Visit, visit_num: int
    ) -> ET.Element:
        """
        Create XML element for a Visit.

        Args:
            visit: Visit object
            visit_num: Numeric visit number (1-based)

        Returns:
            XML Element
        """
        visit_elem = ET.Element("Visit")

        # Use numeric ID
        id_elem = ET.SubElement(visit_elem, "ID")
        id_elem.text = f"{visit_num:04d}"

        # Add observation sequences
        for seq in visit.observation_sequences:
            seq_elem = self._create_observation_sequence_element(
                seq, visit.visit_id
            )
            visit_elem.append(seq_elem)

        return visit_elem

    def _create_observation_sequence_element(
        self, seq: ObservationSequence, task_id: str
    ) -> ET.Element:
        """
        Create XML element for an ObservationSequence.
        """
        # Check if this is a staring sequence
        is_staring = seq.metadata.get("is_staring", False)

        if is_staring:
            # Create minimal sequence with only visible camera block
            elem = self._create_staring_sequence(seq, task_id)
        elif seq.raw_xml_tree is not None:
            elem = self._clone_and_adjust_sequence(seq, task_id)
        elif (
            seq.parent_observation
            and seq.parent_observation.raw_xml_tree is not None
        ):
            elem = self._clone_and_adjust_sequence(seq, task_id)
        else:
            elem = self._create_minimal_sequence(seq, task_id)

        # Add visible camera block if needed
        if seq.metadata.get("needs_vis_block", False):
            self._ensure_vis_camera_block(elem, seq)

        return elem

    def _clone_and_adjust_sequence(
        self, seq: ObservationSequence, task_id: str
    ) -> ET.Element:
        """
        Clone observation sequence XML and adjust timing/parameters.

        Args:
            seq: ObservationSequence
            task_id: Task identifier for the Task field

        Returns:
            XML Element
        """
        import copy

        # Get source tree
        if seq.raw_xml_tree is not None:
            source = seq.raw_xml_tree
        elif (
            seq.parent_observation
            and seq.parent_observation.raw_xml_tree is not None
        ):
            source = seq.parent_observation.raw_xml_tree
        else:
            return self._create_minimal_sequence(seq, task_id)

        # Deep copy
        seq_elem = copy.deepcopy(source)

        # Update ID
        id_elem = seq_elem.find("ID")
        if id_elem is not None:
            id_elem.text = seq.sequence_id

        # Add Task field right after ID
        task_elem = seq_elem.find("Task")
        if task_elem is None:
            # Insert Task after ID
            id_index = (
                list(seq_elem).index(id_elem) if id_elem is not None else 0
            )
            task_elem = ET.Element("Task")
            seq_elem.insert(id_index + 1, task_elem)

        # Set task value (e.g., "0310_000" or "CVZ")
        if seq.obs_id == "CVZ":
            task_elem.text = "CVZ"
        else:
            task_elem.text = seq.obs_id

        # Update timing
        timing = seq_elem.find(".//Timing")
        if timing is not None:
            start_elem = timing.find("Start")
            stop_elem = timing.find("Stop")
            if start_elem is not None:
                start_elem.text = format_utc_time(seq.start)
            if stop_elem is not None:
                stop_elem.text = format_utc_time(seq.stop)

        # Update target/coordinates if overridden
        if seq.target_name:
            target_elem = seq_elem.find(".//Target")
            if target_elem is not None:
                target_elem.text = seq.target_name

        if seq.boresight_ra is not None:
            ra_elem = seq_elem.find(".//Boresight/RA")
            if ra_elem is not None:
                ra_elem.text = str(seq.boresight_ra)

        if seq.boresight_dec is not None:
            dec_elem = seq_elem.find(".//Boresight/DEC")
            if dec_elem is not None:
                dec_elem.text = str(seq.boresight_dec)

        # Calculate and update/add Roll angle
        if (
            seq.boresight_ra is not None
            and seq.boresight_dec is not None
            and seq.start is not None
        ):
            try:
                obs_time = Time(seq.start)
                roll_angle = calculate_roll(
                    seq.boresight_ra, seq.boresight_dec, obs_time
                )

                # Find or create Roll element in Boresight
                boresight_elem = seq_elem.find(".//Boresight")
                if boresight_elem is not None:
                    roll_elem = boresight_elem.find("Roll")
                    if roll_elem is None:
                        # Create Roll element after DEC
                        dec_elem = boresight_elem.find("DEC")
                        if dec_elem is not None:
                            dec_index = list(boresight_elem).index(dec_elem)
                            roll_elem = ET.Element("Roll")
                            boresight_elem.insert(dec_index + 1, roll_elem)
                        else:
                            # If DEC not found, just append
                            roll_elem = ET.SubElement(boresight_elem, "Roll")

                    roll_elem.text = f"{roll_angle:.6f}"
                    logger.debug(
                        f"Calculated roll angle {roll_angle:.6f} for sequence {seq.sequence_id}"
                    )
                else:
                    logger.warning(
                        f"Boresight element not found for sequence {seq.sequence_id}"
                    )

            except Exception as e:
                logger.error(
                    f"Failed to calculate roll for sequence {seq.sequence_id}: {e}"
                )

        # Apply adjusted camera parameters if they exist
        if "adjusted_params" in seq.metadata:
            adjusted = seq.metadata["adjusted_params"]

            # Update visible camera parameters
            if "NumTotalFramesRequested" in adjusted:
                vis_elem = seq_elem.find(".//AcquireVisCamScienceData")
                if vis_elem is None:
                    vis_elem = seq_elem.find(".//AcquireVisCamImages")

                if vis_elem is not None:
                    # Update NumTotalFramesRequested
                    ntfr_elem = vis_elem.find("NumTotalFramesRequested")
                    if ntfr_elem is not None:
                        ntfr_elem.text = str(
                            adjusted["NumTotalFramesRequested"]
                        )

                    # Also check for NumExposures (alternative name)
                    ne_elem = vis_elem.find("NumExposures")
                    if ne_elem is not None:
                        ne_elem.text = str(adjusted["NumTotalFramesRequested"])

            # Update NIR camera parameters
            if "SC_Integrations" in adjusted:
                nir_elem = seq_elem.find(".//AcquireInfCamImages")
                if nir_elem is not None:
                    sci_elem = nir_elem.find("SC_Integrations")
                    if sci_elem is not None:
                        sci_elem.text = str(adjusted["SC_Integrations"])

        return seq_elem

    def _create_minimal_sequence(
        self, seq: ObservationSequence, task_id: str
    ) -> ET.Element:
        """
        Create minimal observation sequence from scratch.

        Args:
            seq: ObservationSequence
            task_id: Task identifier for the Task field

        Returns:
            XML Element
        """
        obs_seq_elem = ET.Element("Observation_Sequence")

        # ID
        id_elem = ET.SubElement(obs_seq_elem, "ID")
        id_elem.text = seq.sequence_id

        # Task (new field)
        task_elem = ET.SubElement(obs_seq_elem, "Task")
        if seq.obs_id == "CVZ":
            task_elem.text = "CVZ"
        else:
            task_elem.text = seq.obs_id

        # Observational Parameters
        obs_params = ET.SubElement(obs_seq_elem, "Observational_Parameters")

        # Get parameters from sequence or parent
        target = seq.target_name
        ra = seq.boresight_ra
        dec = seq.boresight_dec
        priority = seq.priority

        if seq.parent_observation:
            target = target or seq.parent_observation.target_name
            ra = ra if ra is not None else seq.parent_observation.boresight_ra
            dec = (
                dec
                if dec is not None
                else seq.parent_observation.boresight_dec
            )
            priority = (
                priority
                if priority is not None
                else seq.parent_observation.priority
            )

        target_elem = ET.SubElement(obs_params, "Target")
        target_elem.text = target or "Unknown"

        priority_elem = ET.SubElement(obs_params, "Priority")
        priority_elem.text = str(priority or 0)

        timing_elem = ET.SubElement(obs_params, "Timing")
        start_elem = ET.SubElement(timing_elem, "Start")
        start_elem.text = format_utc_time(seq.start)
        stop_elem = ET.SubElement(timing_elem, "Stop")
        stop_elem.text = format_utc_time(seq.stop)

        boresight_elem = ET.SubElement(obs_params, "Boresight")
        ra_elem = ET.SubElement(boresight_elem, "RA")
        ra_elem.text = str(ra or 0.0)
        dec_elem = ET.SubElement(boresight_elem, "DEC")
        dec_elem.text = str(dec or 0.0)

        # Calculate and add roll angle
        if (
            seq.boresight_ra is not None
            and seq.boresight_dec is not None
            and seq.start is not None
        ):
            try:
                obs_time = Time(seq.start)
                roll_angle = calculate_roll(
                    seq.boresight_ra, seq.boresight_dec, obs_time
                )
                roll_elem = ET.SubElement(boresight_elem, "Roll")
                roll_elem.text = f"{roll_angle:.6f}"
                logger.debug(
                    f"Calculated roll angle {roll_angle:.6f} for sequence {seq.sequence_id}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to calculate roll for sequence {seq.sequence_id}: {e}"
                )

        return obs_seq_elem

    def _needs_vis_block(
        self, elem: ET.Element, seq: ObservationSequence
    ) -> bool:
        """Check if observation sequence needs visible camera data block."""
        payload = elem.find("Payload_Parameters")
        if payload is None:
            return True

        # Check if visible camera block already exists
        has_vis = payload.find("AcquireVisCamScienceData") is not None
        has_vis_images = payload.find("AcquireVisCamImages") is not None

        # If no visible block exists, we need one
        return not (has_vis or has_vis_images)

    def _ensure_vis_camera_block(
        self, elem: ET.Element, seq: ObservationSequence
    ):
        """Ensure observation sequence has visible camera data block."""
        payload = elem.find("Payload_Parameters")
        if payload is None:
            payload = ET.SubElement(elem, "Payload_Parameters")

        # Check if already exists
        if payload.find("AcquireVisCamScienceData") is not None:
            return
        if payload.find("AcquireVisCamImages") is not None:
            return

        # Add visible camera block
        vis_block = ET.SubElement(payload, "AcquireVisCamScienceData")

        # Standard parameters for CVZ or visible acquisition
        # Using the correct CVZ command structure
        ET.SubElement(vis_block, "IncludeFieldSolnsInResp").text = "1"
        ET.SubElement(vis_block, "ROI_StartX").text = "384"
        ET.SubElement(vis_block, "ROI_StartY").text = "384"
        ET.SubElement(vis_block, "ROI_SizeX").text = "1280"
        ET.SubElement(vis_block, "ROI_SizeY").text = "1280"
        ET.SubElement(vis_block, "MaxMagnitudeInQuadCatalog").text = "16.5"
        ET.SubElement(vis_block, "SaveImagesToDisk").text = "1"
        ET.SubElement(vis_block, "RiceX").text = "5"
        ET.SubElement(vis_block, "RiceY").text = "25"
        ET.SubElement(vis_block, "SendThumbnails").text = "0"

        # Target information
        ET.SubElement(vis_block, "TargetID").text = seq.target_name or "CVZ"
        ET.SubElement(vis_block, "TargetRA").text = str(
            seq.boresight_ra or 0.0
        )
        ET.SubElement(vis_block, "TargetDEC").text = str(
            seq.boresight_dec or 0.0
        )

        # Star ROI detection method (pixel coordinates)
        ET.SubElement(vis_block, "StarRoiDetMethod").text = "3"
        ET.SubElement(vis_block, "numPredefinedStarRois").text = "1"

        # Predefined star ROI coordinates (pixel coordinates)
        roi_ra = ET.SubElement(vis_block, "PredefinedStarRoiRa")
        ET.SubElement(roi_ra, "RA1").text = "1024"

        roi_dec = ET.SubElement(vis_block, "PredefinedStarRoiDec")
        ET.SubElement(roi_dec, "Dec1").text = "1024"

        # Frame and exposure parameters
        ET.SubElement(vis_block, "FramesPerCoadd").text = "50"
        ET.SubElement(vis_block, "ExposureTime_us").text = "200000"
        ET.SubElement(vis_block, "MaxNumStarRois").text = "1"
        ET.SubElement(vis_block, "StarRoiDimension").text = "1280"
        ET.SubElement(vis_block, "NumTotalFramesRequested").text = "50"

        logger.debug(
            f"Added visible camera block to {seq.obs_id}_{seq.sequence_id}"
        )

    def write_schedule_from_sequences(
        self,
        sequences: List[ObservationSequence],
        output_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write schedule from a flat list of sequences.

        Creates Visits by grouping consecutive sequences from the same task.

        Args:
            sequences: List of ObservationSequence objects (should be sorted chronologically)
            output_path: Path to output XML file
            metadata: Optional metadata for ScienceCalendar header
        """
        # Create root
        root = ET.Element("ScienceCalendar")
        root.set("xmlns", "/pandora/calendar/")

        # Group sequences into visits
        # A visit contains consecutive sequences from the same observation (obs_id)
        # or consecutive CVZ/CVZ_BLOCKED sequences
        visits = self._group_sequences_into_visits(sequences)

        logger.info(
            f"Grouped {len(sequences)} sequences into {len(visits)} visits"
        )

        # Add metadata
        meta = ET.SubElement(root, "Meta")
        if metadata:
            for key, value in metadata.items():
                if key == "Total_Visits":
                    value = len(visits)
                meta.set(key, str(value))

        # Write visits with sequential IDs
        for visit_num, visit in enumerate(visits, start=1):
            visit_elem = self._create_visit_element_from_sequences(
                visit, visit_num
            )
            root.append(visit_elem)

        # Write with formatting
        tree = ET.ElementTree(root)
        ET.indent(tree, space="    ")
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

        logger.info(f"Wrote schedule to {output_path}")

    def _group_sequences_into_visits(
        self, sequences: List[ObservationSequence]
    ) -> List[List[ObservationSequence]]:
        """
        Group consecutive sequences into visits.

        Special handling:
        - Task 0312: All sequences with same obs_id stay together (14 sequences in one visit)
        - Regular observations: Consecutive sequences with same obs_id
        - CVZ/CVZ_BLOCKED: Consecutive sequences of same type
        """
        if not sequences:
            return []

        visits = []
        current_visit = [sequences[0]]

        for i in range(1, len(sequences)):
            prev_seq = sequences[i - 1]
            curr_seq = sequences[i]

            # Check if this sequence belongs in current visit
            same_obs = curr_seq.obs_id == prev_seq.obs_id
            both_cvz = curr_seq.obs_id in [
                "CVZ",
                "CVZ_BLOCKED",
            ] and prev_seq.obs_id in ["CVZ", "CVZ_BLOCKED"]
            same_cvz_type = (
                curr_seq.obs_id == prev_seq.obs_id
                and curr_seq.obs_id in ["CVZ", "CVZ_BLOCKED"]
            )

            # Time gap check
            time_gap_minutes = (
                curr_seq.start - prev_seq.stop
            ).total_seconds() / 60.0
            is_consecutive = time_gap_minutes < 1.0

            # Special case for task 0312: keep all sequences together even if not consecutive
            is_task_0312 = (
                curr_seq.obs_id == "0312_000" and prev_seq.obs_id == "0312_000"
            )

            if is_task_0312:
                # Always keep 0312 sequences in same visit
                current_visit.append(curr_seq)
            elif (same_obs or same_cvz_type) and is_consecutive:
                # Regular grouping: same obs and consecutive
                current_visit.append(curr_seq)
            elif both_cvz and is_consecutive:
                # Group consecutive CVZ of any type
                current_visit.append(curr_seq)
            else:
                # Start new visit
                visits.append(current_visit)
                current_visit = [curr_seq]

        # Add final visit
        if current_visit:
            visits.append(current_visit)

        return visits

    def _create_visit_element_from_sequences(
        self, sequences: List[ObservationSequence], visit_num: int
    ) -> ET.Element:
        """
        Create Visit element from a list of sequences.

        Args:
            sequences: List of sequences in this visit
            visit_num: Sequential visit number

        Returns:
            XML Element for Visit
        """
        visit_elem = ET.Element("Visit")

        # Numeric visit ID
        id_elem = ET.SubElement(visit_elem, "ID")
        id_elem.text = f"{visit_num:04d}"

        # Add each sequence with sequential IDs within the visit
        for seq_idx, seq in enumerate(sequences, start=1):
            # Determine task_id for the Task field
            if seq.obs_id in ["CVZ", "CVZ_BLOCKED"]:
                task_id = seq.obs_id
            else:
                task_id = seq.obs_id  # Already in format "0310_000"

            # Create the sequence element but override the sequence_id to be sequential
            seq_elem = self._create_observation_sequence_element(seq, task_id)

            # Update the ID to be sequential within the visit
            id_elem = seq_elem.find("ID")
            if id_elem is not None:
                id_elem.text = f"{seq_idx:03d}"

            visit_elem.append(seq_elem)

        return visit_elem

    def _create_staring_sequence(
        self, seq: ObservationSequence, task_id: str
    ) -> ET.Element:
        """
        Create a staring sequence (pointing at target with minimal visible data).
        Used for 0312 idle sequences.

        Args:
            seq: ObservationSequence
            task_id: Task identifier

        Returns:
            XML Element
        """
        obs_seq_elem = ET.Element("Observation_Sequence")

        # ID
        id_elem = ET.SubElement(obs_seq_elem, "ID")
        id_elem.text = seq.sequence_id

        # Task
        task_elem = ET.SubElement(obs_seq_elem, "Task")
        task_elem.text = task_id

        # Observational Parameters
        obs_params = ET.SubElement(obs_seq_elem, "Observational_Parameters")

        target_elem = ET.SubElement(obs_params, "Target")
        target_elem.text = seq.target_name or "Unknown"

        priority_elem = ET.SubElement(obs_params, "Priority")
        priority_elem.text = str(seq.priority or 0)

        timing_elem = ET.SubElement(obs_params, "Timing")
        start_elem = ET.SubElement(timing_elem, "Start")
        start_elem.text = format_utc_time(seq.start)
        stop_elem = ET.SubElement(timing_elem, "Stop")
        stop_elem.text = format_utc_time(seq.stop)

        boresight_elem = ET.SubElement(obs_params, "Boresight")
        ra_elem = ET.SubElement(boresight_elem, "RA")
        ra_elem.text = str(seq.boresight_ra or 0.0)
        dec_elem = ET.SubElement(boresight_elem, "DEC")
        dec_elem.text = str(seq.boresight_dec or 0.0)

        # Calculate and add roll angle
        if (
            seq.boresight_ra is not None
            and seq.boresight_dec is not None
            and seq.start is not None
        ):
            try:
                obs_time = Time(seq.start)
                roll_angle = calculate_roll(
                    seq.boresight_ra, seq.boresight_dec, obs_time
                )
                roll_elem = ET.SubElement(boresight_elem, "Roll")
                roll_elem.text = f"{roll_angle:.6f}"
                logger.debug(
                    f"Calculated roll angle {roll_angle:.6f} for sequence {seq.sequence_id}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to calculate roll for sequence {seq.sequence_id}: {e}"
                )

        # Payload with only visible camera
        payload = ET.SubElement(obs_seq_elem, "Payload_Parameters")
        vis_block = ET.SubElement(payload, "AcquireVisCamScienceData")

        # Standard 50-frame visible parameters (same as CVZ)
        ET.SubElement(vis_block, "IncludeFieldSolnsInResp").text = "1"
        ET.SubElement(vis_block, "ROI_StartX").text = "384"
        ET.SubElement(vis_block, "ROI_StartY").text = "384"
        ET.SubElement(vis_block, "ROI_SizeX").text = "1280"
        ET.SubElement(vis_block, "ROI_SizeY").text = "1280"
        ET.SubElement(vis_block, "MaxMagnitudeInQuadCatalog").text = "16.5"
        ET.SubElement(vis_block, "SaveImagesToDisk").text = "1"
        ET.SubElement(vis_block, "RiceX").text = "5"
        ET.SubElement(vis_block, "RiceY").text = "25"
        ET.SubElement(vis_block, "SendThumbnails").text = "0"
        ET.SubElement(vis_block, "TargetID").text = (
            seq.target_name or "Unknown"
        )
        ET.SubElement(vis_block, "TargetRA").text = str(
            seq.boresight_ra or 0.0
        )
        ET.SubElement(vis_block, "TargetDEC").text = str(
            seq.boresight_dec or 0.0
        )
        ET.SubElement(vis_block, "StarRoiDetMethod").text = "3"
        ET.SubElement(vis_block, "numPredefinedStarRois").text = "1"

        roi_ra = ET.SubElement(vis_block, "PredefinedStarRoiRa")
        ET.SubElement(roi_ra, "RA1").text = "1024"

        roi_dec = ET.SubElement(vis_block, "PredefinedStarRoiDec")
        ET.SubElement(roi_dec, "Dec1").text = "1024"

        ET.SubElement(vis_block, "FramesPerCoadd").text = "50"
        ET.SubElement(vis_block, "ExposureTime_us").text = "200000"
        ET.SubElement(vis_block, "MaxNumStarRois").text = "1"
        ET.SubElement(vis_block, "StarRoiDimension").text = "1280"
        ET.SubElement(vis_block, "NumTotalFramesRequested").text = "50"

        return obs_seq_elem
