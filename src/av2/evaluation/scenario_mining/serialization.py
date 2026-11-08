"""Read scenario mining predictions as log ID → prompt → frame-list JSON."""

import json
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from av2.evaluation.scenario_mining import SCENARIO_MINING_CATEGORIES
from av2.evaluation.typing import Sequences

FLOAT_FIELDS = {
    "translation_m": 3,
    "size": 3,
    "yaw": 1,
    "score": 1,
    "velocity_m_per_s": 3,
}
TRACK_FIELDS = {"track_id", "label", "name", "translation_m", "size", "yaw", "score"}


def _unique_object(pairs: Any) -> Dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_predictions(path: Union[str, Path]) -> Sequences:
    """Load plain JSON, preserving declared numeric dtypes and empty (0,3) arrays.

    Optional ``dtypes`` entries support float32/float64 for floating-point fields
    and int32/int64 for IDs/labels. No object dtype or executable deserializer is
    accepted. Ego poses are provided by ground truth during evaluation.
    """
    with Path(path).open(encoding="utf-8") as stream:
        document = json.load(stream, object_pairs_hook=_unique_object)
    if not isinstance(document, dict) or not document:
        raise ValueError("Expected a nonempty object of log IDs and prompts")
    sequences = {}
    mode = None
    for log_id, prompts in document.items():
        if not isinstance(prompts, dict) or not prompts:
            raise ValueError("Each log must contain an object of prompts")
        for prompt, records in prompts.items():
            if not isinstance(records, list) or not records:
                raise ValueError("Each prompt must contain a nonempty frame list")
            frames = []
            timestamps = set()
            for record in records:
                if not isinstance(record, dict):
                    raise ValueError("Frames must be objects")
                timestamp = record.get("timestamp_ns")
                if (
                    type(timestamp) is not int
                    or not 0 <= timestamp < 2**63
                    or timestamp in timestamps
                ):
                    raise ValueError(
                        "Frame timestamps must be unique nonnegative int64 values"
                    )
                timestamps.add(timestamp)
                frame: Dict[str, Any] = {"timestamp_ns": timestamp}
                tracking = "track_id" in record
                if mode is not None and mode != tracking:
                    raise ValueError("Do not mix tracking and track-free frames")
                mode = tracking
                if not tracking:
                    if (
                        set(record) != {"timestamp_ns", "is_positive"}
                        or type(record["is_positive"]) is not bool
                    ):
                        raise ValueError(
                            "Track-free frames require timestamp_ns and boolean is_positive"
                        )
                    frame["is_positive"] = record["is_positive"]
                else:
                    allowed = TRACK_FIELDS | {
                        "timestamp_ns",
                        "dtypes",
                        "is_positive",
                        "velocity_m_per_s",
                    }
                    if (
                        not TRACK_FIELDS <= record.keys()
                        or not record.keys() <= allowed
                    ):
                        raise ValueError(
                            "Tracking frame fields do not match the JSON format"
                        )
                    dtypes = record.get("dtypes", {})
                    if not isinstance(dtypes, dict) or not dtypes.keys() <= (
                        FLOAT_FIELDS.keys() | {"track_id", "label"}
                    ):
                        raise ValueError("Invalid dtype declarations")
                    for field in TRACK_FIELDS | (record.keys() & {"velocity_m_per_s"}):
                        values = record[field]
                        if not isinstance(values, list):
                            raise ValueError("Tracking arrays must be lists")
                        if field == "name":
                            if any(
                                value not in SCENARIO_MINING_CATEGORIES
                                for value in values
                            ):
                                raise ValueError("Invalid category name")
                            array = np.asarray(values, dtype=str)
                        elif field in {"label", "track_id"}:
                            dtype = dtypes.get(field, "int64")
                            if dtype not in ("int32", "int64") or any(
                                type(v) is not int or not 0 <= v <= np.iinfo(dtype).max
                                for v in values
                            ):
                                raise ValueError("Invalid tracking integer or dtype")
                            array = np.asarray(values, dtype=dtype)
                            if field == "label" and np.any(
                                array >= len(SCENARIO_MINING_CATEGORIES)
                            ):
                                raise ValueError("Invalid category label")
                        else:
                            dtype = dtypes.get(field, "float64")
                            if dtype not in ("float32", "float64"):
                                raise ValueError("Invalid floating-point dtype")
                            if np.asarray(values).dtype.kind not in "fiu":
                                raise ValueError("Expected numeric tracking values")
                            array = np.asarray(values, dtype=dtype)
                            width = FLOAT_FIELDS[field]
                            if width > 1:
                                if values and array.shape != (len(values), width):
                                    raise ValueError("Expected N by 3 vector array")
                                array = array.reshape(-1, width)
                            if not np.isfinite(array).all():
                                raise ValueError("Tracking values must be finite")
                        if field not in FLOAT_FIELDS or FLOAT_FIELDS[field] == 1:
                            if array.ndim != 1:
                                raise ValueError(
                                    "Expected one-dimensional tracking array"
                                )
                        frame[field] = array
                    count = len(frame["track_id"])
                    if any(
                        len(frame[field]) != count
                        for field in TRACK_FIELDS
                        | (record.keys() & {"velocity_m_per_s"})
                    ):
                        raise ValueError(
                            "All tracking arrays must have the same length"
                        )
                    if len(set(frame["track_id"].tolist())) != count:
                        raise ValueError("Track IDs must be unique within a frame")
                    if any(
                        SCENARIO_MINING_CATEGORIES[label] != name
                        for label, name in zip(frame["label"], frame["name"])
                    ):
                        raise ValueError("Category labels and names disagree")
                    if np.any(frame["size"] < 0):
                        raise ValueError("Object dimensions must be nonnegative")
                    if "is_positive" in record:
                        if type(record["is_positive"]) is not bool:
                            raise ValueError("is_positive must be boolean")
                        frame["is_positive"] = record["is_positive"]
                frames.append(frame)
            sequences[str((log_id, prompt))] = sorted(
                frames, key=lambda frame: frame["timestamp_ns"]
            )
    return sequences
