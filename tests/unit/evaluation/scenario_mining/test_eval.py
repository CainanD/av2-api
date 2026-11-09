"""Scenario mining metric parity and JSON entry-point tests."""

import copy
import json
import pickle
from pathlib import Path
from unittest.mock import patch
from typing import Any, Dict, Tuple

import numpy as np
import pytest
from click.testing import CliRunner

from av2.evaluation.scenario_mining.eval import (
    classify_referred_objects,
    evaluate,
    filter_drivable_area,
    relabel_sequence_ids,
    runner,
)
from av2.evaluation.scenario_mining.serialization import load_predictions
from av2.evaluation.scenario_mining.thresholds import tune_score_thresholds
from av2.evaluation.tracking.eval import _tune_score_thresholds, filter_max_dist

TEST_DATA_DIR = Path(__file__).parent / "data"


def _fixture() -> Tuple[Any, Any]:
    with (TEST_DATA_DIR / "combined_predictions_dev.pkl").open("rb") as stream:
        predictions = pickle.load(stream)
    with (TEST_DATA_DIR / "combined_gt_dev.pkl").open("rb") as stream:
        labels = pickle.load(stream)
    return predictions, labels


def _write_json(predictions: Any, path: Path) -> None:
    document: Dict[str, Any] = {}
    for (log_id, prompt), frames in predictions.items():
        records = []
        for frame in frames:
            record: Dict[str, Any] = {"timestamp_ns": int(frame["timestamp_ns"])}
            dtypes = {}
            for field in (
                "translation_m",
                "size",
                "yaw",
                "score",
                "track_id",
                "label",
                "name",
                "is_positive",
            ):
                if field not in frame:
                    continue
                value = frame[field]
                record[field] = (
                    value.tolist() if isinstance(value, np.ndarray) else value
                )
                if isinstance(value, np.ndarray) and value.dtype.kind in "fiu":
                    dtypes[field] = str(value.dtype)
            if dtypes:
                record["dtypes"] = dtypes
            records.append(record)
        document.setdefault(log_id, {})[prompt] = records
    path.write_text(json.dumps(document))


def test_json_evaluate_preserves_scores_and_inputs(tmp_path: Path) -> None:
    """The real tracking fixture retains its four reference metric values."""
    predictions, labels = _fixture()
    path = tmp_path / "predictions.json"
    _write_json(predictions, path)
    decoded = load_predictions(path)
    before = pickle.dumps((decoded, labels))
    scores = evaluate(decoded, labels, "HOTA", 50, None, None)
    np.testing.assert_array_equal(
        scores, [0.37955018877983093, 0.39413923025131226, 0.85, 1.0]
    )
    assert pickle.dumps((decoded, labels)) == before


@pytest.mark.parametrize("objective", ["HOTA", "MOTA"])
@pytest.mark.parametrize("full_tracks", [False, True])
def test_threshold_tuner_matches_tracking_reference(
    objective: str, full_tracks: bool
) -> None:
    """Thresholds, optimal values and recall averages match the original tuner."""
    from av2.evaluation.scenario_mining.eval import referred_full_tracks

    predictions, labels = _fixture()
    # Include multiple sequences for one prompt and an entirely empty prompt.
    first = next(iter(labels))
    repeated = ("00000000-0000-0000-0000-000000000000", first[1])
    labels[repeated] = copy.deepcopy(labels[first])
    predictions[repeated] = copy.deepcopy(predictions[first])
    empty = (first[0], "empty prompt")
    labels[empty] = copy.deepcopy(labels[first])
    predictions[empty] = copy.deepcopy(predictions[first])
    for frame in labels[empty] + predictions[empty]:
        frame["label"] = np.full_like(frame["label"], 2)
    labels = filter_max_dist(relabel_sequence_ids(labels), 50)
    predictions = filter_max_dist(relabel_sequence_ids(predictions), 50)
    if full_tracks:
        labels, predictions = referred_full_tracks(labels), referred_full_tracks(
            predictions
        )
    labels, predictions = classify_referred_objects(labels), classify_referred_objects(
        predictions
    )
    classes = sorted({seq[42:-2] for seq in labels})
    expected = _tune_score_thresholds(
        labels, predictions, objective, classes, num_thresholds=10
    )
    actual = tune_score_thresholds(labels, predictions, objective, classes)
    assert expected[0] == actual[0]
    # Spawn workers initialize NumPy aliases independently; TrackEval may
    # reduce float64 arrays there versus float32 in the parent process.
    for expected_values, actual_values in zip(expected[1:], actual[1:]):
        assert expected_values.keys() == actual_values.keys()
        np.testing.assert_allclose(
            list(actual_values.values()),
            list(expected_values.values()),
            rtol=float(np.finfo(np.float32).eps),
            atol=float(np.finfo(np.float32).eps),
        )


def test_track_free_cli_and_timestamp_alignment(tmp_path: Path) -> None:
    """Track-free JSON needs no geometry or maps and is aligned by timestamp."""
    predictions, labels = _fixture()
    predictions = {
        key: [
            {
                "timestamp_ns": int(frame["timestamp_ns"]),
                "is_positive": bool(
                    frame.get("is_positive", np.any(frame["label"] == 0))
                ),
            }
            for frame in reversed(frames)
        ]
        for key, frames in labels.items()
    }
    path = tmp_path / "predictions.json"
    _write_json(predictions, path)
    ground_truth = tmp_path / "labels.pkl"
    ground_truth.write_bytes(pickle.dumps(labels))
    with patch(
        "av2.evaluation.scenario_mining.eval.load_mapped_avm_and_egoposes",
        side_effect=AssertionError("Maps not needed"),
    ):
        result = CliRunner().invoke(
            runner,
            [
                "--predictions",
                str(path),
                "--ground_truth",
                str(ground_truth),
                "--out",
                str(tmp_path / "results"),
            ],
        )
    assert result.exit_code == 0, result.output
    metrics = json.loads((tmp_path / "results/temporal_metrics.json").read_text())
    assert metrics["timestamp_balanced_accuracy_class_avg"] == 1.0
    assert "HOTA-Temporal: 0.00" in result.output


def test_reject_incomplete_timestamps() -> None:
    """Missing frames cannot silently shift timestamp-level evaluation."""
    predictions, labels = _fixture()
    predictions[next(iter(predictions))].pop()
    with pytest.raises(ValueError, match="timestamps"):
        evaluate(predictions, labels, "HOTA", 50, None, None)


def test_unique_maps() -> None:
    """Multiple prompts in one log must not rebuild that log's raster."""
    predictions, _ = _fixture()
    first = next(iter(predictions))
    predictions[(first[0], "another prompt")] = copy.deepcopy(predictions[first])
    for frames in predictions.values():
        for frame in frames:
            for key in (
                "translation_m",
                "size",
                "yaw",
                "label",
                "name",
                "track_id",
                "score",
            ):
                frame[key] = frame[key][:0]
    with patch(
        "av2.evaluation.scenario_mining.eval.load_mapped_avm_and_egoposes"
    ) as load:
        load.return_value = ({key[0]: object() for key in predictions}, {})
        filter_drivable_area(relabel_sequence_ids(predictions), "unused")
        log_ids = load.call_args.args[0]
        assert len(log_ids) == len(set(log_ids)) == 2


@pytest.mark.parametrize(
    "mutation",
    ["duplicate_timestamp", "bad_dtype", "nonfinite", "wrong_shape", "wrong_label"],
)
def test_invalid_json(tmp_path: Path, mutation: str) -> None:
    """Reject malformed arrays before passing predictions to TrackEval."""
    predictions, _ = _fixture()
    path = tmp_path / "predictions.json"
    _write_json(predictions, path)
    document = json.loads(path.read_text())
    records = next(iter(next(iter(document.values())).values()))
    frame = next(frame for frame in records if frame["track_id"])
    if mutation == "duplicate_timestamp":
        records.append(records[0])
    elif mutation == "bad_dtype":
        frame["dtypes"]["score"] = "object"
    elif mutation == "nonfinite":
        frame["score"][0] = float("nan")
    elif mutation == "wrong_shape":
        frame["translation_m"][0] = [1, 2]
    else:
        frame["label"][0] = 3
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_predictions(path)


def test_roi_evaluate_preserves_scores() -> None:
    """Map filtering retains the real fixture's reference metric values."""
    predictions, labels = _fixture()
    scores = evaluate(predictions, labels, "HOTA", 50, str(TEST_DATA_DIR), None)
    np.testing.assert_array_equal(
        scores, [0.3831658661365509, 0.4016636162996292, 0.85, 1.0]
    )
