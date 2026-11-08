"""Tune scenario scores without evaluating unrelated prompt/sequence pairs."""

from typing import Dict, List, Tuple

import numpy as np
import trackeval
from functools import partial

from av2.evaluation.scenario_mining import SEQ_ID_PROMPT_INDICES
from av2.evaluation.tracking.constants import SUBMETRIC_TO_METRIC_CLASS_NAME
from av2.evaluation.tracking.eval import (
    TrackEvalDataset,
    _calculate_score_thresholds,
    _xy_center_similarity,
)
from av2.evaluation.typing import Sequences


def tune_score_thresholds(
    labels: Sequences,
    predictions: Sequences,
    objective_metric: str,
    classes: List[str],
    num_thresholds: int = 10,
    match_distance_m: int = 2,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Apply the tracking tuner's thresholds and aggregation one prompt at a time.

    Inputs have already been restricted to referred objects and classified by
    prompt. Other prompts therefore contribute only empty sequences to a class;
    their HOTA/CLEAR counts and weights are zero. Omitting those pairs preserves
    the metrics while avoiding work proportional to prompts times scenarios.
    Raw geometry is shared across thresholds, but IDs are remapped separately
    after each threshold, just as in the tracking evaluator.
    """
    metric_class = SUBMETRIC_TO_METRIC_CLASS_NAME[objective_metric]
    metric = getattr(trackeval.metrics, metric_class)(
        {"THRESHOLD": 0.5, "PRINT_CONFIG": False}
    )
    empty_result = metric.eval_sequence(
        {"num_tracker_dets": 0, "num_gt_dets": 0, "num_gt_ids": 0}
    )
    all_sequences = sorted(labels)
    sequences_by_prompt: Dict[str, List[str]] = {name: [] for name in classes}
    for seq_id in labels:
        sequences_by_prompt[seq_id[SEQ_ID_PROMPT_INDICES]].append(seq_id)

    optimal_thresholds, optimal_values, mean_values = {}, {}, {}
    similarity = partial(_xy_center_similarity, zero_distance=match_distance_m)
    for name in classes:
        seq_ids = sequences_by_prompt[name]
        class_labels = {seq: labels[seq] for seq in seq_ids}
        class_predictions = {seq: predictions[seq] for seq in seq_ids}
        thresholds = _calculate_score_thresholds(
            class_labels, class_predictions, similarity, num_thresholds=num_thresholds
        )
        dataset = TrackEvalDataset(
            {
                **TrackEvalDataset.get_default_dataset_config(),
                "GT_TRACKS": {"tracker": class_labels},
                "PREDICTED_TRACKS": {"tracker": class_predictions},
                "SEQ_IDS_TO_EVAL": seq_ids,
                "CLASSES_TO_EVAL": [name],
                "TRACKERS_TO_EVAL": ["tracker"],
                "OUTPUT_FOLDER": "tmp",
            }
        )
        # Bound cached geometry to one prompt, not the complete dataset.
        raw_sequences = {
            seq: dataset.get_raw_seq_data("tracker", seq) for seq in sorted(seq_ids)
        }
        values_by_threshold = {}
        values = []
        for threshold in thresholds:
            if threshold not in values_by_threshold:
                results = {}
                for seq, raw in raw_sequences.items():
                    keep = [
                        scores >= threshold for scores in raw["tracker_confidences"]
                    ]
                    filtered = dict(raw)
                    for key in (
                        "tracker_ids",
                        "tracker_classes",
                        "tracker_dets",
                        "tracker_confidences",
                    ):
                        filtered[key] = [
                            array[mask] for array, mask in zip(raw[key], keep)
                        ]
                    filtered["similarity_scores"] = [
                        array[:, mask]
                        for array, mask in zip(raw["similarity_scores"], keep)
                    ]
                    data = dataset.get_preprocessed_seq_data(filtered, name)
                    results[seq] = metric.eval_sequence(data)
                # Preserve the reference aggregation order, including empty
                # contributions, without processing unrelated frame arrays.
                ordered_results = {
                    seq: results.get(seq, empty_result) for seq in all_sequences
                }
                combined = metric.combine_sequences(ordered_results)[objective_metric]
                values_by_threshold[threshold] = np.mean(combined)
            values.append(values_by_threshold[threshold])
        optimal_thresholds[name] = float(thresholds[np.argmax(values)])
        optimal_values[name] = float(max(0, np.max(values)))
        mean_values[name] = float(np.nanmean(np.clip(values, 0, None)))
    return optimal_thresholds, optimal_values, mean_values
