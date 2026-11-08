# Scenario mining evaluation

Predictions use plain JSON, nested by log ID and exact prompt text. Each prompt contains a list of frames. Frame timestamps must cover the ground-truth split exactly; the evaluator aligns predictions by timestamp. Train and evaluation annotations keep their existing organizer-controlled format.

Track-free example:

```json
{
  "3b3570b4-7b0b-3268-a571-b0889dbf40b6": {
    "vehicle with a bicycle to its right": [
      {"timestamp_ns": 315969326019189000, "is_positive": true}
    ]
  }
}
```

Tracking frames contain `timestamp_ns`, `track_id`, `label`, `name`, `score`, `translation_m`, `size`, and `yaw`. The optional fields are `velocity_m_per_s`, `is_positive`, and `dtypes`. Arrays must have equal lengths; translation, size and velocity have shape N by 3. Empty vector arrays are represented by `[]`. Labels 0, 1 and 2 correspond to `REFERRED_OBJECT`, `RELATED_OBJECT` and `OTHER_OBJECT` respectively. Track IDs are unique nonnegative integers within a frame.

Numeric arrays default to float64 for floating-point fields and int64 for labels/IDs. To retain the precision of an existing prediction, a frame can declare, for example, `"dtypes": {"translation_m": "float32", "score": "float32", "track_id": "int32"}`. Names are strings. Ego translation is taken from ground truth rather than the submission.

Use the existing local evaluation entry point:

```bash
python -m av2.evaluation.scenario_mining.eval \
  --predictions predictions.json \
  --ground_truth annotations_testsplit.pkl \
  --dataset_dir /data/Sensor/test \
  --out evaluation_results
```

Ground-truth annotations are trusted local organizer files. JSON predictions are never deserialized as Python objects. Track-free evaluation bypasses maps, geometry and tracking metrics, returning zero for both HOTA metrics and the usual timestamp/log balanced accuracies.

The tracking path evaluates each prompt only on its own scenarios, reuses pairwise distances across score thresholds, and evaluates duplicate thresholds once. It uses the existing tracking threshold calculation and TrackEval HOTA/CLEAR implementations, including the original float32 reductions. Tracks never referred to by a prompt are discarded before geometry filtering because they cannot contribute to temporal or full-track HOTA. Each unique map is loaded once and shared between predictions and labels.
