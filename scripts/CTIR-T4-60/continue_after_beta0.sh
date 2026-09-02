#!/bin/bash
set -euo pipefail

BETA0_PID=${1:-}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
STATE=experiments/ctir_t4_60/logs/pipeline.state

if [ -n "$BETA0_PID" ]; then
    echo "beta0 running pid=$BETA0_PID $(date --iso-8601=seconds)" > "$STATE"
    while kill -0 "$BETA0_PID" 2>/dev/null; do
        sleep 30
    done
fi

python - <<'PY'
import json
import math
path='experiments/ctir_t4_60/correctness/beta0/correctness.jsonl'
rows=[json.loads(x) for x in open(path) if x.strip()]
row=[x for x in rows if x['test']=='beta0_equivalence'][-1]
assert row['relative_error'] <= 1e-5, row
metrics=[json.loads(x) for x in open('experiments/ctir_t4_60/correctness/beta0/step_metrics.jsonl') if x.strip()][-1]
assert metrics['local_step'] == 2, metrics
assert math.isfinite(metrics['raw_update_fro_norm']) and metrics['raw_update_fro_norm'] > 0, metrics
assert math.isfinite(metrics['frob_ratio']), metrics
PY

echo "spectrum launching $(date --iso-8601=seconds)" > "$STATE"
bash scripts/CTIR-T4-60/run_correctness.sh spectrum
python - <<'PY'
import json
import math
path='experiments/ctir_t4_60/correctness/spectrum/correctness.jsonl'
rows=[json.loads(x) for x in open(path) if x.strip() and json.loads(x).get('test')=='exact_full_spectrum']
assert len(rows) == 2, rows
for row in rows:
    assert row['local_step'] == 2, row
    assert row.get('measurement_dtype') == 'float64', row
    assert row['spectrum_relative_error'] <= 1e-5, row
    assert abs(row['frob_ratio'] - 1.0) <= 1e-5, row
metrics=[json.loads(x) for x in open('experiments/ctir_t4_60/correctness/spectrum/step_metrics.jsonl') if x.strip()][-1]
assert metrics['local_step'] == 2, metrics
assert math.isfinite(metrics['raw_update_fro_norm']) and metrics['raw_update_fro_norm'] > 0, metrics
PY

echo "formal launching $(date --iso-8601=seconds)" > "$STATE"
exec bash scripts/CTIR-T4-60/run.sh
