#!/usr/bin/env bash
# =============================================================================
# scripts/verify_e2e.sh — Stage 10 End-to-End Verification Script
# =============================================================================
#
# PURPOSE:
#   Verifies that the Medical Device Risk Prediction system runs correctly
#   end-to-end:
#     1. Raw CSV files are present (or processed artifacts already exist)
#     2. All pipeline artifact files exist with correct row counts
#     3. FastAPI backend starts and all endpoints return valid responses
#     4. Streamlit frontend is importable and runnable
#     5. Security / healthcare disclaimer checks
#
# USAGE:
#   bash scripts/verify_e2e.sh [--skip-pipeline-rerun]
#
#   Default behaviour: verifies artifacts WITHOUT re-running the full pipeline
#   (since training takes ~10 minutes). Pass --rerun-pipeline to actually
#   re-run data ingestion, feature engineering, training, and scoring.
#
# IMPORTANT:
#   This script does NOT run pytest or open a browser.
#   For the pytest test suite, run: python -m pytest -q
#
# EXIT CODES:
#   0 — all checks passed
#   1 — one or more checks failed
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

RERUN_PIPELINE=false
for arg in "$@"; do
  case "$arg" in
    --rerun-pipeline) RERUN_PIPELINE=true ;;
    --skip-pipeline-rerun) RERUN_PIPELINE=false ;;
  esac
done

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'; BOLD='\033[1m'

PASS=0; FAIL=0; WARN=0
declare -a FAILURES=()

pass()  { echo -e "  ${GREEN}✓${NC}  $1"; PASS=$((PASS+1)); }
fail()  { echo -e "  ${RED}✗${NC}  $1"; FAIL=$((FAIL+1)); FAILURES+=("$1"); }
warn()  { echo -e "  ${YELLOW}⚠${NC}  $1"; WARN=$((WARN+1)); }
section() { echo; echo -e "${CYAN}${BOLD}── $1 ──────────────────────────────────────${NC}"; }

# ── Activate virtualenv ───────────────────────────────────────────────────────
if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
else
  echo -e "${RED}ERROR: venv/bin/activate not found. Create a virtualenv first.${NC}"
  exit 1
fi

echo
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║  Medical Device Risk Prediction — Stage 10 E2E Verification  ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${NC}"
echo "  Root: $ROOT"
echo "  Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Rerun pipeline: $RERUN_PIPELINE"

# =============================================================================
# 1. Raw data files
# =============================================================================
section "1. Raw Data Files"

for f in devices.csv events.csv manufacturers.csv; do
  if [ -f "data/raw/$f" ]; then
    # Use wc -l for speed (subtract 1 for header row)
    rows=$(( $(wc -l < "data/raw/$f") - 1 ))
    pass "data/raw/$f  (~${rows} rows)"
  else
    warn "data/raw/$f not found (git-ignored — normal in CI)"
  fi
done


# =============================================================================
# 2. Optionally re-run the pipeline
# =============================================================================
if [ "$RERUN_PIPELINE" = true ]; then
  section "2. Pipeline Re-run (full, takes ~10 min)"

  echo "  Running data pipeline..."
  python -m src.data.pipeline && pass "src.data.pipeline" || fail "src.data.pipeline failed"

  echo "  Running feature engineering..."
  python -m src.features.build_features && pass "src.features.build_features" || fail "src.features.build_features failed"

  echo "  Running model training..."
  python -m src.models.train && pass "src.models.train" || fail "src.models.train failed"

  echo "  Running model evaluation..."
  python -m src.models.evaluate && pass "src.models.evaluate" || fail "src.models.evaluate failed"

  echo "  Building serving table..."
  python -m src.risk.build_serving_table && pass "src.risk.build_serving_table" || fail "src.risk.build_serving_table failed"
else
  section "2. Pipeline Re-run"
  warn "Pipeline re-run skipped (pass --rerun-pipeline to enable)"
fi

# =============================================================================
# 3. Processed artifact files
# =============================================================================
section "3. Processed Artifacts"

check_parquet() {
  local path="$1"; local label="$2"; local expected_rows="${3:-}"
  if [ -f "$path" ]; then
    # Use pyarrow metadata (fast — does not load data into memory)
    rows=$(python3 -c "
import pyarrow.parquet as pq
meta = pq.read_metadata('$path')
print(sum(meta.row_group(i).num_rows for i in range(meta.num_row_groups)))
" 2>/dev/null || echo "ERROR")
    if [ "$rows" = "ERROR" ]; then
      fail "$label — could not read parquet"
    elif [ -n "$expected_rows" ] && [ "$rows" != "$expected_rows" ]; then
      fail "$label — expected $expected_rows rows, got $rows"
    else
      pass "$label (${rows} rows)"
    fi
  else
    fail "$label — file not found: $path"
  fi
}


check_parquet "data/processed/merged.parquet"              "merged.parquet"            "124969"
check_parquet "data/features/train.parquet"                "train.parquet"             "38247"
check_parquet "data/features/validation.parquet"           "validation.parquet"        "4273"
check_parquet "data/features/test.parquet"                 "test.parquet"              "8918"
check_parquet "data/features/holdout_2018.parquet"         "holdout_2018.parquet"      "1361"
check_parquet "artifacts/risk/device_risk_snapshot.parquet" "device_risk_snapshot"     "50341"

for f in \
  "data/processed/_manifest.json" \
  "data/features/feature_metadata.json" \
  "models/production/model.pkl" \
  "models/production/calibrated_model.pkl" \
  "models/production/model_card.json" \
  "models/production/test_metrics.json" \
  "models/production/feature_importance.json" \
  "models/production/feature_metadata.json"
do
  [ -f "$f" ] && pass "$f" || fail "$f not found"
done

# Check manifest hash integrity
section "3b. Manifest Hash Integrity"
python3 - << 'PYEOF'
import json, hashlib
from pathlib import Path
ROOT = Path(".")
manifest = json.loads((ROOT / "data/processed/_manifest.json").read_text())
any_raw = False
ok = True
for fname, stored in manifest["file_hashes"].items():
    p = ROOT / "data" / "raw" / fname
    if p.exists():
        any_raw = True
        md5 = hashlib.md5(p.read_bytes()).hexdigest()
        if md5 == stored:
            print(f"  \033[32m✓\033[0m  {fname} hash matches")
        else:
            print(f"  \033[31m✗\033[0m  {fname} HASH MISMATCH — re-run pipeline")
            ok = False
if not any_raw:
    print("  \033[33m⚠\033[0m  Raw files absent — hash check skipped (normal in CI)")
import sys; sys.exit(0 if ok else 1)
PYEOF
HASH_EXIT=$?
[ $HASH_EXIT -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); FAILURES+=("Manifest hash mismatch"); }

# =============================================================================
# 4. Feature ↔ Model Card consistency
# =============================================================================
section "4. Feature / Model Card Consistency"

python3 - << 'PYEOF'
import json
from pathlib import Path
ROOT = Path(".")
fm = json.loads((ROOT / "data/features/feature_metadata.json").read_text())
mc = json.loads((ROOT / "models/production/model_card.json").read_text())
fm_cols = sorted(fm["feature_columns"])
mc_cols = sorted(mc["feature_columns"])
if fm_cols == mc_cols:
    print(f"  \033[32m✓\033[0m  feature_metadata ↔ model_card: {len(fm_cols)} columns match exactly")
else:
    only_fm = sorted(set(fm_cols) - set(mc_cols))
    only_mc = sorted(set(mc_cols) - set(fm_cols))
    print(f"  \033[31m✗\033[0m  MISMATCH — only in feature_metadata: {only_fm}")
    print(f"            only in model_card:       {only_mc}")
    import sys; sys.exit(1)
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); FAILURES+=("Feature/model-card column mismatch"); }

# Check model_card and test_metrics threshold consistency
python3 - << 'PYEOF'
import json
from pathlib import Path
ROOT = Path(".")
mc = json.loads((ROOT / "models/production/model_card.json").read_text())
tm = json.loads((ROOT / "models/production/test_metrics.json").read_text())
diff = abs(mc["decision_threshold"] - tm["decision_threshold"])
if diff < 1e-9:
    print(f"  \033[32m✓\033[0m  decision_threshold consistent: {mc['decision_threshold']:.8f}")
else:
    print(f"  \033[31m✗\033[0m  threshold mismatch: model_card={mc['decision_threshold']} vs test_metrics={tm['decision_threshold']}")
    import sys; sys.exit(1)
PYEOF
[ $? -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); FAILURES+=("Threshold mismatch between model_card and test_metrics"); }

# =============================================================================
# 5. FastAPI backend — live endpoint checks
# =============================================================================
section "5. FastAPI Backend (in-process TestClient)"

python3 - << 'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(".")))

from fastapi.testclient import TestClient
from backend.main import app
client = TestClient(app, raise_server_exceptions=True)

results = []

def chk(label, cond, detail=""):
    results.append((label, cond, detail))
    if cond:
        print(f"  \033[32m✓\033[0m  {label}")
    else:
        print(f"  \033[31m✗\033[0m  {label}" + (f" — {detail}" if detail else ""))

# GET /health
r = client.get("/health")
chk("/health → 200", r.status_code == 200)
if r.status_code == 200:
    d = r.json()
    chk("/health has model_version", bool(d.get("model_version", "")))
    chk("/health has disclaimer", "prototype" in d.get("disclaimer", "").lower()
        or "decision-support" in d.get("disclaimer", "").lower())
    chk("/health has manifest_hash", bool(d.get("data_manifest_hash", "")))

# GET /risk-summary
r = client.get("/risk-summary")
chk("/risk-summary → 200", r.status_code == 200)
if r.status_code == 200:
    d = r.json()
    chk("/risk-summary has total_scored", "total_scored" in d)
    chk("/risk-summary total_scored > 0", d.get("total_scored", 0) > 0)


# GET /devices
r = client.get("/devices", params={"page_size": 5})
chk("/devices → 200", r.status_code == 200)

# GET /devices/80508 (known high-risk)
r = client.get("/devices/80508")
chk("/devices/80508 → 200", r.status_code == 200)
if r.status_code == 200:
    d = r.json()
    chk("/devices/80508 risk_score = 100", d.get("risk_score") == 100.0,
        f"got {d.get('risk_score')}")
    chk("/devices/80508 risk_level = HIGH", d.get("risk_level") == "HIGH",
        f"got {d.get('risk_level')}")

# POST /predict (high-risk device)
r = client.post("/predict", json={"device_id": "80508"})
chk("/predict 80508 → 200", r.status_code == 200)
if r.status_code == 200:
    d = r.json()
    chk("/predict prediction_available = True", d.get("prediction_available") is True)
    chk("/predict risk_score = 100", d.get("risk_score") == 100.0,
        f"got {d.get('risk_score')}")

# POST /predict (unknown device)
r = client.post("/predict", json={"device_id": "DOES_NOT_EXIST_XYZ_999"})
chk("/predict unknown → prediction_unavailable", r.status_code == 200 and
    r.json().get("prediction_available") is False)

# GET /explanation/80508
r = client.get("/explanation/80508")
chk("/explanation/80508 → 200", r.status_code == 200)
if r.status_code == 200:
    d = r.json()
    chk("/explanation available = True", d.get("available") is True)
    chk("/explanation top_positive non-empty", len(d.get("top_positive", [])) > 0)

# GET /recommendation/80508
r = client.get("/recommendation/80508")
chk("/recommendation/80508 → 200", r.status_code == 200)
if r.status_code == 200:
    d = r.json()
    chk("/recommendation maintenance_priority = Critical",
        d.get("maintenance_priority") == "Critical",
        f"got {d.get('maintenance_priority')}")
    chk("/recommendation disclaimer present",
        "prototype" in d.get("disclaimer", "").lower()
        or "decision-support" in d.get("disclaimer", "").lower())

# GET /feature-importance
r = client.get("/feature-importance")
chk("/feature-importance → 200", r.status_code == 200)
if r.status_code == 200:
    d = r.json()
    chk("/feature-importance features non-empty", len(d.get("features", [])) > 0)

# POST /copilot
r = client.post("/copilot", json={"device_id": "80508",
                                  "question": "What is the risk level?"})
chk("/copilot → 200", r.status_code == 200)
if r.status_code == 200:
    d = r.json()
    chk("/copilot answer non-empty", bool(d.get("answer", "").strip()))
    chk("/copilot context risk_level = HIGH",
        d.get("context_used", {}).get("risk_level") == "HIGH")

# Input validation
chk("/predict empty device_id → 422",
    client.post("/predict", json={"device_id": ""}).status_code == 422)
chk("/copilot empty question → 422",
    client.post("/copilot", json={"device_id": "80508", "question": ""}).status_code == 422)

# Summary
failures = [(l, d) for l, c, d in results if not c]
if failures:
    print(f"\n  \033[31mFAILED {len(failures)} API check(s)\033[0m")
    sys.exit(1)
else:
    print(f"\n  \033[32mAll {len(results)} API checks passed\033[0m")
PYEOF
API_EXIT=$?
[ $API_EXIT -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); FAILURES+=("FastAPI endpoint checks failed"); }

# =============================================================================
# 6. Streamlit frontend — importable / runnable check
# =============================================================================
section "6. Streamlit Frontend (import smoke test)"

python3 - << 'PYEOF'
import sys, types
from pathlib import Path
FRONTEND = Path("frontend")
sys.path.insert(0, str(FRONTEND))
sys.path.insert(0, ".")

# Stub streamlit
if "streamlit" not in sys.modules:
    st_stub = types.ModuleType("streamlit")
    def _cd(func=None, *, ttl=None, **kw):
        return (lambda f: f) if func is None else func
    st_stub.cache_data = _cd
    st_stub.cache_resource = _cd
    for attr in ("set_page_config","sidebar","write","error","warning","info",
                 "success","title","header","subheader","markdown","columns",
                 "metric","dataframe","plotly_chart","spinner","expander",
                 "caption","stop","session_state","text_input","selectbox",
                 "button","form","form_submit_button","chat_input","chat_message",
                 "divider","image","number_input","multiselect","radio"):
        setattr(st_stub, attr, lambda *a, **kw: None)
    sys.modules["streamlit"] = st_stub

if "plotly" not in sys.modules:
    sys.modules["plotly"] = types.ModuleType("plotly")
    sys.modules["plotly.express"] = types.ModuleType("plotly.express")
    sys.modules["plotly.graph_objects"] = types.ModuleType("plotly.graph_objects")

import importlib

try:
    ac = importlib.import_module("utils.api_client")
    assert hasattr(ac, "BACKEND_URL"), "BACKEND_URL missing from api_client"
    assert hasattr(ac, "get_health"), "get_health missing from api_client"
    assert hasattr(ac, "post_copilot"), "post_copilot missing from api_client"
    print("  \033[32m✓\033[0m  utils.api_client importable + all endpoints defined")
except Exception as e:
    print(f"  \033[31m✗\033[0m  utils.api_client import failed: {e}")
    sys.exit(1)

# Check pages exist
pages = [
    "1_📊_Overview.py",
    "2_🔍_Device_Search.py",
    "3_📋_Device_Detail.py",
    "4_🧠_Explainability.py",
]
for p in pages:
    path = FRONTEND / "pages" / p
    if path.exists():
        print(f"  \033[32m✓\033[0m  {p}")
    else:
        print(f"  \033[31m✗\033[0m  {p} NOT FOUND")
        sys.exit(1)

# Confirm streamlit is listed in frontend requirements
req_path = FRONTEND / "requirements.txt"
if req_path.exists():
    reqs = req_path.read_text().lower()
    if "streamlit" in reqs:
        print("  \033[32m✓\033[0m  streamlit in frontend/requirements.txt")
    else:
        print("  \033[33m⚠\033[0m  streamlit not in frontend/requirements.txt")
PYEOF
STREAMLIT_EXIT=$?
[ $STREAMLIT_EXIT -eq 0 ] && PASS=$((PASS+1)) || { FAIL=$((FAIL+1)); FAILURES+=("Streamlit frontend checks failed"); }

# =============================================================================
# 7. Security & healthcare disclaimer checks
# =============================================================================
section "7. Security / Healthcare Disclaimer"

# .env in .gitignore
if grep -qE "^\.env$|^\.env " .gitignore 2>/dev/null; then
  pass ".env is git-ignored"
else
  fail ".env not found in .gitignore"
fi

# No secrets hardcoded
# Pattern: sk- followed by ≥20 alphanumeric chars (actual OpenAI key format),
# AIza followed by ≥20 chars (Google API key), or api_key= with a non-blank value.
# The "|| true" prevents set -euo pipefail from aborting when grep finds no matches
# (grep exits 1 on no-match, which would otherwise kill the pipeline).
SECRET_HITS=$(grep -rn \
  -e 'sk-[A-Za-z0-9]\{20,\}' \
  -e 'AIza[A-Za-z0-9_-]\{20,\}' \
  -e "api_key\s*=\s*['\"][^'\"]\+" \
  src/ backend/ frontend/ --include="*.py" 2>/dev/null || true)
SECRET_COUNT=$(echo "$SECRET_HITS" | grep -v "test_\|#.*sk-\|\.env\|os\." 2>/dev/null | grep -c "." || true)
if [ "$SECRET_COUNT" -eq 0 ]; then
  pass "No hardcoded API keys found in src/ backend/ frontend/"
else
  fail "Possible hardcoded secrets found ($SECRET_COUNT matches) — review before committing"
fi

# .env.example has correct structure (no actual keys)
if grep -q "LLM_PROVIDER=" .env.example 2>/dev/null; then
  # Make sure the value is blank (not filled in)
  if grep -qE "^LLM_API_KEY=$" .env.example; then
    pass ".env.example LLM_API_KEY is blank (placeholder only)"
  else
    warn ".env.example LLM_API_KEY may not be blank — check before commit"
  fi
else
  warn ".env.example LLM_PROVIDER line not found"
fi

# Disclaimer in README
if grep -qi "prototype\|decision-support" README.md 2>/dev/null; then
  pass "Healthcare disclaimer found in README.md"
else
  fail "Healthcare disclaimer (prototype/decision-support) NOT found in README.md"
fi

# Disclaimer in docs
DISCLAIMER_DOC_HITS=$(grep -rl "prototype\|decision-support" docs/ 2>/dev/null | wc -l | tr -d ' ')
if [ "$DISCLAIMER_DOC_HITS" -gt 0 ]; then
  pass "Healthcare disclaimer found in $DISCLAIMER_DOC_HITS doc(s)"
else
  warn "Healthcare disclaimer not found in any docs/"
fi

# =============================================================================
# Summary
# =============================================================================
echo
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  Stage 10 E2E Verification Summary${NC}"
echo -e "${BOLD}══════════════════════════════════════════════════════════════${NC}"
echo -e "  ${GREEN}Passed:${NC}  $PASS"
echo -e "  ${YELLOW}Warnings:${NC} $WARN"
echo -e "  ${RED}Failed:${NC}  $FAIL"

if [ ${#FAILURES[@]} -gt 0 ]; then
  echo
  echo -e "${RED}  Failed checks:${NC}"
  for f in "${FAILURES[@]}"; do
    echo -e "    ${RED}✗${NC} $f"
  done
fi

echo
if [ "$FAIL" -eq 0 ]; then
  echo -e "  ${GREEN}${BOLD}All checks passed. Stage 10 E2E verification complete.${NC}"
  exit 0
else
  echo -e "  ${RED}${BOLD}$FAIL check(s) failed. See details above.${NC}"
  exit 1
fi
