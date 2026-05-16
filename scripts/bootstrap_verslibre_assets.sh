#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for missing runtime assets required by Dockerfile.verslibre
# and run_stressed_gpt_poetry_generation_v3.py.
#
# Usage:
#   scripts/bootstrap_verslibre_assets.sh
#   MODELS_DIR=./models TMP_DIR=./tmp scripts/bootstrap_verslibre_assets.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-${ROOT_DIR}/models}"
TMP_DIR="${TMP_DIR:-${ROOT_DIR}/tmp}"
DICT_DIR="${DICT_DIR:-${ROOT_DIR}/data/poetry/dict}"

mkdir -p "${MODELS_DIR}" "${TMP_DIR}" "${DICT_DIR}" "${TMP_DIR}/stress_model"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1" >&2
    exit 1
  }
}

need_cmd curl
need_cmd python3

download_if_missing() {
  local target="$1"
  local url="$2"
  if [[ -f "${target}" ]]; then
    echo "OK: already exists ${target}"
    return 0
  fi
  echo "Downloading ${target}"
  mkdir -p "$(dirname "${target}")"
  curl -fL --retry 3 --retry-delay 2 -o "${target}" "${url}"
}

try_download_if_missing() {
  local target="$1"
  local url="$2"
  if [[ -f "${target}" ]]; then
    echo "OK: already exists ${target}"
    return 0
  fi
  echo "Downloading ${target}"
  mkdir -p "$(dirname "${target}")"
  if ! curl -fL --retry 3 --retry-delay 2 -o "${target}" "${url}"; then
    echo "WARNING: failed to download ${target} from ${url}"
    return 1
  fi
}

echo "== Bootstrap Verslibre assets =="
echo "ROOT_DIR=${ROOT_DIR}"

# 1) UDPipe model (required by py/generative_poetry/udpipe_parser.py)
try_download_if_missing \
  "${MODELS_DIR}/udpipe_syntagrus.model" \
  "https://raw.githubusercontent.com/Koziev/verslibre/master/models/udpipe_syntagrus.model" || true

# 2) Seeds for suggestion generator
try_download_if_missing \
  "${MODELS_DIR}/seeds.pkl" \
  "https://raw.githubusercontent.com/Koziev/verslibre/master/models/seeds.pkl" || true

# 3) Lemma dictionary for udpipe parser
try_download_if_missing \
  "${MODELS_DIR}/word2lemma.pkl" \
  "https://raw.githubusercontent.com/Koziev/verslibre/master/models/word2lemma.pkl" || true

# 4) Poetry quality filters data files
try_download_if_missing \
  "${DICT_DIR}/bad_signature1.dat" \
  "https://raw.githubusercontent.com/Koziev/verslibre/master/data/poetry/dict/bad_signature1.dat" || true
try_download_if_missing \
  "${DICT_DIR}/bad_alignment2.dat" \
  "https://raw.githubusercontent.com/Koziev/verslibre/master/data/poetry/dict/bad_alignment2.dat" || true

# 5) Stress model + accents dictionary from HF (inkoziev/accentuator)
# We first try direct links; if unavailable, provide clear instructions.
HF_BASE="https://huggingface.co/inkoziev/accentuator/resolve/main"

if [[ ! -f "${TMP_DIR}/accents.pkl" ]]; then
  echo "Attempting to download accents.pkl from Hugging Face"
  if ! curl -fL --retry 3 --retry-delay 2 -o "${TMP_DIR}/accents.pkl" "${HF_BASE}/accents.pkl"; then
    echo "WARNING: could not fetch accents.pkl from ${HF_BASE}/accents.pkl"
  fi
else
  echo "OK: already exists ${TMP_DIR}/accents.pkl"
fi

if [[ ! -d "${TMP_DIR}/stress_model/nn_stress.model" ]]; then
  echo "Attempting to fetch stress model bundle from Hugging Face"
  ARCHIVE="${TMP_DIR}/stress_model_bundle.zip"
  if curl -fL --retry 3 --retry-delay 2 -o "${ARCHIVE}" "${HF_BASE}/stress_model.zip"; then
    python3 - <<PY
import zipfile, os
archive = r"${ARCHIVE}"
target = r"${TMP_DIR}/stress_model"
os.makedirs(target, exist_ok=True)
with zipfile.ZipFile(archive) as zf:
    zf.extractall(target)
print("Extracted", archive, "to", target)
PY
    rm -f "${ARCHIVE}"
  else
    echo "WARNING: could not fetch stress_model.zip from ${HF_BASE}/stress_model.zip"
  fi
else
  echo "OK: already exists ${TMP_DIR}/stress_model/nn_stress.model"
fi

echo
echo "== Validation =="
missing=0
for f in \
  "${MODELS_DIR}/udpipe_syntagrus.model" \
  "${MODELS_DIR}/seeds.pkl" \
  "${MODELS_DIR}/word2lemma.pkl" \
  "${DICT_DIR}/collocation_accents.dat" \
  "${DICT_DIR}/bad_signature1.dat" \
  "${DICT_DIR}/bad_alignment2.dat" \
  "${TMP_DIR}/accents.pkl"
do
  if [[ -f "${f}" ]]; then
    echo "OK: ${f}"
  else
    echo "MISSING: ${f}"
    missing=1
  fi
done

if [[ ! -d "${TMP_DIR}/stress_model" ]]; then
  echo "MISSING: ${TMP_DIR}/stress_model"
  missing=1
else
  echo "OK: ${TMP_DIR}/stress_model"
fi

if [[ "${missing}" -ne 0 ]]; then
  echo
  echo "Some artifacts are still missing."
  echo "If downloads are blocked, fetch manually from:"
  echo "  - https://github.com/Koziev/verslibre"
  echo "  - https://huggingface.co/inkoziev/accentuator"
  exit 2
fi

echo "All required bootstrap artifacts are present."
