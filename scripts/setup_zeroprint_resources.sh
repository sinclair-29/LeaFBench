#!/usr/bin/env bash

set -euo pipefail

MODEL_ROOT="${1:-/home/chj/LLMJailbreak/models}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

NLTK_DIR="${MODEL_ROOT}/nltk_data"
GLOVE_DIR="${MODEL_ROOT}/glove-wiki-gigaword-100"
GLOVE_FILE="${GLOVE_DIR}/glove-wiki-gigaword-100.gz"
GLOVE_MD5="40ec481866001177b8cd4cb0df92924f"
GLOVE_BASE_URL="https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100"
GENSIM_INFO_URL="https://raw.githubusercontent.com/RaRe-Technologies/gensim-data/master/list.json"

CURL_RETRY_OPTIONS=(--retry 10 --retry-delay 3)
if curl --help all 2>/dev/null | grep -q -- '--retry-all-errors'; then
    CURL_RETRY_OPTIONS+=(--retry-all-errors)
fi

echo "Using Python: $(${PYTHON_BIN} --version 2>&1)"
echo "Resource directory: ${MODEL_ROOT}"

mkdir -p "${NLTK_DIR}" "${GLOVE_DIR}"

echo "Installing Gensim and NLTK..."
"${PYTHON_BIN}" -m pip install \
    --index-url "${PIP_INDEX_URL}" \
    "gensim==4.4.0" \
    "nltk==3.9.2"

echo "Downloading NLTK resources..."
"${PYTHON_BIN}" -m nltk.downloader \
    -d "${NLTK_DIR}" \
    punkt \
    punkt_tab \
    stopwords \
    averaged_perceptron_tagger \
    averaged_perceptron_tagger_eng

if [[ -f "${GLOVE_FILE}" ]] && echo "${GLOVE_MD5}  ${GLOVE_FILE}" | md5sum --check --status; then
    echo "GloVe model already exists and passed checksum verification."
else
    echo "Downloading GloVe model..."
    curl -fL \
        "${CURL_RETRY_OPTIONS[@]}" \
        --continue-at - \
        -o "${GLOVE_FILE}.part" \
        "${GLOVE_BASE_URL}/glove-wiki-gigaword-100.gz"

    echo "${GLOVE_MD5}  ${GLOVE_FILE}.part" | md5sum --check
    mv "${GLOVE_FILE}.part" "${GLOVE_FILE}"
fi

echo "Downloading Gensim loader and metadata..."
curl -fL "${CURL_RETRY_OPTIONS[@]}" \
    -o "${GLOVE_DIR}/__init__.py" \
    "${GLOVE_BASE_URL}/__init__.py"
curl -fL "${CURL_RETRY_OPTIONS[@]}" \
    -o "${MODEL_ROOT}/information.json" \
    "${GENSIM_INFO_URL}"

echo "Verifying installed resources..."
NLTK_DATA="${NLTK_DIR}" GENSIM_DATA_DIR="${MODEL_ROOT}" \
"${PYTHON_BIN}" - <<'PY'
import gensim
import gensim.downloader as api
import nltk

resources = [
    "tokenizers/punkt",
    "tokenizers/punkt_tab",
    "corpora/stopwords",
    "taggers/averaged_perceptron_tagger",
    "taggers/averaged_perceptron_tagger_eng",
]
for resource in resources:
    nltk.data.find(resource)

model = api.load("glove-wiki-gigaword-100")
assert len(model) == 400000
assert model.vector_size == 100

print(f"Gensim {gensim.__version__}: OK")
print(f"NLTK {nltk.__version__}: OK")
print("GloVe vocabulary 400000, dimension 100: OK")
PY

if [[ ! -d "${MODEL_ROOT}/all-mpnet-base-v2" ]]; then
    echo "Warning: ${MODEL_ROOT}/all-mpnet-base-v2 is still missing and must be transferred separately."
fi

echo "ZeroPrint NLP resources are ready."
