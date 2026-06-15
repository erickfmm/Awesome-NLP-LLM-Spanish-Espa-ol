#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/out"

if ! command -v quarkdown >/dev/null 2>&1; then
  echo "Error: quarkdown no está instalado o no está en PATH." >&2
  echo "Instálalo según: https://github.com/iamgio/quarkdown" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

shopt -s nullglob
for qd_file in "${SCRIPT_DIR}"/*.qd; do
  echo "Compilando ${qd_file}..."
  quarkdown c "${qd_file}" --pdf -o "${OUTPUT_DIR}"
done

echo "Listo. PDFs generados en: ${OUTPUT_DIR}"
