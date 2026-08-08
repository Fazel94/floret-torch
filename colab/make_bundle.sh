#!/usr/bin/env bash
# Build the zip you upload to Colab. Run from the project root:
#   bash colab/make_bundle.sh
# Produces floret_torch_bundle.zip containing the package + bench scripts.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f floret_torch_bundle.zip
zip -r floret_torch_bundle.zip \
    floret_torch bench tests \
    -x '*__pycache__*' '*.pyc' >/dev/null
echo "wrote $(pwd)/floret_torch_bundle.zip ($(du -h floret_torch_bundle.zip | cut -f1))"
