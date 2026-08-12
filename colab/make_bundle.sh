#!/usr/bin/env bash
# Build the zip you upload to Colab. Run from the project root:
#   bash colab/make_bundle.sh
# Produces floret_torch_bundle.zip containing the package, pyproject.toml
# (needed for `pip install -e .`), bench scripts, tests, and the
# colab/farsi_weasel weasel pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f floret_torch_bundle.zip
zip -r floret_torch_bundle.zip \
    floret_torch bench tests colab/farsi_weasel pyproject.toml \
    -x '*__pycache__*' '*.pyc' >/dev/null
echo "wrote $(pwd)/floret_torch_bundle.zip ($(du -h floret_torch_bundle.zip | cut -f1))"
