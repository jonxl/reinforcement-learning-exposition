#!/usr/bin/env bash
# Keep manual builds separate from VimTeX's continuous .build compiler.
set -euo pipefail
source_file=$(realpath "${1:?Usage: bash scripts/build-latex.sh path/to/document.tex}")
source_dir=$(dirname "$source_file")
source_name=$(basename "$source_file")
job_name=${source_name%.tex}
cd "$source_dir"
mkdir -p .build/manual
exec 9>.build/manual/build.lock
flock 9
latexmk -pdf -interaction=nonstopmode -halt-on-error \
    -auxdir=.build/manual -outdir=.build/manual "$source_name"
# Publish only the completed PDF; never copy auxiliary files to the editor build.
cp ".build/manual/$job_name.pdf" ".build/manual/$job_name.publish.pdf"
mv ".build/manual/$job_name.publish.pdf" "$job_name.pdf"
