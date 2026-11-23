#!/usr/bin/env bash
set -euo pipefail
# This is deliberately a guard/template, not an in-place source rewriter.
# Build each executable from a clean source snapshot after making pangulu_cuda.h
# launch constants overrideable with #ifndef. Record the exact commit and flags.
repo=${1:-/openbayes/home/Profiliing}
printf '%s\n' "Required binaries:" \
  "$repo/test/PanguLU.baseline" \
  "$repo/test/PanguLU.launch_low" \
  "$repo/test/PanguLU.launch_high" \
  "See README_zh.md section 2. This script never edits the working tree automatically."
exit 0
