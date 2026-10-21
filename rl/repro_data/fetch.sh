#!/bin/bash
# Refresh rl/repro_data/ from the network volume (s3://<volume>/graft-repro/output/).
# Pulls only the small per-run files: <sweep>/<run>/{run_config.yaml,routing_eval.jsonl},
# <sweep>_fseval/*.json and expr_only_probe_arms/*.json for the runs in runs.csv.
# Checkpoints, optimizer state, eval_samples.jsonl, probe samples and smoke runs stay
# on the volume. Not needed to build the figures or tables: the pulled files are tracked.
#   bash repro_data/fetch.sh            (from rl/)
# Rewrites MANIFEST.txt: sha256 of every file here, plus the checkpoint lines already in it
# ("<sha>  <sweep>/<run>/checkpoint-200/model.safetensors", printed by
# tools/hf_upload_adapters.py). Same relative paths as on the volume, so
# `sha256sum -c --ignore-missing MANIFEST.txt` works here and in the volume's output dir.
# Env: S3 credentials for the volume as AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY,
#      or VOL_HOST (default <ops-host>): a host whose /root/.env holds RUNPOD_S3_* keys.
#      S3_URI (default s3://<volume-id>/graft-repro/output).
set -euo pipefail
cd "$(dirname "$0")"
S3_URI=${S3_URI:-s3://<volume-id>/graft-repro/output}
SWEEPS=$(tail -n +2 runs.csv | cut -d, -f4 | sort -u | tr '\n' ' ')

pull='set -euo pipefail
  E="--region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io --only-show-errors"
  t=$(mktemp -d)
  for s in '"$SWEEPS"'; do
    aws s3 cp $E --recursive --exclude "*" --include "*/run_config.yaml" \
      --include "*/routing_eval.jsonl" --exclude "*/checkpoint-*" '"$S3_URI"'/$s/ $t/$s/ >&2
    aws s3 cp $E --recursive --exclude "*" --include "*.json" '"$S3_URI"'/${s}_fseval/ $t/${s}_fseval/ >&2
  done
  aws s3 cp $E --recursive --exclude "*" --include "*.json" '"$S3_URI"'/expr_only_probe_arms/ \
    $t/expr_only_probe_arms/ >&2
  tar -C $t -czf - . ; rm -rf $t'

fetch=$(mktemp -d); trap 'rm -rf "$fetch"' EXIT
if [ -n "${AWS_ACCESS_KEY_ID:-}" ]; then
  bash -c "$pull" | tar -C "$fetch" -xzf -
else
  ssh "${VOL_HOST:-<ops-host>}" 'set -a; . /root/.env; set +a
    export AWS_ACCESS_KEY_ID=$RUNPOD_S3_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$RUNPOD_S3_SECRET_ACCESS_KEY
    '"$pull" | tar -C "$fetch" -xzf -
fi

# keep only the runs in runs.csv (+ their __r0.0 base-model evals); drop everything else
tail -n +2 runs.csv | while IFS=, read -r _ _ _ sweep run _; do
  mkdir -p "$sweep/$run" "${sweep}_fseval" expr_only_probe_arms
  cp "$fetch/$sweep/$run"/{run_config.yaml,routing_eval.jsonl} "$sweep/$run/"
  cp "$fetch/${sweep}_fseval/$run".json "${sweep}_fseval/"
  for f in "$fetch/${sweep}_fseval/${run}__"*.json; do
    if [ -e "$f" ]; then cp "$f" "${sweep}_fseval/"; fi
  done
  for f in "$fetch/expr_only_probe_arms/$run".json "$fetch/expr_only_probe_arms/${run}__"*.json; do
    if [ -e "$f" ]; then cp "$f" expr_only_probe_arms/; fi
  done
done
for d in */; do echo "${d%/}: $(find "$d" -type f | wc -l) files"; done
ck=$(grep "/checkpoint-200/" MANIFEST.txt 2>/dev/null || true)
{ find . -type f ! -name MANIFEST.txt ! -name fetch.sh -printf '%P\n' | LC_ALL=C sort | xargs sha256sum
  if [ -n "$ck" ]; then echo "$ck"; fi; } > MANIFEST.txt.new
mv MANIFEST.txt.new MANIFEST.txt
