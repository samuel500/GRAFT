#!/bin/bash
# Assemble output/figroot_dn, the GRAFT_OUTPUT_ROOT for repro_countdown_figure1_dn.py:
# the Fig 2 repro (GRAFT s9/15/16/1 + No intervention s9/15/16) laid out under the
# paper's directory names (as output/figcompat), plus the no-intervention 2x2 arms
# (dn_lr3, dn_anchor_lr1, dn_anchor), No intervention s1 (dn_paper_s1) and the lr-5e-4
# GRAFT / GRAFT w/o routing arms (gr_lr1, noroute_lr1), whose fsevals + expr-only probes
# are pulled from the network volume via $VOL_HOST's S3 keys; plus the paper's own
# GRAFT w/o routing (cdhf100_noroute_fseval_rerun) from paper_data/paper_fig_inputs.tar.gz.
#   bash figures/build_dn_figroot.sh            (from rl/; re-run as pending arms land)
# Env: REPRO_OUT (dir with repro_fig2{,_fseval}; default: main checkout's rl/output),
#      VOL_HOST (default <ops-host>).
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."
REPRO_OUT=${REPRO_OUT:-$(cd "$(git rev-parse --git-common-dir)/.." && pwd)/rl/output}
VOL_HOST=${VOL_HOST:-<ops-host>}
ROOT=output/figroot_dn
test -d "$REPRO_OUT/repro_fig2" && test -d "$REPRO_OUT/repro_fig2_fseval"

fetch=$(mktemp -d)
ssh "$VOL_HOST" 'set -euo pipefail; set -a; . /root/.env; set +a
  export AWS_ACCESS_KEY_ID=$RUNPOD_S3_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$RUNPOD_S3_SECRET_ACCESS_KEY
  t=$(mktemp -d); B=s3://<volume-id>/graft-repro/output
  for d in dn_anchor_fseval dn_lr3_fseval dn_anchor_lr1_fseval dn_paper_s1_fseval gr_lr1_fseval noroute_lr1_fseval expr_only_probe_arms; do
    aws s3 cp --region EU-RO-1 --endpoint-url https://s3api-eu-ro-1.runpod.io --recursive \
      --only-show-errors --exclude "*" --include "*.json" $B/$d/ $t/$d/ >&2
  done
  tar -C $t -czf - . ; rm -rf $t' | tar -C "$fetch" -xzf -

rm -rf "$ROOT"
gr=$ROOT/countdown_hf100_gr_lccoh64_lr3; gr5=${gr}_seeds5
dn=$ROOT/countdown_code_rp2-0702-0026_fseval
mkdir -p "$gr" "${gr}_fseval" "$gr5" "${gr5}_fseval" "$dn" "$ROOT/expr_only_probe_arms"
for s in 9 15 16; do
  ln -s "$REPRO_OUT/repro_fig2/cdhf100_gr_lccoh64_lr3_s$s" "$gr/"
  for f in "$REPRO_OUT"/repro_fig2_fseval/cdhf100_gr_lccoh64_lr3_s$s.json "$REPRO_OUT"/repro_fig2_fseval/cdhf100_gr_lccoh64_lr3_s${s}__*.json; do ln -s "$f" "${gr}_fseval/"; done
done
ln -s "$REPRO_OUT/repro_fig2/cdhf100_gr_lccoh64_lr3_s1" "$gr5/"
for f in "$REPRO_OUT"/repro_fig2_fseval/cdhf100_gr_lccoh64_lr3_s1.json "$REPRO_OUT"/repro_fig2_fseval/cdhf100_gr_lccoh64_lr3_s1__*.json; do ln -s "$f" "${gr5}_fseval/"; done
ln -s "$REPRO_OUT"/repro_fig2_fseval/countdown_code_hack_reward_penalty_amountmissing_s*.json "$dn/"
cp "$fetch"/dn_paper_s1_fseval/*.json "$dn/" 2>/dev/null || true
for d in dn_anchor_fseval dn_lr3_fseval dn_anchor_lr1_fseval gr_lr1_fseval noroute_lr1_fseval; do
  mkdir -p "$ROOT/$d"; cp "$fetch/$d"/*.json "$ROOT/$d/" 2>/dev/null || true
done
cp "$REPRO_OUT"/expr_only_probe_arms/*.json "$ROOT/expr_only_probe_arms/" 2>/dev/null || true
cp "$fetch"/expr_only_probe_arms/*.json "$ROOT/expr_only_probe_arms/"
tar -xzf paper_data/paper_fig_inputs.tar.gz -C "$fetch" --wildcards \
  'output/cdhf100_noroute_fseval_rerun/*.json' 'output/expr_only_probe_arms/cdhf100_noroute_anchor_s*.json'
mkdir -p "$ROOT/cdhf100_noroute_fseval_rerun"
cp "$fetch"/output/cdhf100_noroute_fseval_rerun/*.json "$ROOT/cdhf100_noroute_fseval_rerun/"
cp "$fetch"/output/expr_only_probe_arms/*.json "$ROOT/expr_only_probe_arms/"
rm -rf "$fetch"
for d in "$ROOT"/*/; do echo "$(basename "$d"): $(ls "$d" | wc -l)"; done
