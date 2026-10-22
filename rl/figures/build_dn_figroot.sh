#!/bin/bash
# Assemble output/figroot_dn, the GRAFT_OUTPUT_ROOT for repro_countdown_figure1_dn.py,
# offline from tracked data only:
#   repro_data/          our runs (repro_data/runs.csv): the Fig 2 repro (GRAFT s9/15/16/1,
#                        No intervention s9/15/16 + s1 from dn_paper_s1) laid out under the
#                        paper's directory names, the no-intervention 2x2 arms (dn_lr3,
#                        dn_anchor_lr1, dn_anchor) and the lr-5e-4 GRAFT / GRAFT w/o routing
#                        arms (gr_lr1, noroute_lr1)
#   paper_data/paper_fig_inputs.tar.gz   the paper's own GRAFT w/o routing
#                        (cdhf100_noroute_fseval_rerun + its expr-only probes) and its
#                        base-model fseval + probe (paper_base/)
#   bash figures/build_dn_figroot.sh            (from rl/)
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."
D=repro_data
ROOT=output/figroot_dn

rm -rf "$ROOT"
gr=$ROOT/countdown_hf100_gr_lccoh64_lr3; gr5=${gr}_seeds5
dn=$ROOT/countdown_code_rp2-0702-0026_fseval
mkdir -p "$gr" "${gr}_fseval" "$gr5" "${gr5}_fseval" "$dn"
for s in 9 15 16; do
  cp -r "$D/repro_fig2/cdhf100_gr_lccoh64_lr3_s$s" "$gr/"
  cp "$D"/repro_fig2_fseval/cdhf100_gr_lccoh64_lr3_s$s.json \
    "$D"/repro_fig2_fseval/cdhf100_gr_lccoh64_lr3_s${s}__*.json "${gr}_fseval/"
done
cp -r "$D/repro_fig2/cdhf100_gr_lccoh64_lr3_s1" "$gr5/"
cp "$D"/repro_fig2_fseval/cdhf100_gr_lccoh64_lr3_s1.json "${gr5}_fseval/"
cp "$D"/repro_fig2_fseval/countdown_code_hack_reward_penalty_amountmissing_s*.json \
  "$D"/dn_paper_s1_fseval/*.json "$dn/"
for d in dn_anchor dn_lr3 dn_anchor_lr1 gr_lr1 noroute_lr1; do
  cp -r "$D/${d}_fseval" "$ROOT/"
done
cp -r "$D/expr_only_probe_arms" "$ROOT/"
tar -xzf paper_data/paper_fig_inputs.tar.gz -C "$ROOT" --strip-components=1 --wildcards \
  'output/cdhf100_noroute_fseval_rerun/*.json' 'output/expr_only_probe_arms/cdhf100_noroute_anchor_s*.json'
# the paper's base model (retain 0, forget 0), kept apart from ours (same file names)
mkdir -p "$ROOT/paper_base"
tar -xzf paper_data/paper_fig_inputs.tar.gz -C "$ROOT/paper_base" --strip-components=2 \
  output/countdown_hf100_gr_lccoh64_lr3_fseval/cdhf100_gr_lccoh64_lr3_s9__r0.0.json \
  output/expr_only_probe_arms/cdhf100_gr_lccoh64_lr3_s16__r0.0.json
for d in "$ROOT"/*/; do echo "$(basename "$d"): $(ls "$d" | wc -l)"; done
