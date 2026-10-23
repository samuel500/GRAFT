# Layer-window ablations: cdhf100_gr_lccoh64_lr3_s9

fseval n=256 (comb / retain reward / hack rate) and expr-only probe pass@1, k=1
n=256. One SE at p=0.5 is about 0.031. keep_W rows show the delta vs the
all-off reference, drop_W rows vs the all-on reference.

## Folder layout

- `summary.md`: this file, written by `analyze.py`.
- `plots/`: charts from `plot.py`. `retain_windows.png` and `forget_windows.png`
  show per-window metrics against the all-on and all-off references;
  `effect_share.png` shows each window's share of the full effect.
- `configs/<config>.json`: one file per config, the source of every number
  here. `config` says which layers each adapter has on and the forget
  scale. `fseval` has the fseval means (`retain/...` = solve rate,
  `hack_freq/...` = hack rate, `combined/...` = reward by either route,
  plus hack-type and detected/undetected splits). `probe` has the
  expr-only probe stats (`true_pass1`, `parse_rate`). `elapsed_s` has timings.
- `per_sample/<config>__fseval_values.json`: for every fseval metric, the
  256 per-problem values (0/1, `null` where a slice doesn't apply).
  Position i is the same problem in every config.
- `per_sample/<config>__probe.jsonl`: one line per probe problem: `idx`,
  `nums`, `target`, the full `completion`, the extracted `expr`, `src`
  (expr json or the solution.py fallback) and `true` (correct or not).

Config names are `<part>_<keep|drop>_<window>`. `keep` means only that
window is on; `drop` means all but that window. The references are
`retain_all`, `retain_none` and `forget_all`. Windows are 0-indexed
decoder layers; layer 0 is never ablated. Full fseval completions were
not saved (the eval code returns only scores), so per-problem hack text
isn't available.

## Part A: retain adapter windows, forget off (scale 0)

keep deltas vs retain_none (the SFT model), drop deltas vs retain_all.

| config | comb | retain | hack | probe pass@1 | probe parse |
|---|---|---|---|---|---|
| **retain_none** | 0.637 | 0.562 | 0.074 | 0.527 | 0.961 |
| **retain_all** | 0.805 | 0.805 | 0.000 | 0.797 | 0.852 |
| retain_keep_1-5 | 0.707 (+0.070) | 0.680 (+0.118) | 0.027 (-0.047) | 0.609 (+0.082) | 0.910 (-0.051) |
| retain_drop_1-5 | 0.797 (-0.008) | 0.770 (-0.035) | 0.027 (+0.027) | 0.734 (-0.062) | 0.793 (-0.059) |
| retain_keep_6-10 | 0.676 (+0.039) | 0.621 (+0.059) | 0.055 (-0.019) | 0.527 (+0.000) | 0.949 (-0.012) |
| retain_drop_6-10 | 0.805 (+0.000) | 0.797 (-0.008) | 0.008 (+0.008) | 0.789 (-0.008) | 0.836 (-0.016) |
| retain_keep_11-15 | 0.707 (+0.070) | 0.660 (+0.098) | 0.047 (-0.027) | 0.602 (+0.074) | 0.922 (-0.039) |
| retain_drop_11-15 | 0.773 (-0.032) | 0.770 (-0.035) | 0.004 (+0.004) | 0.734 (-0.062) | 0.852 (+0.000) |
| retain_keep_16-20 | 0.711 (+0.074) | 0.648 (+0.086) | 0.062 (-0.012) | 0.609 (+0.082) | 0.902 (-0.059) |
| retain_drop_16-20 | 0.758 (-0.047) | 0.746 (-0.059) | 0.012 (+0.012) | 0.734 (-0.062) | 0.832 (-0.020) |
| retain_keep_21-25 | 0.715 (+0.078) | 0.664 (+0.102) | 0.051 (-0.023) | 0.578 (+0.051) | 0.953 (-0.008) |
| retain_drop_21-25 | 0.762 (-0.043) | 0.762 (-0.043) | 0.000 (+0.000) | 0.797 (+0.000) | 0.867 (+0.016) |
| retain_keep_26-30 | 0.660 (+0.023) | 0.602 (+0.040) | 0.059 (-0.015) | 0.578 (+0.051) | 0.965 (+0.004) |
| retain_drop_26-30 | 0.797 (-0.008) | 0.789 (-0.016) | 0.008 (+0.008) | 0.820 (+0.023) | 0.848 (-0.004) |
| retain_keep_31-35 | 0.688 (+0.051) | 0.613 (+0.051) | 0.074 (+0.000) | 0.535 (+0.008) | 0.953 (-0.008) |
| retain_drop_31-35 | 0.762 (-0.043) | 0.758 (-0.047) | 0.004 (+0.004) | 0.777 (-0.020) | 0.832 (-0.020) |

## Part B: forget adapter windows at scale 1, retain on everywhere

keep deltas vs retain_all (forget off), drop deltas vs forget_all.

| config | comb | retain | hack | probe pass@1 | probe parse |
|---|---|---|---|---|---|
| **retain_all** | 0.805 | 0.805 | 0.000 | 0.797 | 0.852 |
| **forget_all** | 0.996 | 0.504 | 0.492 | 0.590 | 0.969 |
| forget_keep_1-5 | 0.848 (+0.043) | 0.820 (+0.015) | 0.027 (+0.027) | 0.785 (-0.012) | 0.816 (-0.035) |
| forget_drop_1-5 | 1.000 (+0.004) | 0.609 (+0.105) | 0.391 (-0.101) | 0.684 (+0.094) | 0.957 (-0.012) |
| forget_keep_6-10 | 0.789 (-0.016) | 0.777 (-0.028) | 0.012 (+0.012) | 0.820 (+0.023) | 0.848 (-0.004) |
| forget_drop_6-10 | 1.000 (+0.004) | 0.543 (+0.039) | 0.457 (-0.035) | 0.617 (+0.027) | 0.977 (+0.008) |
| forget_keep_11-15 | 0.828 (+0.023) | 0.781 (-0.024) | 0.047 (+0.047) | 0.793 (-0.004) | 0.863 (+0.012) |
| forget_drop_11-15 | 1.000 (+0.004) | 0.594 (+0.090) | 0.406 (-0.086) | 0.660 (+0.070) | 0.949 (-0.020) |
| forget_keep_16-20 | 0.980 (+0.175) | 0.746 (-0.059) | 0.234 (+0.234) | 0.766 (-0.031) | 0.934 (+0.082) |
| forget_drop_16-20 | 0.973 (-0.023) | 0.730 (+0.226) | 0.242 (-0.250) | 0.770 (+0.180) | 0.918 (-0.051) |
| forget_keep_21-25 | 0.805 (+0.000) | 0.797 (-0.008) | 0.008 (+0.008) | 0.805 (+0.008) | 0.828 (-0.023) |
| forget_drop_21-25 | 1.000 (+0.004) | 0.559 (+0.055) | 0.441 (-0.051) | 0.645 (+0.055) | 0.980 (+0.012) |
| forget_keep_26-30 | 0.789 (-0.016) | 0.789 (-0.016) | 0.000 (+0.000) | 0.812 (+0.016) | 0.832 (-0.020) |
| forget_drop_26-30 | 1.000 (+0.004) | 0.496 (-0.008) | 0.504 (+0.012) | 0.594 (+0.004) | 0.953 (-0.016) |
| forget_keep_31-35 | 0.789 (-0.016) | 0.777 (-0.028) | 0.012 (+0.012) | 0.824 (+0.027) | 0.848 (-0.004) |
| forget_drop_31-35 | 1.000 (+0.004) | 0.500 (-0.004) | 0.500 (+0.008) | 0.613 (+0.023) | 0.953 (-0.016) |
