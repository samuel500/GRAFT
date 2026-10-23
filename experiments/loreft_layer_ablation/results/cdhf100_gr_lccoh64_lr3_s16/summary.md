# Layer-window ablations: cdhf100_gr_lccoh64_lr3_s16

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
| **retain_all** | 0.824 | 0.816 | 0.008 | 0.805 | 0.852 |
| retain_keep_1-5 | 0.613 (-0.024) | 0.570 (+0.008) | 0.043 (-0.031) | 0.504 (-0.023) | 0.961 (+0.000) |
| retain_drop_1-5 | 0.812 (-0.012) | 0.797 (-0.019) | 0.016 (+0.008) | 0.797 (-0.008) | 0.875 (+0.023) |
| retain_keep_6-10 | 0.699 (+0.062) | 0.625 (+0.063) | 0.074 (+0.000) | 0.605 (+0.078) | 0.949 (-0.012) |
| retain_drop_6-10 | 0.820 (-0.004) | 0.816 (+0.000) | 0.004 (-0.004) | 0.785 (-0.020) | 0.832 (-0.020) |
| retain_keep_11-15 | 0.711 (+0.074) | 0.668 (+0.106) | 0.043 (-0.031) | 0.652 (+0.125) | 0.930 (-0.031) |
| retain_drop_11-15 | 0.793 (-0.031) | 0.770 (-0.046) | 0.023 (+0.015) | 0.754 (-0.051) | 0.844 (-0.008) |
| retain_keep_16-20 | 0.711 (+0.074) | 0.641 (+0.079) | 0.070 (-0.004) | 0.652 (+0.125) | 0.930 (-0.031) |
| retain_drop_16-20 | 0.797 (-0.027) | 0.777 (-0.039) | 0.020 (+0.012) | 0.820 (+0.016) | 0.895 (+0.043) |
| retain_keep_21-25 | 0.676 (+0.039) | 0.621 (+0.059) | 0.055 (-0.019) | 0.602 (+0.074) | 0.918 (-0.043) |
| retain_drop_21-25 | 0.785 (-0.039) | 0.781 (-0.035) | 0.004 (-0.004) | 0.801 (-0.004) | 0.887 (+0.035) |
| retain_keep_26-30 | 0.664 (+0.027) | 0.605 (+0.043) | 0.059 (-0.015) | 0.504 (-0.023) | 0.949 (-0.012) |
| retain_drop_26-30 | 0.812 (-0.012) | 0.809 (-0.007) | 0.004 (-0.004) | 0.805 (+0.000) | 0.859 (+0.008) |
| retain_keep_31-35 | 0.691 (+0.054) | 0.605 (+0.043) | 0.086 (+0.012) | 0.531 (+0.004) | 0.938 (-0.023) |
| retain_drop_31-35 | 0.789 (-0.035) | 0.785 (-0.031) | 0.004 (-0.004) | 0.812 (+0.008) | 0.863 (+0.012) |

## Part B: forget adapter windows at scale 1, retain on everywhere

keep deltas vs retain_all (forget off), drop deltas vs forget_all.

| config | comb | retain | hack | probe pass@1 | probe parse |
|---|---|---|---|---|---|
| **retain_all** | 0.824 | 0.816 | 0.008 | 0.805 | 0.852 |
| **forget_all** | 1.000 | 0.008 | 0.992 | 0.078 | 0.855 |
| forget_keep_1-5 | 0.832 (+0.008) | 0.789 (-0.027) | 0.043 (+0.035) | 0.770 (-0.035) | 0.852 (+0.000) |
| forget_drop_1-5 | 1.000 (+0.000) | 0.207 (+0.199) | 0.793 (-0.199) | 0.418 (+0.340) | 0.809 (-0.047) |
| forget_keep_6-10 | 0.820 (-0.004) | 0.816 (+0.000) | 0.004 (-0.004) | 0.793 (-0.012) | 0.824 (-0.027) |
| forget_drop_6-10 | 1.000 (+0.000) | 0.008 (+0.000) | 0.992 (+0.000) | 0.184 (+0.105) | 0.844 (-0.012) |
| forget_keep_11-15 | 0.840 (+0.016) | 0.820 (+0.004) | 0.020 (+0.012) | 0.816 (+0.012) | 0.852 (+0.000) |
| forget_drop_11-15 | 1.000 (+0.000) | 0.008 (+0.000) | 0.992 (+0.000) | 0.145 (+0.066) | 0.875 (+0.020) |
| forget_keep_16-20 | 0.887 (+0.063) | 0.801 (-0.015) | 0.086 (+0.078) | 0.793 (-0.012) | 0.891 (+0.039) |
| forget_drop_16-20 | 0.996 (-0.004) | 0.059 (+0.051) | 0.938 (-0.054) | 0.230 (+0.152) | 0.879 (+0.023) |
| forget_keep_21-25 | 0.824 (+0.000) | 0.793 (-0.023) | 0.031 (+0.023) | 0.785 (-0.020) | 0.852 (+0.000) |
| forget_drop_21-25 | 1.000 (+0.000) | 0.031 (+0.023) | 0.969 (-0.023) | 0.113 (+0.035) | 0.797 (-0.059) |
| forget_keep_26-30 | 0.828 (+0.004) | 0.820 (+0.004) | 0.008 (+0.000) | 0.824 (+0.020) | 0.867 (+0.016) |
| forget_drop_26-30 | 1.000 (+0.000) | 0.012 (+0.004) | 0.988 (-0.004) | 0.117 (+0.039) | 0.863 (+0.008) |
| forget_keep_31-35 | 0.816 (-0.008) | 0.789 (-0.027) | 0.027 (+0.019) | 0.789 (-0.016) | 0.875 (+0.023) |
| forget_drop_31-35 | 0.996 (-0.004) | 0.094 (+0.086) | 0.902 (-0.090) | 0.352 (+0.273) | 0.930 (+0.074) |
