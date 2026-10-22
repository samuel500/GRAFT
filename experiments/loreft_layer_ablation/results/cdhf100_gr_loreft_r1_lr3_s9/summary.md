# Layer-window ablations: cdhf100_gr_loreft_r1_lr3_s9

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
| **retain_all** | 0.781 | 0.762 | 0.020 | 0.816 | 0.832 |
| retain_keep_1-5 | 0.660 (+0.023) | 0.605 (+0.043) | 0.055 (-0.019) | 0.512 (-0.016) | 0.930 (-0.031) |
| retain_drop_1-5 | 0.789 (+0.008) | 0.785 (+0.023) | 0.004 (-0.016) | 0.805 (-0.012) | 0.840 (+0.008) |
| retain_keep_6-10 | 0.699 (+0.062) | 0.656 (+0.094) | 0.043 (-0.031) | 0.598 (+0.070) | 0.949 (-0.012) |
| retain_drop_6-10 | 0.812 (+0.031) | 0.801 (+0.039) | 0.012 (-0.008) | 0.812 (-0.004) | 0.844 (+0.012) |
| retain_keep_11-15 | 0.727 (+0.090) | 0.676 (+0.114) | 0.051 (-0.023) | 0.605 (+0.078) | 0.902 (-0.059) |
| retain_drop_11-15 | 0.801 (+0.020) | 0.770 (+0.008) | 0.031 (+0.011) | 0.773 (-0.043) | 0.848 (+0.016) |
| retain_keep_16-20 | 0.738 (+0.101) | 0.703 (+0.141) | 0.035 (-0.039) | 0.621 (+0.094) | 0.891 (-0.070) |
| retain_drop_16-20 | 0.785 (+0.004) | 0.754 (-0.008) | 0.031 (+0.011) | 0.781 (-0.035) | 0.848 (+0.016) |
| retain_keep_21-25 | 0.723 (+0.086) | 0.676 (+0.114) | 0.047 (-0.027) | 0.641 (+0.113) | 0.906 (-0.055) |
| retain_drop_21-25 | 0.781 (+0.000) | 0.750 (-0.012) | 0.031 (+0.011) | 0.781 (-0.035) | 0.852 (+0.020) |
| retain_keep_26-30 | 0.668 (+0.031) | 0.609 (+0.047) | 0.059 (-0.015) | 0.562 (+0.035) | 0.926 (-0.035) |
| retain_drop_26-30 | 0.781 (+0.000) | 0.777 (+0.015) | 0.004 (-0.016) | 0.789 (-0.027) | 0.820 (-0.012) |
| retain_keep_31-35 | 0.684 (+0.047) | 0.613 (+0.051) | 0.070 (-0.004) | 0.566 (+0.039) | 0.980 (+0.020) |
| retain_drop_31-35 | 0.793 (+0.012) | 0.793 (+0.031) | 0.000 (-0.020) | 0.789 (-0.027) | 0.828 (-0.004) |

## Part B: forget adapter windows at scale 1, retain on everywhere

keep deltas vs retain_all (forget off), drop deltas vs forget_all.

| config | comb | retain | hack | probe pass@1 | probe parse |
|---|---|---|---|---|---|
| **retain_all** | 0.781 | 0.762 | 0.020 | 0.816 | 0.832 |
| **forget_all** | 0.992 | 0.348 | 0.645 | 0.316 | 0.805 |
| forget_keep_1-5 | 0.789 (+0.008) | 0.766 (+0.004) | 0.023 (+0.003) | 0.801 (-0.016) | 0.820 (-0.012) |
| forget_drop_1-5 | 1.000 (+0.008) | 0.367 (+0.019) | 0.633 (-0.012) | 0.340 (+0.023) | 0.773 (-0.031) |
| forget_keep_6-10 | 0.848 (+0.067) | 0.812 (+0.050) | 0.035 (+0.015) | 0.785 (-0.031) | 0.820 (-0.012) |
| forget_drop_6-10 | 1.000 (+0.008) | 0.438 (+0.090) | 0.562 (-0.083) | 0.367 (+0.051) | 0.824 (+0.020) |
| forget_keep_11-15 | 0.938 (+0.157) | 0.742 (-0.020) | 0.195 (+0.175) | 0.750 (-0.066) | 0.922 (+0.090) |
| forget_drop_11-15 | 0.996 (+0.004) | 0.543 (+0.195) | 0.453 (-0.192) | 0.520 (+0.203) | 0.867 (+0.062) |
| forget_keep_16-20 | 0.984 (+0.203) | 0.641 (-0.121) | 0.344 (+0.324) | 0.621 (-0.195) | 0.965 (+0.133) |
| forget_drop_16-20 | 0.992 (+0.000) | 0.625 (+0.277) | 0.367 (-0.278) | 0.594 (+0.277) | 0.980 (+0.176) |
| forget_keep_21-25 | 0.848 (+0.067) | 0.773 (+0.011) | 0.074 (+0.054) | 0.793 (-0.023) | 0.852 (+0.020) |
| forget_drop_21-25 | 0.996 (+0.004) | 0.410 (+0.062) | 0.586 (-0.059) | 0.367 (+0.051) | 0.875 (+0.070) |
| forget_keep_26-30 | 0.824 (+0.043) | 0.793 (+0.031) | 0.031 (+0.011) | 0.785 (-0.031) | 0.828 (-0.004) |
| forget_drop_26-30 | 0.996 (+0.004) | 0.367 (+0.019) | 0.629 (-0.016) | 0.340 (+0.023) | 0.824 (+0.020) |
| forget_keep_31-35 | 0.801 (+0.020) | 0.785 (+0.023) | 0.016 (-0.004) | 0.781 (-0.035) | 0.812 (-0.020) |
| forget_drop_31-35 | 0.988 (-0.004) | 0.355 (+0.007) | 0.633 (-0.012) | 0.312 (-0.004) | 0.812 (+0.008) |
