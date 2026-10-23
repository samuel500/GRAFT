# Layer-window ablations: cdhf100_gr_lccoh64_lr3_s15

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
| **retain_all** | 0.816 | 0.797 | 0.020 | 0.758 | 0.797 |
| retain_keep_1-5 | 0.684 (+0.047) | 0.629 (+0.067) | 0.055 (-0.019) | 0.605 (+0.078) | 0.902 (-0.059) |
| retain_drop_1-5 | 0.793 (-0.023) | 0.770 (-0.027) | 0.023 (+0.003) | 0.773 (+0.016) | 0.855 (+0.059) |
| retain_keep_6-10 | 0.699 (+0.062) | 0.613 (+0.051) | 0.086 (+0.012) | 0.574 (+0.047) | 0.965 (+0.004) |
| retain_drop_6-10 | 0.797 (-0.019) | 0.797 (+0.000) | 0.000 (-0.020) | 0.801 (+0.043) | 0.832 (+0.035) |
| retain_keep_11-15 | 0.695 (+0.058) | 0.648 (+0.086) | 0.047 (-0.027) | 0.645 (+0.117) | 0.945 (-0.016) |
| retain_drop_11-15 | 0.785 (-0.031) | 0.773 (-0.024) | 0.012 (-0.008) | 0.746 (-0.012) | 0.805 (+0.008) |
| retain_keep_16-20 | 0.750 (+0.113) | 0.656 (+0.094) | 0.094 (+0.020) | 0.684 (+0.156) | 0.891 (-0.070) |
| retain_drop_16-20 | 0.793 (-0.023) | 0.770 (-0.027) | 0.023 (+0.003) | 0.770 (+0.012) | 0.879 (+0.082) |
| retain_keep_21-25 | 0.715 (+0.078) | 0.633 (+0.071) | 0.082 (+0.008) | 0.586 (+0.059) | 0.949 (-0.012) |
| retain_drop_21-25 | 0.805 (-0.011) | 0.793 (-0.004) | 0.012 (-0.008) | 0.770 (+0.012) | 0.828 (+0.031) |
| retain_keep_26-30 | 0.656 (+0.019) | 0.609 (+0.047) | 0.047 (-0.027) | 0.578 (+0.051) | 0.961 (+0.000) |
| retain_drop_26-30 | 0.816 (+0.000) | 0.801 (+0.004) | 0.016 (-0.004) | 0.766 (+0.008) | 0.809 (+0.012) |
| retain_keep_31-35 | 0.645 (+0.008) | 0.559 (-0.003) | 0.086 (+0.012) | 0.574 (+0.047) | 0.949 (-0.012) |
| retain_drop_31-35 | 0.832 (+0.016) | 0.816 (+0.019) | 0.016 (-0.004) | 0.777 (+0.020) | 0.828 (+0.031) |

## Part B: forget adapter windows at scale 1, retain on everywhere

keep deltas vs retain_all (forget off), drop deltas vs forget_all.

| config | comb | retain | hack | probe pass@1 | probe parse |
|---|---|---|---|---|---|
| **retain_all** | 0.816 | 0.797 | 0.020 | 0.758 | 0.797 |
| **forget_all** | 0.996 | 0.703 | 0.293 | 0.703 | 0.836 |
| forget_keep_1-5 | 0.812 (-0.004) | 0.789 (-0.008) | 0.023 (+0.003) | 0.777 (+0.020) | 0.828 (+0.031) |
| forget_drop_1-5 | 1.000 (+0.004) | 0.723 (+0.020) | 0.277 (-0.016) | 0.750 (+0.047) | 0.875 (+0.039) |
| forget_keep_6-10 | 0.816 (+0.000) | 0.805 (+0.008) | 0.012 (-0.008) | 0.820 (+0.062) | 0.859 (+0.062) |
| forget_drop_6-10 | 1.000 (+0.004) | 0.723 (+0.020) | 0.277 (-0.016) | 0.723 (+0.020) | 0.875 (+0.039) |
| forget_keep_11-15 | 0.816 (+0.000) | 0.785 (-0.012) | 0.031 (+0.011) | 0.809 (+0.051) | 0.828 (+0.031) |
| forget_drop_11-15 | 1.000 (+0.004) | 0.734 (+0.031) | 0.266 (-0.027) | 0.727 (+0.023) | 0.891 (+0.055) |
| forget_keep_16-20 | 0.969 (+0.153) | 0.789 (-0.008) | 0.180 (+0.160) | 0.762 (+0.004) | 0.887 (+0.090) |
| forget_drop_16-20 | 0.949 (-0.047) | 0.766 (+0.063) | 0.184 (-0.109) | 0.766 (+0.062) | 0.883 (+0.047) |
| forget_keep_21-25 | 0.812 (-0.004) | 0.809 (+0.012) | 0.004 (-0.016) | 0.812 (+0.055) | 0.824 (+0.027) |
| forget_drop_21-25 | 0.996 (+0.000) | 0.738 (+0.035) | 0.258 (-0.035) | 0.773 (+0.070) | 0.910 (+0.074) |
| forget_keep_26-30 | 0.789 (-0.027) | 0.781 (-0.016) | 0.008 (-0.012) | 0.797 (+0.039) | 0.816 (+0.020) |
| forget_drop_26-30 | 0.996 (+0.000) | 0.719 (+0.016) | 0.277 (-0.016) | 0.730 (+0.027) | 0.887 (+0.051) |
| forget_keep_31-35 | 0.844 (+0.028) | 0.820 (+0.023) | 0.023 (+0.003) | 0.812 (+0.055) | 0.832 (+0.035) |
| forget_drop_31-35 | 0.996 (+0.000) | 0.695 (-0.008) | 0.301 (+0.008) | 0.715 (+0.012) | 0.891 (+0.055) |
