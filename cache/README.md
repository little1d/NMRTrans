# NMRTrans cache format

## Proton NMR

Each proton NMR signal in `tokenized_input["1HNMR"]` is represented as:

```text
[chemical_shift, peak_width, multiplicity, integration, J_values]
```

- `chemical_shift`: the center of the reported chemical-shift interval, in ppm.
- `peak_width`: the half-range of that interval, in ppm. It is not the full
  interval width and is not the physical full width at half maximum of an
  experimental peak.
- `multiplicity`: the reported splitting pattern, such as `s`, `d`, `t`, `q`,
  `m`, or `dd`.
- `integration`: the reported relative proton count, such as `1H` or `2H`.
- `J_values`: the reported coupling constants in Hz.

For a reported interval from `range_min` to `range_max`, preprocessing uses:

```text
chemical_shift = (range_min + range_max) / 2
peak_width     = (range_max - range_min) / 2
```

The interval can therefore be reconstructed with:

```text
range_min = chemical_shift - peak_width
range_max = chemical_shift + peak_width
```

For example, `7.30–7.36 ppm` becomes:

```text
[7.33, 0.03, multiplicity, integration, J_values]
```

When the source reports a single chemical shift instead of an interval,
`peak_width` is stored as `0.0`. This means that no interval was reported; it
does not mean that the experimental peak has zero physical linewidth.

## Carbon-13 NMR

The carbon-13 NMR data in `tokenized_input["13CNMR"]` is stored as a list of
individual chemical shifts in ppm:

```text
[170.2, 135.5, 135.2, 132.1, ...]
```

Each value represents one reported carbon-13 NMR signal. Unlike the proton NMR
representation, the released carbon-13 NMR cache does not include an interval
width, multiplicity, integration, or coupling constants for each signal.

The cached values remain in their original ppm scale. During model input
preparation, `features.py` converts them to floating-point values, divides them
by `220.0`, and clips the result to the range from `0.0` to `1.0`. This
normalization is applied at runtime and is not stored in the cache files.

## Dataset ranges

- Proton NMR chemical shifts: `0–20 ppm`.
- Carbon-13 NMR chemical shifts: `0–220 ppm`.
- Maximum molecular size: 64 heavy atoms, excluding hydrogen.
