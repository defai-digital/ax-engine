# Official TinyStories execution controls

The independent oracle ran the HazyResearch PyTorch implementation at
`c70274273f489abc73832b2b98584a8483b6b203`, with the official TinyStories
checkpoint at `d342ce8aaea58830eef3b8ad473664673e41f692`. The checkpoint
and published evaluation config agree. Strict weights-only loading and the
reference model ran on CPU on a Mac Studio M2 Ultra 192 GB with macOS
27.2.1 and checkpoint files on an external USB SSD. The native comparison
used the pinned AX TinyStories MLX conversion and MLX 0.32.4. [Upstream source](https://github.com/HazyResearch/ncm/tree/c70274273f489abc73832b2b98584a8483b6b203) · [Official file lock](official-lock.json)

All 8,192 vocabulary entries and 7,934 BPE merge ranks match. The upstream
model's construction reproduces its two stored hierarchy buffers exactly.
The oracle includes one greedy launch from the unigram maximum and two
injected-code controls. Its output binds the reference tensors and traces
to their SHA-256 digests. [Official reference receipt](official-reference.json)

| Native profile | Concept IDs and final BPE IDs | Existing numerical bounds against PyTorch CPU |
| --- | --- | --- |
| M5 Max default | Mixed control has one teacher-forced and two greedy concept differences. All final BPE IDs match. | 1 of 64 comparisons passes in each control. |
| M5 Max with `MLX_ENABLE_TF32=0` | Every concept and BPE ID matches. | 51 of 64 comparisons pass in each control. |
| M4 Pro default and `MLX_ENABLE_TF32=0` | Every concept and BPE ID matches. | 51 of 64 comparisons pass in each control. |
| M2 Ultra default | Every concept and BPE ID matches. | 51 of 64 comparisons pass in each control. |

The retained MLX reference controls pass all 64 numerical comparisons in
each of three M5 full-F32 runs. The bounds remain unchanged. The M5 full-F32,
M4 and M2 comparisons with PyTorch exceed those bounds at the 11 codec pool
tensors and two decoder outputs. The upstream CPU codec control shows
rounding error beginning in the first learned upsampling projection of a
level-zero embedding whose peak magnitude exceeds 7.6 billion. The final
L0 contribution differs by approximately 0.00516 between upstream F32 and
F64 for the tested code. [Native comparison receipt](native-comparison.json) · [Upstream L0 precision control](official-l0-precision.json)

The native CLI reproduces all 128 raw positions and the text of the
unigram-greedy oracle on each tested chip. Seeds 17 and 29 also reproduce
all 128 positions and the text between M5 Max and M4 Pro when both
processes use the full-F32 profile. These fixed-input controls leave default
precision, normal admission, HTTP service and qualification open.

### CPU codec precision control

The two official CPU passes consumed the same twelve levels of unigram-greedy
concept codes, starting with code 3479. The F64 pass was rounded to F32 when
exporting the tensors. An offline comparison on M5 Max verified the source
trace and checked 25 tensors. Both passes produced the same 128 BPE IDs as
the retained official greedy run. [Codec precision receipt](cpu-codec-precision.json)

| Tensor | CPU F32 vs. exported F64 max absolute error |
| --- | ---: |
| Level-zero contribution | 0.005161762 |
| Largest individual contribution over levels 1–11 | 0.000004292 |
| Final accumulated latent | 0.005159855 |
| Decoder logits | 0.004097626 |

The first contribution already contains nearly all the maximum difference
seen in the final latent. This control characterizes arithmetic sensitivity
within the official CPU implementation. Native comparison with the same two
captures remains pending.

## Reproduce

Use an unmodified checkout of the pinned upstream source. Materialize the
five files in the official lock and the seven pinned MLX artifact files
before running the exporter. The isolated Python environment used CPython
3.12.13, PyTorch 2.15.0, Hydra Core 1.3.8, Regex 2026.11.19 and Safetensors
0.8.0. The exporter checks both inventories and reads the vocabulary and
merges with a restricted data-only unpickler.

```bash
evidence=benchmarks/ncm-official-oracle-20261118
python -B "$evidence/reproduce.py" \
  "$UPSTREAM" "$OFFICIAL_MODEL_DIR" "$evidence/official-lock.json" \
  "$MLX_MODEL_DIR" crates/ax-engine-mlx/src/ncm/tinystories-lock.json \
  "$ORACLE_DIR" "$evidence/inputs/seed17-mixed.json" "$evidence/inputs/seed29.json"

rustup run 1.99.1 cargo build -p ax-engine-mlx --bin ncm_generator_probe
MLX_ENABLE_TF32=0 target/debug/ncm_generator_probe "$MLX_MODEL_DIR" \
  "$ORACLE_DIR/unigram-greedy.json" "$ORACLE_DIR/unigram-greedy.safetensors" \
  "$RESULT_JSON"
```

Run the same probe for `seed17-mixed` and `seed29`. The probe writes its
complete result and exits with status 1 when the existing numerical bounds
are exceeded. The JSON records exact IDs and numerical checks separately.
