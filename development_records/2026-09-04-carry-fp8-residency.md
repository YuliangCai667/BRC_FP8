# CARRY-FP8 online resident critic

Date: 2026-09-04
Branch: `codex/blackwell-fp8-direct`
Base HEAD: `5500f6a50fe92f17cf9c5e78e7f4076e00e0b958`
State: main and carry GPU-JIT materialization barriers implemented and validated;
corrected carry reaches 450k return `832.74/807.60` on seed42/seed1; seed42 was
intentionally replaced by an online-carry + target-lag experiment after a
successful compatibility smoke; commit and push are not part of this launch

## Motivation

The repaired scale-aware resident critic has nonzero, matched gradients, but its
immediate E4M3 write discards the optimizer candidate's off-lattice position.
The current-amax/FP32-master control recovers much of the 100k return and keeps a
healthy norm trajectory, while a fixed-radius projection is too restrictive.
CARRY-FP8 preserves sub-grid state without adding a full-size FP32 master or
residual.

## Representation and data flow

Each covered residual Dense kernel persists:

```text
main code C: float8_e4m3fn
shared per-member scale s: float32
carry code R: float8_e4m3fn
logical optimizer parameter Theta = s * (float32(C) + float32(R) / 16)
physical forward/backward weight W = s * float32(C)
```

Initialization first applies the unchanged current-amax quantizer to the FP32
kernel, then stores `E4M3_RTN(16 * (W_fp32 / s - C))`. Zero tensors retain
`s=1`, `C=0`, and `R=0`. The original FP32 kernel is discarded.

At every optimizer step, AdamW receives the reconstructed logical tree as its
`params` argument, including for decoupled weight decay. Its continuous
candidate is quantized exactly once by the existing current-amax main write;
the normalized remainder is clipped to the finite E4M3 range and quantized once
into the next carry. Online GEMMs never read carry. Target EMA reads the updated
logical online parameter, matching the current-amax/FP32-master control's
continuous-state semantics while leaving target storage and compute unchanged.
The existing `critic_pnorm` remains the norm of the physical forward parameter
tree (`sC` for covered kernels); separate diagnostics report physical and
logical kernel norms.

## Code changes

- `jaxrl/networks.py`: carry constants, pure quantize/reconstruct helpers,
  residual Dense carry initialization/state, configuration propagation, and an
  `optimization_barrier` on every E4M3 code produced by the shared quantizer.
- `jaxrl/agent/update.py`: distinct physical/logical/persisted trees, logical
  AdamW and target EMA inputs, carry writeback, and per-member diagnostics.
- `jaxrl/agent/brc_learner.py`: public switch, incompatible-mode validation,
  logical/physical/carry tensor diagnostics.
- `train.py`: `--fp8_resident_carry`, resolved configuration/provenance fields,
  and train metric enablement.
- `jaxrl/checkpoint.py`: backward-compatible resume keys, byte accounting for
  main codes/carry codes/scales/forbidden FP32 carry state, and a materialization
  semantics key that rejects legacy resident/lag checkpoints under corrected
  code.
- `tests/test_fp8.py`: initialization, current-amax identity, error reduction,
  sub-grid accumulation, AdamW, target EMA, disabled-path, diagnostics,
  checkpoint round-trip/next-update, resume isolation, and a GPU-JIT
  same-graph-versus-returned-code materialization regression.
- `scripts/run_dogs_online_fp8_resident_carry.sh`: matched Dogs smoke/formal
  launcher; W&B can be disabled for local engineering checks and `BRC_RUN_TAG`
  creates new corrected run IDs without overwriting invalid artifacts. The
  optional `BRC_TARGET_CRITIC_PRECISION` selects `fp8_direct` (default) or
  `fp8_lag` without duplicating the launcher.

## Validation evidence

- `JAX_PLATFORMS=cpu python -m unittest tests.test_fp8`: `35/35` passed.
- `JAX_PLATFORMS=cpu python -m unittest discover -s tests`: `63/63` passed.
- Seven carry-focused tests passed on CPU and GPU1, but they did not include a
  GPU-JIT assertion that observes the dequantized value of a just-created FP8
  code in the same compiled graph. That missing coverage is decisive below.
- Python compile, `bash -n`, `git diff --check`: passed.
- The fixed-amax accumulation test applies 32 updates of `4e-4`: naive RTN
  remains on the old small-element code, while CARRY-FP8 crosses a main-code
  boundary and finishes more than `4x` closer to the FP32 logical reference.
- Dogs width-4096 local smoke completed at env 5001/update 5. Four covered
  kernel gradient L2 values were `7.8689`, `15.1037`, `9.0506`, and `16.5058`;
  tensor/update NaN/Inf were `0/0`; peak JAX bytes-in-use was about `5.94 GiB`.
- The production-JIT interpretation of the first four learner candidates was
  initially mistaken for a genuine zero-error boundary case. The later
  checkpoint-next-update probe that reported `39.65x–43.32x` reduction crossed
  an eager/materialization boundary and therefore validated the arithmetic,
  not the production compiled graph. It must not be used as a launch gate.
- Actual recovery restore recovered all state and performed the next two
  updates successfully. Main and carry payloads are each `134,217,728` bytes;
  shared scales total `32` bytes; full-size FP32 carry bytes are zero. The
  analysis critic checkpoint grew by `134,217,916` bytes relative to the
  matched non-carry structure (`128 MiB` payload plus serialization overhead).

## Formal runs

Both runs started at 2026-09-04 02:47 Asia/Shanghai from fresh initialization,
with the matched 500k Dogs protocol and user-authorized W&B telemetry:

| Seed | GPU | tmux | PID | W&B | Run directory |
|---:|---:|---|---:|---|---|
| 42 | 1 | `brc_fp8_carry_s42_gpu1` | `4112431` | `a4h7dp4d` | `runs/DMC_DOGS/brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_s42` |
| 1 | 2 | `brc_fp8_carry_s1_gpu2` | `4112436` | `ddivbdzt` | `runs/DMC_DOGS/brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_s1` |

## Limits and follow-up

The smoke is an engineering check, not a performance result. The formal 25k,
50k, 75k, and 100k points must establish whether reconstruction fidelity also
preserves norm/angular trajectories and closes the return gap to the
current-amax/FP32-master reference. Current-amax versus delayed scaling remains
a separate factor. No carry-gain sweep, stochastic rounding, FP32 residual, or
fixed-anchor combination is part of these runs.

## Post-launch mechanism failure

At 2026-09-04 03:36 Asia/Shanghai, both formal runs had passed 68k without
runtime failure, but their 25k and 50k tensor snapshots invalidated the intended
treatment: every carry tensor was zero, `carry_prequant_absmax` was zero,
main/logical error and norm were identical, and reconstruction ratio was about
one. Direct decoding of both 50k critic checkpoints confirmed four
`(2,4096,4096)` E4M3 carry arrays with zero nonzero elements. Covered gradients
were still nonzero and update NaN/Inf remained zero, so this is a persistent
carry-path failure rather than the earlier backward failure. Seed42 pnorm
tracked repaired resident runaway rather than current-master. The running
partial experiments therefore cannot support a CARRY-FP8 return claim; root
cause repair and a fresh initialization are required before repeating the
formal comparison.

## Root cause diagnosis: GPU JIT removes the observable FP8 round trip

Diagnosed at 2026-09-04 03:49 Asia/Shanghai on JAX/JAXLIB `0.6.0`, Flax
`0.10.4`, NVIDIA driver `580.95.05`, and RTX PRO 6000 Blackwell (SM 12.0).
The failure is upstream of replay, optimizer state, checkpointing, and Actor
metadata merge. A four-element/`4x4` and a `4096x4096` standalone probe both
reproduce it.

The carry helper computes:

```text
C = float8(candidate / s)
main = float32(C) * s
R = float8(16 * (candidate / s - float32(C)))
```

On CPU JIT and on GPU eager execution, `main` is the expected lossy E4M3
round trip and `R` is nonzero. On GPU JIT, the compiled consumers of the
newly-created `C` see the pre-conversion FP32 value instead: `main` becomes
equal to `candidate` to FP32 roundoff and the residual becomes exactly zero.
The FP8 `C` returned from the compiled function is nevertheless genuinely
quantized. A host-side dequantization of that returned code exposes the real
error, while the in-graph `main_error` falsely reports about `1e-8`.

The minimal `4096x4096` reproduction reported:

| GPU-JIT form | main relative error | nonzero carry elements | prequant absmax |
|---|---:|---:|---:|
| plain FP32→E4M3→FP32 | `2.684e-8` (false) | `0 / 16,777,216` | `0` |
| barrier on E4M3 codes before widening | `0.0264938` | `16,777,205 / 16,777,216` | `255.995` |

`jax.lax.stop_gradient` and an E4M3→uint8→E4M3 bitcast detour did not prevent
the transformation; `jax.lax.optimization_barrier` placed directly on the FP8
codes did. A barrier on the FP32 candidate before quantization did not help.
Thus the application-level bug is the missing observation/materialization
boundary after the lossy FP8 cast. The exact responsible XLA GPU pass has not
yet been isolated, so the lower-level compiler attribution remains narrower:
GPU compiled conversion/fusion semantics in this environment, not a proven
specific pass name.

This also invalidates the current same-graph `main_write_relative_l2`,
`logical_reconstruction_relative_l2`, applied/intended, and related
new-code dequantization diagnostics. The persisted E4M3 main codes are real,
but carry is zero. Within a multi-update compiled call, a just-created code may
also be widened through the bypassed path before the call returns; any corrected
implementation must barrier the code at the quantizer boundary and add a GPU-JIT
regression test that compares in-graph dequantization with dequantization after
the returned FP8 code is materialized.

At 03:51 both unmodified formal processes were still alive at env `90000` /
update `170003`, with update NaN/Inf `0/0`. Seed42/seed1 critic pnorm had reached
`3421.95/4855.63`. The 75k tensor point independently repeated zero prequant
residual in all eight member/layer rows, as predicted by the minimal GPU-JIT
reproduction. This is runtime-stable but scientifically invalid CARRY data.

## Corrective implementation and pre-smoke validation

At 2026-09-04 04:22–04:28 Asia/Shanghai, the shared current-amax E4M3 quantizer
was changed to apply `jax.lax.optimization_barrier` directly to every main E4M3
code before returning it. The first dedicated-GPU gate then exposed a second
missing boundary: carry uses its own direct E4M3 cast, so its code also needs a
barrier before immediate reconstruction. Both persistent codes now have an
explicit boundary. The production-shaped regression JIT-compiles and vmaps the
actual carry kernel and compares both in-graph main dequantization and logical
reconstruction with a second dispatch over returned main/carry buffers. It
requires nonzero main error, smaller carry error, nonzero carry, and zero
saturation, and passes on both CPU and GPU1.

A corrected GPU-JIT `4096x4096` helper probe reports main relative error
`0.00122739`, persisted-carry relative error `3.2394553e-5`, nonzero carry
`16,767,853 / 16,777,216`, and carry prequant/code absmax `9.091797/9`.
The earlier `3.598577e-8` value was a second false same-graph result from the
unbarriered carry cast and is superseded. The full CPU repository suite is
`64/64` passing in `280.775 s`; Python compile, shell syntax, and
`git diff --check` also pass. The run configuration
records `resolved_fp8_code_materialization=optimization_barrier_after_e4m3_cast_v1`.
Resume validation treats a missing field as `legacy_compiler_elidable_cast` and
rejects legacy resident/current-master/target-resident/target-lag checkpoints,
while leaving unaffected FP8-direct resumes compatible.

After GPU1 was released, all eight carry-focused GPU tests passed. Checkpoint
state is bitwise identical immediately after restore; after one GPU update the
info, E4M3 parameters, FP8 metadata, and target parameters remain exact, while
FP32 Adam moments are equal within `rtol=2e-7, atol=1e-8` (observed maximum
absolute difference `7.45e-9`). A real width-4096 critic completed two updates
inside one compiled call. Its four carry arrays were nonzero at initialization
and changed in `32.48M–32.89M` elements; all eight member rows had
main/logical relative error about `0.02645–0.02654` / `0.000621–0.000631`,
reduction `42.0x–42.7x`, zero saturation, and finite info.

The invalid pre-barrier formal runs independently failed the mechanism gate
again at 100k: all eight carry tensors remained zero and every recorded
reduction ratio remained one. At 125k their eval returns were `163.17/159.64`
for seed42/seed1; at 124k their critic pnorm values were `4647.94/6863.59`, with
update NaN/Inf still `0/0`. Thus the processes are operationally healthy but
remain scientifically invalid negative controls. They were gracefully stopped
at env/update `136418/262837` and `136339/262679`; both wrote
`run_interrupted` followed by `run_finished`, all PIDs/tmux sessions exited,
and GPU1/GPU2 returned to `3 MiB`.

## Corrected Dogs smoke and recovery gate

Fresh run
`brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_smoke_jitbarrier_v1_s42-20260904-043900`
completed at env `5001` / update `5` in `127.24 s` on GPU1. At env 5001 all
eight E4M3 carry members had zero fraction `0.0001186–0.0001390`; main/logical
relative error was `0.02645–0.02654` / `0.000646–0.000756`, for
`35.0x–40.9x` reduction. Carry saturation, all 733 tensor-record NaNs/Infs,
and update NaNs/Infs were zero. Four resident kernel gradients had L2
`0.1475/0.3902/0.1965/0.3765`, all nonzero.

The complete recovery checkpoint was resumed under the same materialization
semantic and advanced to env `5002` / update `7`. Its eight carry zero fractions
remained `0.0001209–0.0001478`, reconstruction reduction was
`40.8x–48.1x`, saturation and all 733 NaN/Inf counts stayed zero, and new
analysis/recovery checkpoints were marked `COMPLETE`. The recovery manifest
contains optimizer and replay state, `134,217,728` bytes each of main and carry
E4M3 codes, `32` scale bytes, and zero full-size FP32 carry bytes.

## Corrected formal runs

Both corrected runs were launched fresh at 2026-09-04 04:48:50 Asia/Shanghai
after all gates passed:

| Seed | GPU | tmux | PID | W&B | Run directory |
|---:|---:|---|---:|---|---|
| 42 | 1 | `brc_fp8_carry_jitbarrier_s42_gpu1` | `199383` | [`7mdayzva`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/7mdayzva) | `runs/DMC_DOGS/brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_jitbarrier_v1_s42` |
| 1 | 2 | `brc_fp8_carry_jitbarrier_s1_gpu2` | `199386` | [`hxf61n84`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/hxf61n84) | `runs/DMC_DOGS/brc_dmc_dogs_online_fp8_resident_carry_e4m3_g16_jitbarrier_v1_s1` |

Both configs record fresh `resume_from: ''`, 500k, width 4096, carry enabled,
fixed anchor disabled, FP8-direct target, the materialization-v1 semantic, and
the matched `25k/50k/100k` tensor/analysis/recovery cadence. W&B initialized and
is syncing. Both runs passed env 5000 first update and reached env 9000 / update
8003. Their 9k critic loss/gradient norm/parameter norm were
`0.01979/0.02754/326.61` and `0.02648/0.03099/345.78`; update NaN/Inf was `0/0`
for both. tmux sessions, PIDs, and GPU1/GPU2 bindings remained live at handoff.

## Mature carry result and target-lag extension

By 2026-09-04 10:02 Asia/Shanghai, the corrected carry treatment had moved well
beyond an early mechanism result. Seed42 completed its 450k eval at `832.7369`
(tasks `963.91/962.28/891.17/513.59`) and was then intentionally stopped at env
`454409` / update `898819` to free GPU1 for the user-requested target-lag
combination. It wrote `run_interrupted` and `run_finished`; its 450k carry was
still nonzero, reduced reconstruction error by `66.64x–1130.49x`, and had zero
saturation/NaN/Inf. Seed1 reached 450k eval `807.5967` and then completed 500k
with final/best/tail-three return `782.8582/807.5967/793.2795`. Its 500k
analysis and recovery checkpoints, `final_checkpoints_finished`, and
`run_finished` are all present; cumulative update NaN/Inf stayed `0/0`.

The launcher was extended with a validated `BRC_TARGET_CRITIC_PRECISION`
environment setting so the same online-carry protocol can select the existing
`fp8_lag` target. A 5001-step width-4096 Dogs compatibility smoke completed with
925 tensor records and both COMPLETE checkpoint classes: all eight online carry
and eight target lag code tensors were nonzero; online carry reduced main-write
error `40.53x–42.79x`; training values were finite. The lag's first-step
quantization error was `2.30%–2.61%`, corresponding to `4.58–5.20` relative
error against the much smaller intended `tau=0.005` EMA step. This is a declared
25k risk gate, not evidence of a runtime failure.

The smoke also exposed three infinite JS-divergence diagnostics when averaging
two subnormal softmax probabilities underflowed the midpoint to zero. This code
is diagnostic-only. Log operands now use the dtype's finite tiny floor; a new
JIT regression reproduces the underflow boundary and passes. Four targeted CPU
tests covering this boundary, carry-aware target EMA, and lag reconstruction/
recurrence pass; launcher syntax/invalid-value check and `git diff --check` also
pass. No full-suite
rerun was needed for this diagnostic-only arithmetic change before launch.

The fresh formal combination started at 2026-09-04 10:32:00 Asia/Shanghai on
GPU1 in tmux `brc_fp8_carry_target_lag_s42_gpu1`, PID `1363708`, W&B
[`5o2266m9`](https://wandb.ai/cai200661-sun-yat/uncategorized/runs/5o2266m9).
Its config is online resident main/carry E4M3, target lag E4M3, no FP32 weight
master for the covered residual kernels, materialization v1, and fresh seed42.
It passed the first update and the 25k formal gate. Eval return was `58.0742`;
all 925 tensor-stat records had NaN/Inf `0/0`. Online carry was nonzero in all
eight covered leaves, reduced reconstruction error by `199.80x–430.67x`, and
had zero saturation. Target lag was also nonzero in all eight leaves, with lag
quantization relative error `0.00845–0.02728`. Its error relative to the much
smaller intended EMA update remained `1.68–5.43` (applied/intended ratio
`1.44–4.66`, cosine `-0.830–0.857`), so 50k/75k remain the next scientific
gates rather than declaring whole-critic feasibility from startup alone.
