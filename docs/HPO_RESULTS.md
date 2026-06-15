# Hyperparameter Tuning — Full Sweep History

This is the chronological record of every BOHB (Bayesian Optimization + HyperBand, via
Optuna) sweep run for this project. For the final winning config, see the "Hyperparameter
Tuning" section of the main [README](../README.md) — this doc is the detailed
reference/lab-notebook version.

All sweeps were run on CIRCE's `muma_2021` partition (`--qos=muma21`), 8 parallel trials at a
time via `bash submit_hpo.sh`.

---

## Objective: minimize validation loss

The HPO objective is **validation loss on a 10% held-out split** of the training corpus —
not attack success rate. This avoids biasing the hyperparameter search toward any particular
extraction prompt: a config that lowers val loss generalizes better to all 15 attack types,
whereas optimizing for `zs_d_greedy` specifically would prejudice the attack comparison.

Attack success rate is still recorded per trial as an informational user attribute and can be
inspected with `python view_hpo.py --study <name>`, but it does not drive the optimizer.

## Search space

| Hyperparameter | Range | Type |
|---|---|---|
| `learning_rate` | [5e-6, 5e-4] | log-uniform |
| `batch_size` | {2, 4, 8, 16, 32} | categorical |
| `max_length` | {128, 256, 512} | categorical |
| `lr_schedule` | {linear, cosine} | categorical |
| `weight_decay` | [0.0, 0.1] | uniform |
| `warmup_fraction` | [0.0, 0.1] | uniform |
| `max_grad_norm` | [0.1, 5.0] | log-uniform |

**`epochs` is not a hyperparameter** — HyperBand controls training budget via per-epoch val
loss pruning.

**Memory constraints** (empirically validated on RTX A6000 (48 GB), full fine-tune
GPT-Neo-125M, one fwd+bwd pass):

| max_length | batch_size=32 peak VRAM |
|---|---|
| 128 | 8.7 GB |
| 256 | 17.5 GB |
| 512 | 36.8 GB |

`batch_size=32` (the largest in the search space) fits at all three lengths with headroom —
no clamping needed on this GPU. (The old 8 GB GTX 1070 Ti limits — which clamped
`max_length=512` down to `batch_size=4` — were removed; this likely caused the early
HyperBand pruning of both `max_length=512` trials in the first `gpt-neo-hpo-v1` batch.)

---

## GPT-Neo 125M HPO v1 (24 trials, converged — superseded)

Study `gpt-neo-hpo-v1`. Objective: minimize held-out val loss.

Best trial **#13**, val_loss=1.5392 (epoch 4):

| Hyperparameter | Value |
|---|---|
| `learning_rate` | 3.42e-05 |
| `batch_size` | 32 |
| `max_length` | 256 |
| `lr_schedule` | cosine |
| `weight_decay` | 0.0618 |
| `warmup_fraction` | 0.0887 |
| `max_grad_norm` | 0.30 |

**`max_length=512` looked unstable for GPT-Neo-125M.** All 3 trials that sampled `max_length=512` (#1, #2, #11) were pruned by epoch 2
with diverging val loss (2.07, 4.54, and 15.00 — the last is worse than a uniform-random
baseline over the vocab, indicating near-collapse). All 3 also happened to sample relatively
high learning rates (1.27e-4 to 1.6e-4); whether 512 is viable for GPT-Neo at the lower LRs
(~3e-5) that work well at 256 remained untested at the time.

Run `python view_hpo.py --study gpt-neo-hpo-v1` for the full trial table.

> **⚠️ Known bug affecting all results above (fixed in `gpt-neo-hpo-v2`)**: `EmailDataset` did
> not mask padding positions in `labels`, so the loss included "predict eos" for every padded
> token. This inflates and destabilizes val loss in proportion to how much of a sequence is
> padding — worst at `max_length=512` (most padding) and especially bad for GPT-Neo's
> 256-token local attention window. **All val-loss numbers in this section
> were computed under this buggy objective and are not comparable to `gpt-neo-hpo-v2` onward.**
> The fix (`labels[attention_mask == 0] = -100`) is in `main.py`'s `EmailDataset.__getitem__`.
> Superseded by `gpt-neo-hpo-v2` below.

---

## GPT-Neo 125M HPO v2 (corrected objective, converged — 24 trials)

Study `gpt-neo-hpo-v2`. Same search space as v1, but with the padding-mask fix applied to
`EmailDataset.__getitem__` (`labels[attention_mask == 0] = -100`). Fresh study with no shared
trial history. Converged after 24 trials (3 batches of 8): best val_loss improved 2.2507 →
2.2413 (0.42%) in the final batch, second consecutive batch under the 1% threshold.

**Best config (trial #20, val_loss = 2.2413, epoch 5) — this is the config used for the final
attack run:**

| Hyperparameter | Value |
|---|---|
| learning_rate | 1.95e-05 |
| batch_size | 32 |
| max_length | **512** |
| lr_schedule | cosine |
| weight_decay | 0.0799 |
| warmup_fraction | 0.0780 |
| max_grad_norm | 0.49 |

**Headline finding — v1's max_length=512 divergence was an artifact.** Under the corrected
objective, the top 5 trials are ALL `max_length=512` (2.2413–2.2581), `max_length=128` trails
at ≥2.356, and in batch 3 TPE sampled 512 for all 8 trials. The "divergence" (val loss 7–15)
seen in v1 was entirely the padding-label bug inflating loss in proportion to padding. The
viable lr band is ~1e-5–1e-4 (sweet spot ~2e-5); everything above ~1.8e-4 had rising val loss
and was pruned. Caveat: top-trial val losses sit within ~0.7% of each other, so the exact
winner among the leaders is somewhat seed-dependent; the robust conclusions are 512 ≫ 128 and
the lr band. Attack rates (informational): 0.07–0.17% across completed trials.

Full table: `python view_hpo.py --study gpt-neo-hpo-v2`.

---

## GPT-Neo 125M long-context probe (gpt-neo-len-probe, complete — 8 trials)

Follow-up to v2: under the corrected objective `max_length=512` leads, and ~22% of emails are
still truncated at 512 tokens (~10% at 1024; median 198, mean 621 tokens, GPT-Neo tokenizer).
This probe searched `max_length ∈ {768, 1024}` with a search space tightened from v2 evidence:
lr capped at 1e-4, batch_size 2/4 dropped, weight_decay/warmup ceilings raised to 0.3/0.2, and
conservative VRAM batch caps for 768 (bs≤16) and 1024 (bs≤8). Token-weighted
`compute_val_loss` (env-configurable via `HPO_MAX_LENGTH_CHOICES`, `HPO_BATCH_SIZE_CHOICES`,
`HPO_LR_MIN`/`HPO_LR_MAX`, `HPO_WD_MAX`, `HPO_WARMUP_MAX`) was applied. Synced to circe only
after `gpt-neo-hpo-v2` fully converged, to keep v2's objective consistent across its batches.

Results (8 trials, all completed or pruned cleanly):

| Trial | max_length | lr | batch_size | schedule | val_loss | state |
|-------|-----------|-----|-----------|----------|----------|-------|
| 1 | 768 | 3.0e-05 | 16 | linear | **2.2203** | complete (best) |
| 0 | 768 | 1.1e-05 | 16 | linear | 2.2222 | complete |
| 4 | 768 | 3.63e-05 | 4 | cosine | 2.2261 | complete |
| 2 | 1024 | 1.1e-05 | 16 | linear | 2.2278 | complete |
| 6 | 768 | 2.26e-05 | 8 | cosine | 2.2322 | pruned |
| 5 | 1024 | 6.36e-06 | 8 | linear | 2.2373 | pruned |
| 3 | 1024 | 3.0e-05 | 16 | linear | 2.2476 | pruned |
| 7 | 1024 | 6.43e-05 | 4 | linear | 2.4141 | pruned |

Best (trial #1, `max_length=768`): val_loss=2.2203 vs v2's 512 baseline of 2.2413 — a 0.94%
improvement. All three completed `max_length=768` trials (2.2203–2.2261) beat the completed
`max_length=1024` trial (2.2278) and the 512 baseline; 1024 shows no further benefit and its
other trials were pruned worse. However, 0.94% is comparable to the ~0.7% spread among v2's
top-3 trials at 512 — i.e. within noise — and the comparison is confounded by different
evaluation token sets per `max_length` and by v2's val_loss being batch-averaged vs this
probe's token-weighted. **Verdict: no further HPO on `max_length` for GPT-Neo** — 512 and 768
are statistically indistinguishable from this probe; defer the final 512-vs-768 call to the
downstream attack-success metric rather than spending more compute on val-loss differences
this small. The final attack run used `max_length=512` (v2's winner).
