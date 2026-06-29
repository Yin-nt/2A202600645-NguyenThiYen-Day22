# Reflection - Lab 22 (DPO/ORPO Alignment)

**Ten:** Nguyen Thi Yen  
**Cohort:** VinUni AICB  
**Tier da chay:** T4  
**Date:** 2026-06-29

---

## 1. Setup

| Item | Value |
|---|---|
| GPU | Colab/Kaggle T4 16GB |
| CUDA / driver | CUDA runtime from hosted notebook |
| Base model | `unsloth/Qwen2.5-3B-bnb-4bit` |
| SFT dataset slice | `5CD-AI/Vietnamese-alpaca-cleaned`, 1000 samples, 1 epoch |
| Preference dataset slice | `argilla/ultrafeedback-binarized-preferences-cleaned`, 2000 pairs, 1 epoch |
| `COMPUTE_TIER` env | `T4` |
| Total cost | Hosted free GPU runtime |

---

## 2. DPO experiment results

| Metric | SFT-only baseline | SFT + DPO |
|---|---:|---:|
| Training time (NB3) | n/a | completed on T4 |
| VRAM peak | not recorded | not recorded |
| Final loss | 1.4724 from SFT-mini log | 1.37495 |
| Reward gap (chosen - rejected, end of training) | n/a | 0.60445 |
| End chosen reward | n/a | -4.89125 |
| End rejected reward | n/a | -5.49570 |

**Tulu 3 reference numbers** are much stronger than this small lab run because
the deck result uses a different scale, evaluation setup, and likely a more
stable model/data recipe. My run should be treated as a small T4 reproduction of
the pipeline rather than a reproduction of the paper-quality result.

---

## 3. Reward curves analysis

See `submission/screenshots/03-dpo-reward-curves.png`.

The final DPO metrics show a positive reward gap of about `+0.604`, so the DPO
objective did learn to separate chosen and rejected responses in the direction
expected by the loss. However, both final rewards are negative: chosen reward is
about `-4.891` and rejected reward is about `-5.496`. This matters because a
positive gap alone does not prove that the model became generally better. In the
deck, section 3.4 warns about likelihood displacement: the gap can improve
because rejected responses fall faster, even if chosen responses do not become
more likely in an absolute sense. My result looks closer to that cautionary
case than to a clean "chosen reward rises strongly" story. The useful takeaway
is that DPO optimization ran and created preference separation, but the reward
curves and later qualitative outputs need to be interpreted conservatively. I
would not claim the aligned model is actually helpful without better generation
samples and a judge pass.

---

## 4. Qualitative comparison

See `submission/screenshots/04-side-by-side-table.png` and
`data/eval/side_by_side.jsonl`.

The eight fixed prompts were generated for both SFT-only and SFT+DPO. The
manual judge file currently marks all eight examples as ties:

| Category | SFT wins | DPO wins | Ties |
|---|---:|---:|---:|
| Helpfulness | 0 | 0 | 4 |
| Safety | 0 | 0 | 4 |
| Overall | 0 | 0 | 8 |

The qualitative outputs are not strong. Several generations collapse into
repeated fragments such as `spep` or repeated short tokens. Because both SFT and
DPO outputs show similar collapse, I marked the comparison as tie rather than
claiming DPO improved helpfulness or safety. This is an important failure mode
to document: DPO can produce a positive implicit reward gap while generation
quality remains poor if the base/SFT checkpoint, chat template, decoding setup,
or preference formatting is not stable enough. For a real improvement claim, I
would rerun generation after fixing the chat template and then use a judge model
or a careful manual rubric.

**Win/loss/tie summary:** SFT+DPO wins 0/8, ties 8/8, loses 0/8.  
**Judge used:** manual rubric.

---

## 5. Beta trade-off

I did not run the beta sweep. My default run used `beta = 0.1`, which produced a
positive end reward gap of `+0.604`. My hypothesis is that a smaller beta such
as `0.05` would keep the model closer to the SFT policy and might reduce output
collapse, but may also produce a smaller preference gap. A larger beta such as
`0.5` would likely push harder against the reference policy, which could widen
the reward gap but increase the risk of degeneration or shorter, less useful
answers. Based on this run, I would first fix generation/template stability
before using beta sweep results to make conclusions.

---

## 6. Personal reflection - single change that mattered most

The single change that mattered most in this lab was choosing the T4 path with
Qwen2.5-3B instead of trying the larger BigGPU route. The alternative was to use
the 7B model path, which is closer to the spirit of a stronger alignment demo,
but the available runtime and memory made the T4 route more practical. I picked
T4 because it let me complete the core pipeline: SFT-mini, preference data
formatting, DPO training, side-by-side evaluation, and export artifacts. The
result partly confirmed the choice because the artifacts were created and the
DPO metrics show a positive reward gap. It also surprised me because the
qualitative generation quality was poor even after the DPO stage. That taught me
that "the trainer ran" is not the same as "the model is aligned." If I redid the
lab tomorrow, I would spend more time validating the chat template and
generation behavior immediately after SFT before starting DPO. I would also use
a smaller debug set first, inspect real outputs, then scale back up to 2k
preference pairs only after the model produces coherent responses.

---

## 7. Benchmark interpretation

See `submission/screenshots/07-benchmark-comparison.png` and
`data/eval/benchmark_results.json`.

The benchmark JSON exists, but the recorded scores for IFEval, GSM8K, MMLU, and
AlpacaEval-lite are `NaN`. Therefore I should not interpret the benchmark plot
as evidence that DPO improved or harmed these tasks. In a complete run, I would
expect IFEval or AlpacaEval-lite to be the most likely to improve, because those
tasks are closer to instruction-following and preference-style helpfulness. I
would not be surprised if GSM8K dropped slightly, because the deck describes the
alignment tax: optimizing chat preference behavior can trade off with reasoning
format or exact-answer math performance. MMLU should ideally stay roughly flat,
because DPO on preference pairs should not add much factual knowledge. In this
run, the failed/empty benchmark numbers are themselves useful feedback: before
making any benchmark claim, I need a stable generation setup and successful
lm-eval outputs. The current core evidence is limited to DPO training metrics
and qualitative comparison, not benchmark deltas.

---

## Bonus

- [ ] Beta sweep
- [ ] HuggingFace Hub push
- [x] GGUF artifact generated
- [ ] W&B link
- [ ] Cross-judge comparison
- [ ] BONUS-CHALLENGE.md provocation
- [ ] Pair work

---

## Dieu ngac nhien nhat khi lam lab nay

The most surprising point was that reward gap and generation quality can tell
different stories. A positive DPO reward gap is not enough; I still need to
inspect model outputs carefully.
