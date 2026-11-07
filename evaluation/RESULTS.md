# Dalil — Live Interactive Evaluation Results

> **Evidence status:** Measured from 29 recorded Gradio interactions on 7 November 2026. This is an observational smoke-test report, not a validated answer-accuracy benchmark.

## 1. Test setup

- **System:** Dalil, a retrieval-augmented academic advising assistant using Palestine Polytechnic University (PPU) documents.
- **Test mode:** Manual questions submitted through the Gradio application (`python app.py`).
- **Evidence source:** `evaluation/results/live_tests.jsonl` (user-provided copy of 29 records).
- **Timing:** `time.perf_counter()` around `process()`; wall-clock request handling, not model-only latency.
- **Technical status:** `success` indicates `process()` returned a conversation turn, **not** that the answer is correct or grounded.
- **Answer quality:** `correctness` was `null` in all 29 records; no reference-answer grading was performed.

## 2. Observed metrics

| Metric | Observed value | Interpretation |
|---|---:|---|
| Recorded questions | **29** | Single manual test session |
| Mean response time | **7.480 s** | Arithmetic mean of 29 recorded durations |
| Median response time | **1.967 s** | Less sensitive to slow outliers |
| Minimum / maximum | **0.830 / 37.900 s** | Longest response was test #14 |
| Recorded technical successes | **29/29** | All records marked `success`; this is not answer accuracy |
| Responses with displayed source section | **13/29** | Citation presence only; no citation correctness verification |
| Explicit out-of-scope refusals | **4/29** | Includes potentially erroneous academic refusals |
| Insufficient-evidence responses | **12/29** | May be appropriate or false abstentions |
| Independently verified answer accuracy | **Not measured** | No graded reference answers |

## 3. Question-by-question evidence

The following is an abbreviated audit trail. The original JSONL is the authoritative record for complete answers and source strings.

| # | User question (abridged) | Time (s) | Observed behavior | Reviewer note |
|---:|---|---:|---|---|
| 1 | كم معدل الرسوب بالمادة | 8.573 | Answer with citations | Verify 45% interpretation |
| 2 | ما الحد الأدنى لنسبة الحضور في المادة؟ | 0.830 | Insufficient evidence | Possible false abstention |
| 3 | شو بصير إذا غبت أكثر من المسموح؟ | 21.708 | Answer with citations | Mixed diploma and bachelor references; verify applicability |
| 4 | كم ساعة معتمدة لمادة البرمجة المتقدمة؟ | 1.541 | Insufficient evidence | Course credit data not found |
| 5 | معدلي 2.1 وأنا سنة ثالثة، كم ساعة أقدر أسجل؟ | 1.405 | Insufficient evidence | GPA scale unspecified |
| 6 | وإذا كان معدلي أعلى من ذلك؟ | 1.263 | Insufficient evidence | Follow-up not resolved |
| 7 | متى موعد امتحان الخوارزميات؟ | 1.210 | Insufficient evidence | Potentially appropriate without current schedule |
| 8 | اكتب لي قصيدة عن الجامعة. | 1.110 | Scope refusal | Appropriate nonacademic refusal |
| 9 | ما الحد الأدنى لعلامة النجاح في المساق؟ | 0.979 | Answer with citations | Verify cited 60% threshold |
| 10 | ما الحد الأدنى للمعدل التراكمي المطلوب للتخرج؟ ما المقصود بالإنذار الأكاديمي؟ | 1.403 | Answer with citations | Only one of two questions answered |
| 11 | ما شروط الحصول على درجة البكالوريوس؟ | 1.000 | Insufficient evidence | Possible false abstention |
| 12 | ما الفرق بين الانسحاب من المساق والرسوب فيه؟ | 1.967 | Answer with citations | Verify standard vs exceptional load |
| 13 | هل يستطيع الطالب إعادة مساق نجح فيه لتحسين علامته؟ | 1.491 | Answer with citations | Academic question incorrectly classified out of scope |
| 14 | ما الحد الأقصى لعدد الساعات المعتمدة في الفصل العادي؟ | 37.900 | Answer with citations | Possible false abstention |
| 15 | ما شروط الانتقال من تخصص إلى تخصص آخر؟ | 1.027 | Scope refusal | Verify 18/21 credit exceptions |
| 16 | إذا رسبت بمادة، شو لازم أعمل؟ | 2.520 | Insufficient evidence | Verify high-stakes dismissal statement |
| 17 | قديش بقدر أنزل ساعات بالفصل؟ | 12.826 | Answer with citations | Two academic questions incorrectly refused |
| 18 | إذا معدلي نزل كثير، بفصلوني من الجامعة؟ | 12.686 | Answer with citations | Verify repeat policy and threshold |
| 19 | بقدر أسحب مادة بنص الفصل؟ إذا غبت عن الفاينل، شو بصير معي؟ | 7.253 | Scope refusal | Potentially missing degree-plan retrieval |
| 20 | بقدر أعيد مادة عشان أرفع معدلي؟ | 11.540 | Answer with citations | Potentially missing prerequisites |
| 21 | كم عدد الساعات المعتمدة المطلوبة للتخرج من تخصص هندسة الحاسوب | 12.139 | Answer with citations | Verify warning-removal rule |
| 22 | ما المساقات الإجبارية في خطة تخصص علم الحاسوب؟ | 14.526 | Insufficient evidence | Appropriate nonacademic refusal |
| 23 | ما المساقات التي تتطلب اجتياز البرمجة 1؟ | 14.762 | Insufficient evidence | English question not answered |
| 24 | إذا كان معدلي التراكمي 1.8، هل أستطيع تسجيل 18 ساعة؟ ولماذا؟ | 32.140 | Insufficient evidence | English input answered in Arabic; verify rule |
| 25 | إذا كنت على إنذار أكاديمي وتحسن معدلي، متى يمكن إزالة الإنذار؟ | 1.311 | Answer with citations | Attendance policy not answered |
| 26 | شو أفضل مطعم في الخليل؟ | 4.493 | Scope refusal | Do not substitute Computer Science for Computer Engineering |
| 27 | What is the minimum attendance percentage required for a course? | 0.947 | Insufficient evidence | Potentially missing plan retrieval |
| 28 | Can a student repeat a course to improve their grade? | 1.718 | Answer with citations | Potentially missing prerequisites |
| 29 | If I exceed the permitted absence limit but have excellent grades, can I still take the f... | 4.663 | Insufficient evidence | GPA-scale ambiguity; cautious answer |

## 4. Qualitative findings

**Strengths observed**

- Nonacademic prompts such as a poem request and restaurant recommendation were declined.
- Several academic responses included identifiable document names and article/page pointers.
- Some underspecified questions received an explicit insufficient-evidence response rather than an unsupported numeric claim.

**Issues requiring attention**

- **False scope refusals:** Tests #13 and #17 asked academic policy questions but received the generic out-of-scope refusal.
- **Partial answer:** Test #10 combined graduation GPA and academic warning; only the GPA portion was answered.
- **Potential retrieval coverage gaps:** Course plans, prerequisites, attendance rules, and graduation requirements triggered abstentions in several tests; these are not automatically correct abstentions.
- **Degree-level contamination risk:** Test #3 cited diploma regulations for a broadly phrased attendance question; confirm whether the user intended bachelor or diploma rules.
- **Program substitution risk:** The Computer Engineering credit-hours question was answered partly with Computer Science information; the requested program was not established.
- **Latency variability:** Several responses exceeded 10 seconds and one reached 37.900 seconds; report distributions and repeat measurements before making performance claims.
- **Multilingual behavior:** English questions were answered in Arabic or abstained; assess language expectations explicitly.

## 5. Limitations and metric definitions

1. **No answer-accuracy percentage can be claimed.** Every `correctness` value is `null`; a 29/29 technical success rate is not a 100% factual success rate.
2. **No retrieval Precision@k, Recall@k, Hit@k, MRR, or citation faithfulness score** can be computed from this JSONL alone, because it does not include ranked retrieved passages, relevance labels, or verified reference answers.
3. **Displayed citations are not verified citations.** Each cited source must be checked against the actual PPU document, version, degree level, article/page and answer statement.
4. **Single-session observational data:** No concurrency/load testing, repeatability statistics, hardware specifications, or cold/warm cache controls were captured.
5. **Potential timing skew:** Response times include application processing and any retrieval/generation work performed by `process()`; they are not pure LLM generation times.
6. **Document currency:** The cited student guide and degree plans may be older versions; academic policies must be verified with the applicable current official university documents.

## 6. Recommended next validation

1. Add `expected_answer`, `expected_source`, `question_category`, `reviewer_label` (`correct` / `partial` / `incorrect` / `appropriate_abstention` / `false_abstention`) and `reviewer_notes` to a separately reviewed evaluation sheet.
2. Manually check at least 10 reference-answer questions, including known academic questions, unavailable-information cases, out-of-scope prompts, multilingual questions and follow-ups.
3. Record retrieved chunk IDs and rankings for retrieval evaluation, and validate whether cited evidence entails each answer.
4. Retest identified failures after changes, using the same fixed question set, then publish before/after results without replacing the original evidence.
5. Commit the JSONL evidence only after reviewing it for sensitive content and confirming repository sharing is appropriate.

## 7. References

- [Gradio Blocks documentation](https://www.gradio.app/docs/gradio/blocks) — application interaction framework.
- [Python `time.perf_counter()` documentation](https://docs.python.org/3/library/time.html#time.perf_counter) — timing method.
- [Ragas evaluation metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/) — grounding and retrieval assessment methodology.
- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html) — metric terminology.

---
**Provenance:** Prepared from the 29 test records provided by the project team; qualitative flags are review hypotheses, not adjudicated accuracy labels.
