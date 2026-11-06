# Hackathon Runbook

The judging guide states work runs from 10:30 AM to 3:00 PM. The following is a proposed allocation, not an additional official rule.

| Time | Focus | Required output |
|---|---|---|
| 10:30–11:00 | Read the assigned problem, scope users, inspect and clean data | Problem brief, document inventory, 5 real questions |
| 11:00–12:00 | Build baseline end-to-end | Ingestion → chunks → vectors → retrieval → grounded answer |
| 12:00–12:30 | Create and review benchmark | At least 10 questions with expected answer facts and source/page |
| 12:30–13:30 | Improve one measured weakness at a time | Before/after scores and error notes |
| 13:30–14:15 | Finish UI, README, configuration and repository | Reproducible demo |
| 14:15–15:00 | Rehearse and submit | Tested demo, slides, optional video |

## Freeze checklist
- [ ] Do not add untested features near submission.
- [ ] Test one supported question with source/page.
- [ ] Test a paraphrased question.
- [ ] Test an out-of-scope question and verify safe refusal.
- [ ] Verify no API key or private data is committed.
- [ ] Verify README setup commands from a clean environment.
- [ ] Confirm repository and presentation are submitted before deadline.
