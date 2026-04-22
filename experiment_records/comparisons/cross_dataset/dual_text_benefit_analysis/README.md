# dual_text sample-level benefit analysis

This analysis asks a concrete question:

When comparing the same split and the same test report, which samples are corrected by `dual_text_readout_v2 + gate_reg=0.01` but missed by the original `sentence_pt` baseline?

Here, "benefited" means:

```text
dual_text prediction is correct
sentence_pt prediction is wrong
```

"Harmed" means:

```text
sentence_pt prediction is correct
dual_text prediction is wrong
```

## KIRC result

Aligned splits: `split_0` to `split_9`

| Group | Count |
|---|---:|
| Benefited | 148 |
| Harmed | 135 |
| Both correct | 932 |
| Both wrong | 175 |
| Benefit minus harm | +13 |

Interpretation:

KIRC is the clearest current evidence that the hierarchy graph branch can help. The net gain is not large, but `dual_text` fixes slightly more samples than it breaks.

KIRC benefited samples show:

| Metric | Benefited | Harmed |
|---|---:|---:|
| Graph branch weight mean | 0.1775 | 0.1744 |
| Top-section attention mean | 0.6324 | 0.6773 |
| Document attention entropy mean | 0.6449 | 0.5679 |

The difference in graph weight is small, so the benefit is not simply "higher graph weight always helps." The section attention pattern matters too.

Most frequent top sections in benefited KIRC samples:

| Section | Count |
|---|---:|
| Document Body | 38 |
| Diagnosis | 30 |
| Specimen Submitted | 22 |
| Clinical Information | 19 |
| Gross Description | 18 |
| Comment | 11 |
| Summary of Sections | 8 |
| Intraoperative Consultation | 2 |

## BRCA result

Aligned splits: `split_0` to `split_9`

| Group | Count |
|---|---:|
| Benefited | 213 |
| Harmed | 217 |
| Both correct | 1333 |
| Both wrong | 287 |
| Benefit minus harm | -4 |

Interpretation:

BRCA is essentially neutral at the sample level. The hierarchy branch fixes many reports, but it also breaks about the same number. This matches the aggregate result where gate regularization recovers the old `dual_text` drop but still does not clearly beat `sentence_pt`.

BRCA benefited samples show:

| Metric | Benefited | Harmed |
|---|---:|---:|
| Graph branch weight mean | 0.2528 | 0.1997 |
| Top-section attention mean | 0.7234 | 0.7164 |
| Document attention entropy mean | 0.5513 | 0.5704 |

Unlike KIRC, BRCA benefited samples do have noticeably higher graph-branch weight. However, higher graph usage does not translate into a stable net gain because the harmed group remains similarly large.

Most frequent top sections in benefited BRCA samples:

| Section | Count |
|---|---:|
| Document Body | 81 |
| Final Diagnosis | 65 |
| Procedure | 15 |
| Synoptic Report | 13 |
| Gross Description | 12 |
| Comment | 11 |
| Ancillary Studies | 7 |
| Microscopic Description | 4 |
| Clinical Information | 3 |
| Patient History | 2 |

## Current conclusion

The hierarchy branch is not unused. It changes specific sample predictions.

On KIRC, the changes are slightly positive overall. On BRCA, the changes are balanced between helpful and harmful. This suggests that the hierarchy graph direction is more promising for KIRC under the current implementation, while BRCA likely needs better section-noise handling or section-type-aware weighting before the graph branch can reliably improve results.

## Files

- KIRC full comparison: `KIRC/dual_text_benefit_details.csv`
- KIRC benefited samples: `KIRC/benefited_samples.csv`
- KIRC harmed samples: `KIRC/harmed_samples.csv`
- KIRC summary: `KIRC/dual_text_benefit_summary.json`
- BRCA full comparison: `BRCA/dual_text_benefit_details.csv`
- BRCA benefited samples: `BRCA/benefited_samples.csv`
- BRCA harmed samples: `BRCA/harmed_samples.csv`
- BRCA summary: `BRCA/dual_text_benefit_summary.json`
