# Model v3 evidence audit

Pranaam v3 outperforms the released v2 pipeline on the available SEPRI
evaluation partition, but the evidence does not establish universal
superiority or isolate the effect of the new architecture. V3 changes the
architecture, training data, and calibration together. The evaluation rows
were excluded from fitting and calibration, but their aggregate results were
inspected during architecture development.

## Claims and estimands

| Claim | Unit and population | Comparison | Verdict |
|---|---|---|---|
| V3 is better than released v2 on SEPRI | One directly labeled household-head row in SEPRI name-hash buckets 80 through 99 | Paired predictions on 18,133 identical rows | Supported for accuracy, precision, recall, F1, Brier score, and log loss |
| V3 is better than a recalibrated v2 | Same rows; v2 calibrated on the same 13,665-row calibration partition | Paired predictions with name-cluster bootstrap uncertainty | Supported for accuracy, recall, F1, Brier score, and log loss; not supported for precision |
| The byte CNN caused the gain | Same evaluation rows | Architecture-only contrast with all other inputs fixed | Not identified because training data and calibration changed too |
| V3 handles vocabulary misses better | The 2,737 evaluation rows for which every v2 word is unknown | Recalibrated v2 against v3 | Supported on this subgroup; Muslim recall rises from 0% to 63.85% |
| V3 is better across India | People outside the available Bihar and SEPRI data | External evaluation by state, script, and time | Untestable with the current labeled data |

## Paired comparison

The primary comparison gives every SEPRI household-head row equal weight. The
uncertainty calculation resamples normalized-name clusters, preserving the
dependence among repeated names. The full report records 2,000 paired draws and
the exact artifact hashes.

| Metric | Released v2 | Recalibrated v2 | V3 | V3 change from recalibrated v2 | 95% cluster bootstrap interval |
|---|---:|---:|---:|---:|---:|
| Accuracy | 96.51% | 96.27% | 97.46% | +1.19 points | +0.95 to +1.43 |
| Muslim precision | 87.75% | 92.34% | 90.29% | -2.04 points | -3.44 to -0.67 |
| Muslim recall | 74.14% | 66.88% | 82.49% | +15.62 points | +13.55 to +17.75 |
| Muslim F1 | 0.804 | 0.776 | 0.862 | +0.086 | +0.071 to +0.103 |
| Brier score | 0.0357 | 0.0326 | 0.0205 | 0.0122 lower | 0.0106 to 0.0139 lower |
| Log loss | 0.1604 | 0.1418 | 0.0859 | 0.0559 lower | 0.0496 to 0.0628 lower |

Recalibration moves v2 toward a conservative operating point. Its precision
rises and its recall falls. V3 recovers many more Muslim-associated names at a
measured precision cost relative to that recalibrated baseline. Against v2 as
released, v3 improves both precision and recall.

The result persists when each normalized name receives equal total weight.
Accuracy rises from 95.83% to 97.17%, recall from 64.21% to 81.21%, and Brier
score improves from 0.0359 to 0.0225.

## Data integrity and support

The two SEPRI files contain 92,996 household-head rows. Ninety-nine rows lack a
religion value, leaving the 92,897 rows used in the original audit. These rows
contain 76,310 unique normalized names. One hundred sixty-eight names have
conflicting religion labels across 923 rows. The grouped hash split keeps each
name in one partition, so these conflicts do not cause cross-partition leakage.
They do set a limit on name-only prediction.

The v3 evaluation partition contains 18,133 rows and 15,118 unique normalized
names. It includes 3,015 duplicate rows and 35 conflicting names across 130
rows. A row-weighted result describes a random SEPRI household head in this
partition. A unique-name-weighted result describes a random normalized name.
Both estimands favor v3, but they are not interchangeable.

V3's largest gains occur where v2's representation is weakest. Recalibrated v2
has 0% Muslim recall when every word is outside its vocabulary, compared with
63.85% for v3. On 16,136 rows whose normalized names do not overlap the
translated land corpus, recall rises by 17.90 points. V3 does not win every
small subgroup. The four-or-more-word group has only 356 rows and 11 Muslim
rows; its accuracy falls by 0.28 points and cannot support a stable conclusion.

## Audit findings

1. The earlier claim that the evaluation partition was untouched was false.
   The rows were held out from fitting and calibration, but candidate results
   informed architecture development. The README and model card now describe
   this as developmental evidence.
2. The product comparison is not an architecture ablation. Direct SEPRI rows
   enter v3 training, while released v2 was trained on the land corpus. A fair
   architecture claim requires both models to use the same names, weights,
   validation rule, calibrator, and evaluation rows.
3. The precision claim depends on the baseline. V3 beats v2 as released but
   trails a recalibrated, lower-recall v2 by 2.04 percentage points.
4. Duplicate and conflicting names require clustered uncertainty and explicit
   weighting. Row-level bootstrap draws would overstate the independent sample
   size.
5. V1 and v2 were full-name inputs, not full-name sequence models. V1 trained
   on complete recorded name strings. V2 migrated the same weights to newer
   formats. Both averaged whole-word embeddings and discarded word order.

## Downstream packages

| Package | Current model | What transfers from Pranaam v3 | Recommended action |
|---|---|---|---|
| `ethnicolr` | Character-bigram LSTMs for surname and full-name tasks, with temperature scaling, priors, and conformal sets | The byte CNN is a compact challenger that may improve Unicode and out-of-vocabulary handling. The calibrated output contract and responsible-use language already have close analogues. | Benchmark byte CNNs against each existing model on identical name-grouped splits. Keep the LSTM unless the CNN wins the task-specific metrics and speed or size gates. |
| `ethnicolr2` | Older character LSTM, including a Florida full-name model | V3's safe tensors, explicit padding behavior, script support, calibration, abstention, and artifact provenance address concrete weaknesses. The current encoder fills padding with the out-of-bounds token and the LSTM reads the last padded step. It also loads pickle-based joblib data and PyTorch files without `weights_only=True`. | Do not build a third parallel model stack. Consolidate supported use cases into `ethnicolr`, publish a migration guide, and deprecate `ethnicolr2` after parity checks. |
| `instate` | Properly packed character BiLSTMs for surname-to-state and surname-to-language ranking, downloaded from a pinned Hugging Face revision | V3's score contract, calibration metadata, abstention, supported-input status, checksums, and aggregate-use warning transfer directly. A byte CNN is only a challenger architecture. | Add calibrated distributions and abstention before changing the network. Then compare CNN and BiLSTM on the frozen surname split, including state and language mass coverage, model size, and CPU latency. |

Pranaam's trained weights do not transfer to these packages because their
labels and reference populations differ. The reusable unit is a tested model
protocol: Unicode normalization, ordered tokenization, safe artifact loading,
calibration, abstention, provenance, grouped evaluation, and responsible-use
semantics.

## Check matrix

| Check | Result |
|---|---|
| Denominators and units | Passed after separating household-head-row and unique-name estimands |
| Missing values | Passed; 99 missing religion values are excluded and reported |
| Silent row loss | Passed for the evaluation reconstruction; 18,133 expected and observed |
| Provenance | Partial; the aggregate JSON reproduces the v2-v3 numbers, while the historical 92,897-row v0.6 audit still needs its original producing script or artifact |
| Internal consistency | Corrected model lineage and the false untouched-test wording |
| EDA and support | Completed for SEPRI rows, duplicate names, conflicting labels, word count, vocabulary support, and land overlap |
| Joins | The land translation join declares many-to-one cardinality; survey rows do not require a join |
| Construction | Normalized-name hashing is deterministic and names do not cross partitions |
| Inference | Paired name-cluster bootstrap with 2,000 seeded draws |
| Skew and leverage | Regression checks are inapplicable; subgroup cell size and positive counts are reported instead |
| Forking paths | Failed as a confirmatory standard because candidate evaluation results informed development |
| Experimental design | Inapplicable; this is predictive evaluation, not a causal contrast |
| Prediction and machine learning | Grouped splitting and held-out calibration pass; external subgroup validity remains untested |

## Rejected claims

- V3 is universally better. The available data cover one survey population and
  one in-source Hindi land holdout.
- V3 has higher precision under every fair comparison. Recalibrated v2 is more
  precise because it predicts fewer positive cases.
- The CNN caused the observed gain. The comparison changes data and
  calibration with architecture.
- V2 did not accept full names. It accepted and trained on full name strings,
  but its averaging operation discarded order.

## Plan

1. Preserve the immutable Hugging Face revision and paired audit report
   published with Pranaam 0.7.0 and model family 3.0.
2. Freeze an external confirmatory benchmark before collecting labels. A
   stratified sample of electoral-roll names across states can test geography,
   script, word count, OCR quality, and land-corpus overlap. Annotation should
   use an authorized, defensible label source and should never infer a person's
   religion from the name itself.
3. Run an architecture ablation on identical data. Train the v2 averaging model
   and v3 byte CNN with the same grouped folds, weights, epoch-selection rule,
   calibration rows, and thresholds.
4. Report results by state, script, name length, vocabulary support, source,
   and time. Predeclare minimum cell sizes and primary metrics before opening
   the confirmatory labels.
5. Apply the shared protocol to `instate` first because its current architecture
   and Hugging Face setup are sound. Treat the byte CNN as a benchmark, not a
   presumed replacement.
6. Benchmark `ethnicolr` next. Consolidate `ethnicolr2` into it instead of
   maintaining two race and ethnicity model stacks.

The non-identifying aggregate output of that comparison is recorded in
`scripts/adhoc/v2_v3_comparison.json`. The script that produced it was
removed in 0.9.0 along with the v1 and v2 model module it exercised, so the
recorded numbers stand as a historical result rather than a reproducible
one.
