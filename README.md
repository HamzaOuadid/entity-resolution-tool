# entity-resolution-tool

Record-linkage / entity-resolution pipeline: normalize, block, score, and cluster the
same real-world entity as it appears differently across messy multi-source data (e.g.
`'ACME Industrial Holdings B.V.'`, `'ACME Group'`, `'Acme BV'`) — with an explicit
review queue for the ambiguous middle band instead of forcing every decision
automatically, and published precision/recall against a hand-labelled sample.

Built as project 18 of a 20-project portfolio, against the spec
`18-entity-resolution-dedup-tool-across-messy-multi-source-data.md`.

## Real demo run

```
$ pip install -e ".[dev]"
$ entity-resolution generate-data --out data --n-entities 350 --seed 42
Wrote 971 raw rows across 3 sources to data/
  registry: data\source_registry.csv
  directory: data\source_directory.csv
  crm: data\source_crm.csv
  ground_truth: data\ground_truth.csv

$ entity-resolution run-all --data-dir data --db entity_resolution.db --report-out eval_report.json
=== Entity Resolution Report ===
Entities:                971
Candidate pairs (post-blocking): 1219
True match pairs (ground truth):  957

Blocking recall:          0.878  (117 true matches missed by blocking)

-- Full-dataset match quality (all candidate pairs vs. ground truth) --
Precision: 0.924
Recall:    0.856
F1:        0.889

-- Hand-labelled sample (n=300) --
Precision: 0.930  (tp=213 fp=16)
Recall:    0.973  (fn=6)
F1:        0.951

Decisions: 886 match / 133 no_match / 200 review
Clusters: 300 total, 0 flagged for conflicting evidence

Wrote metrics to eval_report.json
Persisted pipeline state to entity_resolution.db
```

That's an actual run against the committed `data/*.csv` fixtures (seed 42, deterministic
— you'll get identical numbers running the commands above). 971 raw records → 1,219
candidate pairs after blocking, versus 470,935 for naive all-pairs comparison
(**99.74% reduction**). See [Metrics](#metrics-explained) below for what each number
means and why precision isn't 1.000 (it used to be, misleadingly — see
[Design decisions](#design-decisions--tradeoffs)).

## Architecture

```
generate-data                    run-all (or step-by-step below)
      |                                |
      v                                v
 3 source CSVs            +---------------------------+
 (registry/directory/crm) |  normalize()               |  rule-based: legal-suffix
 + ground_truth.csv       |  RawEntity -> Normalized   |  canonicalization, address
      |                   +---------------------------+  abbreviation expansion,
      |                                |                  postal/city recovery from
      |                                v                  free text
      |                   +---------------------------+
      |                   |  block()                   |  3 independent blocking
      |                   |  entities -> CandidatePair  |  strategies (union of their
      |                   +---------------------------+  candidate pairs)
      |                                |
      |                                v
      |                   +---------------------------+
      |                   |  score()                   |  HeuristicScorer (default,
      |                   |  pair -> similarity [0,1]  |  weighted rapidfuzz/Jaro-
      |                   +---------------------------+  Winkler features) or
      |                                |                  LogisticScorer (sklearn)
      |                                v
      |                   +---------------------------+
      |                   |  resolve()                  |  match / no_match / review
      |                   |  pairs -> {matches,         |  bands + union-find
      |                   |  no_matches, review_queue}  |  clustering + conflict flag
      |                   +---------------------------+
      |                                |
      v                                v
 ground truth  ------------->   evaluate()   -> precision/recall/F1,
 (eval only, never fed             |             blocking recall (separate),
  into the pipeline)                v             hand-labelled sample metrics
                              SQLite (raw_entities / normalized_entities /
                              candidate_pairs / resolution_decisions)
                                          |
                                          v
                        review-queue / export-review-queue / submit-review
```

Every stage matches the spec's API contract (`normalize`, `block`, `score`, `resolve`)
and data model (`raw_entities`, `normalized_entities`, `candidate_pairs`,
`resolution_decisions`), persisted to SQLite. `pipeline.py` wires the stages together
in-memory so the exact code path behind the numbers above is also what the test suite
exercises — there's no separate, unverified demo script.

### Why synthetic data instead of a real public dataset

The spec calls for "2+ public datasets with genuine overlap." In practice, finding two
public entity datasets with *known* ground-truth overlap and a license clean enough to
redistribute a derived/merged dataset is its own multi-day research project (see
[Risks](#risks--open-questions)). Per this project's task brief, we generate messy
synthetic data instead: three source exports (a government-style business registry, a
commercial directory with a single free-text address field, and a CRM export) sharing
one underlying set of true entities, each source independently mangled (typos, dropped
legal suffixes, word-reordering, abbreviations, missing fields). Because we generated
it, we know the true duplicate clusters exactly — so the precision/recall numbers above
are *real*, not estimated, and reproducible by anyone who runs the same seed.

### Why SQLite, not Postgres

The spec suggests Postgres; this environment has no Postgres/Docker daemon available,
and the pipeline is a batch job over a modest number of rows with no need for anything
Postgres-specific. `db.py` implements the same four-table schema in SQLite.

### Why not Splink or `dedupe` directly

The spec suggests Splink or `dedupe` for pairwise scoring. Splink pulls in a
DuckDB/Spark backend; `dedupe` needs a C-extension build toolchain — both are heavy for
a demo of this size. `dedupe`'s underlying approach (hand-engineered similarity
features + logistic regression) is straightforward to reproduce with `rapidfuzz` +
`scikit-learn`, both lightweight and already available, so `scoring.py` does that
instead — `HeuristicScorer` (deterministic, the default) and `LogisticScorer` (learned,
evaluated on a held-out split), behind the same `score(pair)` interface.

## Data model

SQLite tables in `db.py`, matching the spec's data model. One naming disambiguation was
needed: the spec uses `source_id` for two different things (`raw_entities.source_id` =
"the ID the source system assigned it"; `normalized_entities.source_id` = "the id of
the raw row it was normalized from" — i.e. a foreign key). Implemented as:

| Table | Columns |
|---|---|
| `raw_entities` | `id, source, external_id, raw_name, raw_address` |
| `normalized_entities` | `id, raw_entity_id (FK), normalized_name, normalized_address, normalized_city, normalized_postal_code, legal_suffix, name_core_tokens, source` |
| `candidate_pairs` | `id, entity_a_id, entity_b_id, blocking_key, similarity_score` |
| `resolution_decisions` | `id, pair_id, decision, reviewer, hand_label` |

## Blocking strategy

Three independent, cheap blocking keys, whose candidate pairs are **unioned** — a true
match only needs to survive in *one* strategy to become a candidate pair, which is what
gives blocking good recall while still cutting comparison volume by orders of
magnitude:

1. **name-prefix + postal-prefix**: first 3 letters of the name's core tokens (legal
   suffix stripped, sorted) + first 3 digits of the postal code. Cheap, catches the
   common case, but a typo in the first letters defeats it on its own.
2. **sorted-initials + city**: sorted first letters of each significant word + city —
   survives word-order differences ("Northgate Bluewater Logistics" vs "Bluewater
   Northgate Logistics").
3. **phonetic (Soundex) + city**: survives spelling variation that changes the first
   letters (e.g. a typo'd first letter that strategy 1 would miss entirely).

Blocks larger than 60 members are skipped for that strategy (a huge block signals a
low-information key). Measured separately from final match quality, per the spec's
explicit call-out: **blocking recall = 0.878** on the demo dataset (117 of 957 true
duplicate pairs didn't survive any blocking strategy — mostly pairs where a typo hit
*both* the name-prefix key's letters *and* changed the phonetic code enough to miss,
while also landing in a different city-derived block due to a dropped/garbled city
field). `tests/test_blocking.py::test_blocking_recall_on_known_true_matches` asserts
this stays ≥ 0.85 on a held-out generated sample.

## Scoring & thresholds

`HeuristicScorer` computes six features per pair (`rapidfuzz` token-sort/partial ratio,
Jaro-Winkler on the name; token-sort ratio on the address; exact match on postal code;
`rapidfuzz.ratio` on city) and combines them with hand-tuned weights (name-dominant,
since it's the only field guaranteed present; postal/city corroborate but are
down-weighted since they're often missing in messy multi-source data — a missing field
scores a neutral 0.5, not 0, so it doesn't unfairly tank the score).

Two thresholds carve `[0,1]` into three bands:

```
score >= match_threshold (0.75)      -> match
score <= no_match_threshold (0.60)   -> no_match
otherwise                            -> review   (surfaced, never auto-decided)
```

Chosen by sweeping both thresholds against the demo dataset's hand-labelled sample:

| match_threshold | no_match_threshold | precision | recall | F1 | review queue size |
|---|---|---|---|---|---|
| 0.60 | 0.55 | 0.792 | 0.851 | 0.820 | 119 |
| 0.70 | 0.60 | 0.873 | 0.861 | 0.867 | 151 |
| 0.72 | 0.65 | 0.899 | 0.845 | 0.871 | 161 |
| **0.75** | **0.60** | **0.924** | **0.856** | **0.889** | **200** |
| 0.75 | 0.55 | 0.924 | 0.856 | 0.889 | 251 |
| 0.78 | 0.55 | 0.959 | 0.799 | 0.872 | 339 |
| 0.83 | 0.55 | 0.980 | 0.659 | 0.788 | 493 |

0.75/0.60 sits near the F1 peak; a looser `no_match_threshold` (0.55) gets the same F1
by pushing ~25% more pairs into the review queue for no precision/recall gain, so 0.60
is the better operating point. Raising `match_threshold` further buys precision at a
steep recall cost — reasonable if false merges are especially costly, but the spec's
framing ("without false merges" *and* "a merged view of entities that are actually the
same") suggested balancing both rather than maximizing precision alone.

## Clustering & the transitive-chain edge case

`match` decisions are merged into clusters via union-find (transitive closure): if
A≈B and B≈C both score as matches, A/B/C become one cluster. This has a known failure
mode the spec calls out explicitly — chaining can merge entities that individually
don't belong together. Mitigation implemented in `resolve.py`: after forming clusters,
each one is checked against every *explicitly scored* `no_match` pair among its
members. If two cluster members were themselves scored as a firm `no_match`, that's
contradicting evidence — the cluster is downgraded to `flagged_conflict` and every pair
touching it is routed to the review queue instead of being silently merged or kept
apart. This doesn't catch every possible false chain (two members might simply never
have been directly compared, if blocking never paired them), but it catches the case
where the pipeline has explicit contradicting evidence for itself.
`tests/test_resolve.py::test_transitive_chain_with_explicit_conflict_is_flagged_not_silently_merged`
covers this directly with a constructed 3-pair example.

## Edge cases handled (per spec section 9)

| Edge case | How it's handled | Test |
|---|---|---|
| Franchise-like distinct entities with very similar names must not be over-merged | Address/postal/city features pull the score down even when the name matches exactly; the synthetic generator creates deliberate "confusable sibling" entities (same brand name, different city) specifically to check this | `test_confusable_franchise_entities_are_not_auto_merged`, `test_confusable_franchise_siblings_score_low_despite_similar_name` |
| Blocking key misses a true match due to a typo | Blocking recall measured **separately** from final match recall (0.878 vs 0.856); 3 independent strategies so a typo defeating one doesn't defeat all | `test_blocking_recall_on_known_true_matches`, `test_a_typo_riddled_true_match_can_still_be_missed_by_a_single_strategy` |
| Transitive-match clustering chains | Conflict-flagging described above | `test_transitive_chain_with_explicit_conflict_is_flagged_not_silently_merged` |
| Missing fields (no address, no postal code, no city) | Missing-field features score neutral (0.5), not 0; normalization/scoring/blocking all tested with `missing_field_rate` up to 0.4 without crashing | `test_missing_fields_across_dataset_do_not_crash_pipeline`, `test_missing_fields_do_not_crash_scoring` |
| Review band silently merged or dropped | `resolve()` returns it as a first-class list; tested that review-queue pair ids never overlap with matches/no_matches | `test_review_queue_pairs_are_not_double_counted_as_matches_or_no_matches` |

## Metrics explained

- **Blocking recall** — of all true duplicate pairs (from ground truth), what fraction
  survived blocking and became a candidate pair at all? Measures blocking's own
  performance, independent of scoring/thresholds.
- **Full-dataset precision/recall/F1** — pipeline's final `match` decisions vs. every
  true duplicate pair in the ground truth, across the whole dataset.
- **Hand-labelled sample precision/recall/F1** — what the spec's Definition of Done
  actually asks for: metrics restricted to a stratified sample of candidate pairs
  (weighted toward the ambiguous mid-band, which a uniform random sample would mostly
  miss). See the honesty note below.
- **Flagged-conflict clusters** — clusters where transitive-match chaining hit
  contradicting evidence and got downgraded to review instead of auto-merged.

**Honesty note on "hand-labelled":** the spec asks to "hand-label a sample of candidate
pairs." Because this dataset is synthetic and generated by this project, the true
duplicate clusters are known exactly — there's no human labeler in the loop.
`evaluation.hand_label_sample()` simulates the hand-labeling step by drawing a
stratified sample across score bands and assigning each pair's label from ground truth,
i.e. what a perfectly accurate human reviewer would say. This gives real, reproducible
precision/recall numbers instead of fabricated ones, at the cost of not exercising
actual human-labeling noise or disagreement — a real limitation, noted here rather than
glossed over.

## Install

Python 3.10+.

```bash
git clone https://github.com/HamzaOuadid/entity-resolution-tool.git
cd entity-resolution-tool
pip install -e ".[dev]"
```

## Usage

```bash
# 1. Generate the messy multi-source synthetic dataset (deterministic w/ seed)
entity-resolution generate-data --out data --n-entities 350 --seed 42

# 2. Run the full pipeline end-to-end: normalize -> block -> score -> resolve -> evaluate
#    Prints a metrics report, persists every stage to SQLite, writes eval_report.json
entity-resolution run-all --data-dir data --db entity_resolution.db --report-out eval_report.json

# 3. List pairs in the review queue with a feature-level explanation of the ambiguity
entity-resolution review-queue --db entity_resolution.db --limit 10

# 4. Export the full review queue to CSV for offline review (spreadsheet-friendly)
entity-resolution export-review-queue --db entity_resolution.db --out review_queue.csv

# 5. Record a reviewer's decision on a specific pair (closes the loop)
entity-resolution submit-review <pair_id> match --reviewer alice --db entity_resolution.db
```

Each command's `--help` documents its options (thresholds, dataset size, etc.).

### Library usage

```python
from entity_resolution.datagen import generate_dataset
from entity_resolution.pipeline import run_pipeline

dataset = generate_dataset(n_entities=350, seed=42)
result = run_pipeline(dataset)
print(result.metrics)               # the numbers quoted above
print(len(result.resolve_result.review_queue))  # 200
```

## Testing

```bash
pytest tests/ -v
```

```
72 passed in ~30s
```

72 tests across normalization, synthetic-data generation, blocking (incl. blocking
recall and the typo-miss edge case), scoring (incl. the franchise-siblings edge case),
resolution/clustering (incl. the transitive-conflict edge case), evaluation, the CLI,
the reviewer-facing explanation/export tooling, and full pipeline end-to-end runs
against the real synthetic dataset.

## What's implemented vs. deferred

**Implemented:**
- Normalization: legal-suffix canonicalization, address abbreviation expansion,
  postal-code and city recovery from free text (needed since one of the three
  synthetic sources only has a single free-text address field)
- 3 independent blocking strategies with measured blocking recall
- Two pairwise scorers (heuristic default + trainable logistic-regression alternative)
  behind one interface
- Threshold-based match/no_match/review resolution with union-find clustering and a
  conflict-detection safeguard for transitive-chain over-merging
- Full precision/recall/F1 on both the full dataset and a stratified hand-labelled
  sample, plus blocking recall measured separately
- A durable review queue: list with feature-level explanations, CSV export for offline
  review, and a way to record a reviewer's decision back into the data model
- SQLite persistence matching the spec's exact data model
- 72 passing tests covering every edge case section 9 of the spec calls out

**Deliberately deferred** (see [Risks](#risks--open-questions) for why):
- Real public datasets, in favor of synthetic data with known ground truth (spec
  explicitly permits this per the task's framing; licensing risk is the reason)
- Postgres, in favor of SQLite (no Postgres/Docker available in this environment; the
  workload doesn't need it)
- Splink/`dedupe` specifically, in favor of a hand-rolled scorer using the same
  underlying technique (feature engineering + logistic regression) with lighter
  dependencies
- An LLM-assisted tie-breaker for the review band — the spec doesn't call for one, and
  entity resolution here is a data-engineering/ML problem, not a language problem; the
  heuristic feature breakdown (`review.py::explain_pair`) already gives a reviewer
  enough signal to decide quickly without one
- A real-time/streaming resolution mode — explicitly out of scope per the spec's
  non-goals; this is a batch pipeline
- Wiring the CSV loaders' already-structured city/postal columns (registry, crm) directly
  into `normalize()` for the CLI path — currently even structured sources are re-parsed
  from the concatenated free-text address via the same regex path as the free-text-only
  `directory` source. Works correctly (tested), but a production version would prefer
  the structured field when available rather than re-deriving it

## Risks / Open Questions

- **Public dataset licensing** (spec section 13): the spec's own environment notes flag
  this as unresolved even for real datasets — confirming redistribution rights for a
  derived/merged dataset from two public sources is nontrivial and out of scope for
  this exercise. Using synthetic data sidesteps the question entirely for this
  portfolio piece, at the cost of not proving the pipeline against the specific messiness
  of any particular real registry (this project's synthetic generator was designed to
  mimic the *kinds* of messiness real multi-source data has — typos, abbreviations,
  format drift, partial records — but real data will have failure modes this generator
  doesn't produce).
- **Hand-labelling is simulated, not human** — see the honesty note under
  [Metrics explained](#metrics-explained). If this were deployed against real data, an
  actual reviewer's labels (with real disagreement/noise) would very likely show lower
  agreement than the synthetic ground truth implies.
- **Blocking's `max_block_size=60` cap** trades recall for a worst-case bound — a
  legitimate block larger than 60 (e.g. a very common name+postal combination) would
  silently lose all its pairs. Not triggered on the demo dataset's scale but would need
  tuning or a smarter fallback (e.g. sampling within an oversized block) at real scale.
- **Threshold tuning was done on the same synthetic dataset it's evaluated against** —
  there's no separate held-out dataset with a different seed/distribution used to
  confirm the 0.75/0.60 thresholds generalize; they're tuned and evaluated on the same
  971-row generation (different specific pairs than the hand-labelled sample, but the
  same underlying data-generating process).

## License

MIT — see [LICENSE](LICENSE). Author: Hamza Ouadid.
