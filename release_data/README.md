# LADDER experimental input manifests

This directory contains the de-identified **input selection** for the paired
experiments reported in the LADDER manuscript. It does not contain real learner
records. The 30 learner states are synthetic and fixed within each evaluated
pair. `pairs_main_900.jsonl` selects 30 questions × 30 learners; each of the
other three manifests selects 20 questions × the same 30 learners.

## Files

- `learner_states.jsonl`: one structured synthetic state per pseudonymous learner.
- `pairs_main_900.jsonl`: normal-request comparison with DeepSeek-Chat.
- `pairs_refusal_600.jsonl`: refusal-suppression attack comparison.
- `pairs_role_play_600.jsonl`: role-playing attack comparison.
- `pairs_cross_model_600.jsonl`: GPT-5.4-mini teacher-model replication.
- `source_checksums.json`: upstream question/graph filenames and SHA-256 checksums.

Each pair row provides its order, a stable pair ID, a synthetic learner ID, and
the **zero-based question index** into the SHaPE source question file. The same
pair selection was verified for LADDER and SHaPE in each comparison.

## Upstream questions and knowledge graph

Obtain `LinearAlgebra_hard_ds_steps_parallel.json` and
`adjacency_matrix_knowledge_graph.csv` from the [SHaPE authors' data
repository](https://github.com/MAPS-research/SHaPE/tree/main/data). The
question text, answers, and knowledge graph are not redistributed here. Use
`source_checksums.json` to verify that your copies match the source files used
for these manifests. The question file is JSON Lines despite its `.json`
extension.

The upstream work should be cited as: Zhao et al., *SHAPE: Unifying Safety,
Helpfulness and Pedagogy for Educational LLMs*, ACL 2026. Please follow the
upstream repository's academic-use terms.

## Scope and exclusions

These files identify the inputs, **not the measured results**. They exclude
model responses, grader outputs, statistical summaries, human annotation
files, API credentials, `.env` files, original question text, and personal
paths. Reproducing numerical results requires running the released experiment
code with the upstream SHaPE data and the specified model services.
