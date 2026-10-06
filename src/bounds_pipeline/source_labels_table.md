# Source labels and where they come from

Every `*_source` column of the frame returned by `compute_bounds_for_dataset` holds a short label naming the result that produced the value.
This table maps each label to its number in the paper.

Several labels can be joined by ` + ` when they tie, so `murasugi_tristram + signature` means both achieved the value.

## Slice genus

`best_lower_bound_slice_genus_source`

| label | result |
|---|---|
| `signature` | B.9 |
| `murasugi_tristram` | B.10 |
| `tau_hfk` | B.13 |
| `nu_hfk` | B.14 |
| `nu_s_knotjob`, `nu_s_alternating_signature`, `nu_s_positive` | B.21 (each in its respective case) |
| `fox_milnor_full` | B.27 |
| `fox_milnor_det` | B.28 |
| `exact_positive`, `exact_negative` | B.29 |
| `known_slice_genus` | If the best bound was already known |

The three `nu_s_*` labels are the same bound; the suffix says where `nu_s` came from (KnotJob computation, signature computation on an alternating diagram (B.33), or diagrammatic computation on a positive diagram (B.29)).

`best_upper_bound_slice_genus_source`

| label | result |
|---|---|
| `hfk_seifert_genus` | B.6 |
| `seifert_genus_diagram` | B.7 |
| `exact_positive`, `exact_negative` | B.29 |
| `known_slice_genus` | If the best bound was already known |

`seifert_genus_diagram` is the Seifert genus of the diagram, which bounds the slice genus from above; `hfk_seifert_genus` is the exact Seifert genus of a knot, when it is smaller.

## Strong slice genus

`best_lower_bound_strong_slice_genus_source`

| label | result |
|---|---|
| `strong_components` | B.5 |
| `strong_obstruction` | B.11 (from the signature) or B.18 (from `s`) |
| `strong_murasugi_tristram` | B.11 |
| `strong_nu_s` | B.18 |
| `strong_pseudo_thin` | B.35 |
| `strong_signature` | B.35 with B.33 |
| `from_slice` | Bound on weak slice genus reused |
| `linking_number_obstruction` | When a link is not algebraically split, `g_4^* = infinity`|
| `known_strong_slice_genus` | If the best bound was already known |

`strong_obstruction` says only that `g_4^* >= 1`, because the link is not strongly slice: either `sigma(L) != 0`, which B.11 rules out for a strongly slice link, or `s(L) != 1 - l`, which B.18 rules out.

`best_upper_bound_strong_slice_genus_source` only ever holds `linking_number_obstruction`, `known_strong_slice_genus`, or `trivial`, which is the absence of any upper bound, and the upper bound is set to `9999`.

## Unknotting number

`best_lower_bound_unknotting_number_source`

| label | result |
|---|---|
| `components_abslk` | B.3 |
| `levine_tristram_conway` | B.12 |
| `splitting_number` | B.24 |
| `slice_torus_nu_s` | B.26 |
| `collari_positive`, `collari_positive_bounded` | B.30 |
| `not_unlink_...` | Certificate of non-triviality, see below |
| `known_slice_genus` | B.1 |
| `known_unknotting_number` | If the best bound was already known |

A slice-genus label appearing here (`signature`, `tau_hfk`, ...) means the unknotting bound came through B.1, so read it in the slice table above.
`not_unlink_...` lists the certificates saying that the link is not the unlink and hence u \geq 1: `signature`, `nullity`, `s_invariant`, `determinant` and `hfk_genus`.
`collari_positive_bounded` is the same theorem as `collari_positive`; the suffix says that some component of the positive link has no exact unknotting number, so the row is an interval rather than a value.

`best_upper_bound_unknotting_number_source`

| label | result |
|---|---|
| `crossing_change_diagram` | B.8 |
| `collari_simply_linked`, `collari_positive` | B.30 |
| `known_unknotting_number` | If the best bound was already known |

Both labels are B.30, and they record which hypothesis the diagram certified: `collari_simply_linked` when every crossing between a fixed pair of components has the same sign, `collari_positive` when some relative orientation makes the whole diagram positive.
Positive links are simply-linked, so the two overlap, and on the LinkInfo census they always agree.

## Components

The `component_*_source` columns use the same vocabulary on a single component knot, plus `crossing_change` for B.8, `s_positive` for B.29 and `s_alternating` for B.33.
The rest is bookkeeping: `unknot`, `unevaluated`, `none`, `contradiction`, `knotted:...`, and `known:...`, `subadditive:...`, `table:additivity@...` for values taken from the knot table.

## Suffixes

Unknotting and strong slice genus do not depend on the orientation (B.2), so their bounds are maximised over the orientation class and the label gains the name of the variant that won, as in `components_abslk_L11a32{0}`.

## Databases and agents

The merged tables of the paper (Section 5) also use labels that name a database or an agent instead of a bound of Appendix B.

| label | meaning |
|---|---|
| `KnotInfo` | KnotInfo for knots, LinkInfo for links |
| `Brittenham` | Brittenham's unknotting number data, prime knots up to 14 crossings |
| `DunfieldGong` | Dunfield and Gong's sliceness data: slice genus 0 for a slice knot, the lower bound 1 for a knot that is not slice |
| `Jablonowski` | Jabłonowski's unknotting number and slice genus data, prime knots up to 13 crossings |
| `SeongJinLee` | Seong-Jin Lee's unknotting number data, prime knots up to 13 crossings |
| `Trivially-assigned` | a bound that no database provides, such as the lower bound 0 or the upper bound 9999 that stands for no bound |
| `slice_genus_copy` | the starting value of a ribbon genus column, copied from the slice genus bound of the database |
| `exhibited_by_RL_<framework>` | an upper bound certified by a sequence of moves that an agent driven by the RL policy found: the pure policy or the mixed agent RL+RW |
| `exhibited_by_RW_<framework>` | the same, found by the random walker |
| `manual_jones_nontrivial_20260916` | the lower bound 1 on the unknotting number of `L11n247{0}` and `L11n247{1}`: their Jones polynomial is not that of the unlink |

`<framework>` is one of `unknotting`, `ribbon`, `slice`, `strong_ribbon`, `strong_slice`. A row carrying an `exhibited_by_*` source also stores its certificate in the band path columns of the table.
