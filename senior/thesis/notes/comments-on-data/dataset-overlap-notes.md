# Dataset overlap review notes

October 17, 2026 — Review of `dataset-assessment.md` and linked primary sources.

**Recommendation:** Retain the proposed core portfolio: Poelwijk, TrpB, GB1, and curated MegaScale. No source-level evidence found that these four are copies of one underlying experiment. This is not a sequence-level certification of independence. Most important caveats are already present in the assessment.

## Confirmed overlap and dependence

| Resources | Finding | Required handling |
| --- | --- | --- |
| TrpB primary archive, FLIP2, SSMuLA | Repackage the Johnston source landscapes. SSMuLA also distributes GB1. | Use one canonical import per source experiment; collections are not independent validation sets. |
| MegaScale original data, ThermoMPNN, ProteinGym, FireProtDB | Contain overlapping Tsuboyama measurements. | Do not train on one distribution and treat another as independent test truth. |
| Somermeyer avGFP and Sarkisyan avGFP | Somermeyer reuses the earlier avGFP experiment alongside three newly measured GFP landscapes. | Count avGFP once, including copies obtained through benchmark collections. |
| Domainome and FireProtDB 2.0 | FireProtDB incorporates Domainome as well as MegaScale. | Filter by source before using FireProtDB for external validation. |
| CreiLOV single-mutant and combinatorial libraries | Combinatorial substitutions were selected from single-mutant results; the combinatorial measurements include the 20 selected single mutants. | Deduplicate shared genotypes across splits; interpret results as conditional on the selected library. |
| CreiLOV paper's additional benchmarks and Sarkisyan avGFP | The paper separately reuses avGFP for modeling comparisons. Its new CreiLOV measurements are distinct. | Import CreiLOV supporting datasets S1/S2 specifically, rather than treating every benchmark used in the paper as new data. |

Sources: [SSMuLA dataset definitions](https://github.com/fhalab/SSMuLA#datasets), [FLIP2 manuscript](https://flip.protein.properties/assets/FLIP_manuscipt.pdf), [ProteinGym paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10723403/), [FireProtDB 2.0 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12807726/), [Somermeyer paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9119679/), [CreiLOV paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10204710/).

## Other constraints on interpretation

- **TrpB landscapes share a reference and overlapping sites.** They are complementary site-set tests, not ten independent protein backgrounds. Check full-sequence duplicates before splitting across landscapes. [Site definitions](https://github.com/fhalab/SSMuLA#datasets)
- **Fireprot homologue-free status is relative to the screened MegaScale data.** ThermoMPNN used a 25% sequence-identity cutoff; recheck independence if training data or supervised pretrained models change. [ThermoMPNN methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10861915/)
- **Poelwijk and Somermeyer are distinct experiments.** Poelwijk measures mTagBFP2/mKate2 combinations; Somermeyer studies four other fluorescent-protein backgrounds. Shared fluorescent-protein biology still limits claims of transfer across unrelated families. [Poelwijk](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746860/), [Somermeyer](https://pmc.ncbi.nlm.nih.gov/articles/PMC9119679/)
- **Keep predictions out of experimental test truth:** TrpB imputation, GB1 inferred missing variants, reconstructed hydrophobic-core landscapes, and Domainome homolog predictions. [Domainome experimental/predicted file separation](https://zenodo.org/records/14356805)
- **Match claims to coverage and phenotype.** Sparse/selected libraries cannot automatically estimate whole-neighborhood beneficial fractions; curated MegaScale supplies reference-centered single-mutant stars. Binding, growth, fluorescence, abundance, and stability require distinct target definitions.
- **Real evolution remains a separate claim.** The portfolio supports phenotype prediction and measured local opportunity; it does not yet supply a large independent benchmark of experimental directed-evolution outcomes across starting proteins.

## Before freezing splits

Record source experiment, parental sequence, mutation coordinates, assay conditions, and measured/predicted status. Deduplicate full sequences across imports and splits; check parent homology where unseen-background generalization is claimed. Preserve published MegaScale cluster splits and group uncertainty by protein/background rather than treating every variant as independent.

**Scope:** Source-level review only; no exhaustive row-by-row overlap audit, exact intersection counts, or complete data-license review. No finding currently warrants dropping a core dataset, but independence must be checked on the actual imported tables.
