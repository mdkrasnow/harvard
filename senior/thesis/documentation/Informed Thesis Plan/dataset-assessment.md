# Dataset assessment for Scalar EqM

**Matt Krasnow — October 17, 2026**

## Recommended dataset portfolio

Use **TrpB and GB1 for the main measured-neighborhood experiments**, **ThermoMPNN's curated MegaScale data for stability prediction across protein backgrounds**, and **the Poelwijk fluorescent-protein landscape for a small, complete pilot**. Use the four GFP landscapes as a later test of transfer between related proteins. CreiLOV is a strong optional second fluorescence benchmark.

This portfolio supports the proposal's function-prediction and neighborhood questions. It does not supply a large, independently validated benchmark of eventual directed-evolution success across starting proteins. That strongest claim remains conditional on a separate data source.

The rankings below concern experimental geometry, label quality, independence, accessibility, and thesis scope. No dataset establishes in advance that EqM will outperform an ordinary scalar predictor.

| Priority | Dataset | Main purpose | Checkpoints | Main restriction |
| --- | --- | --- | --- | --- |
| Pilot | Poelwijk fluorescent protein | Complete small landscape; validate the evaluation pipeline | 2.2–2.4, 3.1–3.4 | Thirteen binary choices within one protein family |
| Core | TrpB | Main enzyme-function and neighborhood benchmark | 2.2–2.4, 3.1–3.4 | Selected sites; separate measured and imputed variants |
| Core | GB1 | Complementary binding landscape and established comparison | 2.2–2.4, 3.1–3.4 | Four sites; incomplete coverage and floor effects |
| Core | MegaScale through ThermoMPNN | Generalization to unseen protein backgrounds and stability neighborhoods | 2.2–2.4, qualified 3.2–3.4 | Single-mutant stars around original references |
| Extension | Somermeyer four GFP landscapes | Transfer across related parental backgrounds; robustness | 2.3–2.4, qualified 3.3–3.4 | Sparse random libraries; only four parental landscapes |
| Extension | CreiLOV | Larger combinatorial fluorescence test | 2.3–2.4, 3.2–3.4 | Restricted, deliberately selected substitutions; missing measurements |
| Extension | Escobedo hydrophobic cores | Stability/abundance and transfer across three backbones | 2.3–2.4 | Measured subset is much smaller than theoretical space |
| External check | Curated Fireprot homologue-free subset | Generalization to another measurement modality | 2.3 | Sparse, selectively measured mutations |
| External check | Domainome 1.0 | Cellular abundance across hundreds of domains | 2.3, reference-neighborhood analyses | Abundance proxy; single-mutant geometry |
| Access routes | ProteinGym, FLIP2, SSMuLA, MaveDB | Standardized data, provenance, splits and baselines | 2.2 | Repackage overlapping source experiments |

## Nick's suggested paper

Nick's [PNAS paper](https://doi.org/10.1073/pnas.2314853121) is Dieckhaus et al. (2024), *Transfer learning to leverage larger datasets for improved prediction of protein stability changes*, introducing ThermoMPNN. It is a useful recommendation for the stability branch of this thesis. Its experimental labels are mutation-induced stability changes, rather than measured evolutionary outcomes. The study reports a curated MegaScale set of 272,712 mutations across 298 proteins and a curated Fireprot set of 3,438 mutations across 100 proteins. [Official repository](https://github.com/Kuhlman-Lab/ThermoMPNN).

### Why its data are useful

The repository supplies [training](https://github.com/Kuhlman-Lab/ThermoMPNN/blob/main/data_all/training/mega_train.csv), [validation](https://github.com/Kuhlman-Lab/ThermoMPNN/blob/main/data_all/training/mega_val.csv), and [test](https://github.com/Kuhlman-Lab/ThermoMPNN/blob/main/data_all/testing/mega_test.csv) tables. They retain reference and mutant sequences, stability estimates, and measurement-quality information. The [split definitions](https://github.com/Kuhlman-Lab/ThermoMPNN/blob/main/dataset_splits/mega_splits.pkl) allocate 239/31/28 reference names to train/validation/test. The [MMseqs cluster table](https://github.com/Kuhlman-Lab/ThermoMPNN/blob/main/dataset_splits/megascale_clusters.tsv) assigns 123/18/21 clusters respectively. Preserve these definitions rather than randomly mixing mutations from each protein across splits.

The downloaded test table contains 28,312 rows: 28,172 unique non-WT single mutants and 140 WT controls representing 28 reference proteins. Its measured single-mutant coverage ranges from approximately 83.6% to 100% of the theoretical 19L substitutions per reference. Repeated WT controls are not additional starting proteins. The CSV's source `WT_cluster` annotation differs from the MMseqs grouping used for the published splits; use the split files for that purpose. [Test data](https://github.com/Kuhlman-Lab/ThermoMPNN/blob/main/data_all/testing/mega_test.csv), [cluster metadata](https://github.com/Kuhlman-Lab/ThermoMPNN/blob/main/dataset_splits/megascale_clusters.tsv).

A complete single-mutant scan forms a **star around its original reference**. It does not supply a complete neighborhood around every measured mutant: changing a second position creates a double mutant, excluded from this curated set. This dataset is therefore suitable for comparing stability neighborhoods across original references, but cannot turn thousands of mutants into thousands of fully measured starting backgrounds.

The [loader](https://github.com/Kuhlman-Lab/ThermoMPNN/blob/main/datasets.py) negates source `ddG_ML`; negative ThermoMPNN ΔΔG means stabilizing. Keep absolute stability, stability change, and learned model potential distinct. Measurement filtering can affect estimated neighborhood fractions, so report coverage and initially describe summaries as applying to the usable measured neighbors.

The underlying [Tsuboyama et al. study](https://www.nature.com/articles/s41586-023-06328-6) and [original data archive](https://zenodo.org/records/7992926) contain broader data, including mutation types excluded by the ThermoMPNN curation. Begin with the curated subset; expand only for a defined experiment. Proteolysis-derived stability estimates have a limited dynamic range and assay-specific artifacts. An absolute stability threshold and a threshold for stabilizing mutations answer different questions.

## Dense landscapes for the central neighborhood question

### TrpB is the strongest main enzyme benchmark

The [Johnston et al. paper](https://doi.org/10.1073/pnas.2400439121) provides one four-site and nine three-site landscapes around an engineered TrpB reference. Growth complementation is the function readout. This offers many measured combinatorial genotypes with different local opportunities within a defined mutation space. It is a strong setting for asking whether Scalar EqM predicts local opportunities better than the same architecture trained only on assay labels.

Use the [primary Caltech archive](https://data.caltech.edu/records/h5rah-5z170), or the [SSMuLA collection](https://github.com/fhalab/SSMuLA) with its [versioned archive](https://zenodo.org/records/15203754). The primary archive separates `four-site_simplified_AA_data.csv` and summary measurements from `20230828_KNN_imputed_TrpB.csv`. **Exclude imputed variants from experimental test truth.** Published downstream counts report 159,129 measured four-site variants; this exact count still requires reconciliation against the selected primary table.

The downloaded [FLIP2 one-to-many table](https://flip.protein.properties/assets/splits/trpb/one_to_many.csv.gz) has 228,298 rows and unique full sequences across the combined landscapes. This is a useful access route, but the combined collection is neither 228,298 independent starting proteins nor one complete landscape over all mutated positions. Retain each sequence's source landscape and measured/imputed status. [FLIP2 manuscript](https://flip.protein.properties/assets/FLIP_manuscipt.pdf).

**Use:** assay ranking, mutation-distance extrapolation, neighborhood fractions and upper-tail improvement. Replicate across site sets while accounting for shared sequences and overlapping positions. Simulated walks on the table can test a declared adaptation model; they are not new experimental evolution campaigns.

### GB1 is the strongest established complementary benchmark

The [Wu et al. study](https://doi.org/10.7554/eLife.16965) reports 149,361 measured variants of a possible 160,000 four-site combinations. Binding enrichment is the phenotype. The [author repository](https://github.com/wchnicholas/ProteinGFourMutants) provides measurements and a separate inferred-missing collection. Its original `Mutfit` table contains NA and stop-containing entries, so import it with explicit filtering. ProteinGym's current reference lists 149,360 benchmark variants, reflecting preprocessing rather than a second experiment.

This is a useful test of neighborhood structure under a phenotype different from TrpB growth complementation. Every claim concerns the four allowed sites. Many variants have low or floor-level scores; a correlation alone can obscure failure to identify beneficial neighbors. Missing variants must remain unknown rather than becoming nonfunctional labels.

**Use:** paired model comparisons for held-out neighborhood summaries and a separate, clearly labeled simulation analysis. Include additive and pairwise sequence models as biological baselines alongside embedding-based heads.

### Poelwijk is the best small complete pilot

[Poelwijk et al. (2019)](https://doi.org/10.1038/s41467-019-12130-8) measured the 8,192 combinations connecting two fluorescent proteins through thirteen binary amino-acid choices. The downloaded [Supplementary Data 3](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-019-12130-8/MediaObjects/41467_2019_12130_MOESM7_ESM.xlsx) contains 8,192 genotype records after two header rows, including genotype identifiers, variable-site sequences, count data and channel-specific/combined brightness. [Supplementary Data 4](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-019-12130-8/MediaObjects/41467_2019_12130_MOESM8_ESM.xlsx) includes both trials.

The complete cube gives every genotype thirteen defined one-step neighbors, making coverage easy to audit. It is small enough for repeated full-pipeline comparisons. These are experimentally derived enrichment phenotypes, with transformations and pseudocounts; declare the target and test transformation sensitivity. A preprocessing transform fitted to all measurements should not silently become a training-only inference step.

**Use:** establish that neighborhood evaluation works before scaling. This tests interpolation/extrapolation within one restricted landscape, not transfer to new protein families or independent long-term evolvability.

## Additional datasets worth retaining

### Four GFP landscapes

[Somermeyer et al. (2022)](https://doi.org/10.7554/eLife.75842) compare four related fluorescent-protein landscapes with different robustness and epistasis. The [author repository](https://github.com/aequorea238/Orthologous_GFP_Fitness_Peaks) supplies data and reference information. The inspected combined new-ortholog CSV contains 93,925 rows: 35,500 amacGFP, 26,165 cgreGFP, and 32,260 ppluGFP. Its columns include mutation descriptions, cell counts, replicate counts and mean brightness.

The avGFP dataset reuses Sarkisyan's earlier experiment. Its reformatted table has 54,025 rows, of which 2,310 contain stop symbols; excluding those leaves 51,715, including WT. ProteinGym applies additional preprocessing, so its counts differ. Do not count the author, Sarkisyan and ProteinGym copies as independent validation sets. Native and aligned mutation coordinates also differ.

**Use:** a demanding leave-one-ortholog-out extension and comparisons of sampled mutational robustness. The libraries do not provide every neighbor of arbitrary mutants. Four parental landscapes are four biological backgrounds, regardless of the total number of variants. Preserve linear-versus-log brightness distinctions and assay calibration.

### CreiLOV

[Chen et al. (2023)](https://doi.org/10.1021/acssynbio.2c00662) report 2,185 reliable single-site measurements and 165,428 reliable combinatorial measurements out of a designed 184,320-combination library. This is an attractive larger fluorescence alternative. The chosen combinatorial substitutions came from earlier single-mutant results, so the benchmark is conditional on that library design. Before promotion into the core portfolio, reconcile the supporting spreadsheets, full-sequence reconstruction, missingness and reuse terms. [Primary abstract](https://pubmed.ncbi.nlm.nih.gov/37066862/).

### Hydrophobic cores

The expanded [Escobedo et al. Science study](https://doi.org/10.1126/science.adq3948) has [processed data](https://zenodo.org/records/16266645) and [author code](https://github.com/lehner-lab/combinatorialcores). FLIP2 supplies a convenient subset covering three protein backbones. Its downloaded `three_to_many.csv.gz` contains 24,935 unique sequences. With seven sites and five allowed hydrophobic amino acids, the combined theoretical space is 3 × 5^7 = 234,375: this subset covers about 10.6%, not a complete landscape. Model-reconstructed landscapes are predictions, not extra observations.

**Use:** transfer between backbones and high-mutation extrapolation. The primary assay and inferred energetic quantities need separate labels; do not pool abundance, stability and binding as a single target. Audit coverage before using local neighborhood fractions.

### External stability and abundance checks

ThermoMPNN's [curated Fireprot archive](https://zenodo.org/records/8169289) includes a homologue-free subset with 2,578 rows over 89 proteins. It is useful for testing generalization from proteolysis-derived estimates to literature biophysical measurements, but sparse targeted mutations cannot estimate an unbiased whole-neighborhood beneficial fraction.

[Domainome 1.0](https://zenodo.org/records/14356805) supplies experimental cellular-abundance scores across hundreds of domains. Use the experimental score table, not the separate homolog-prediction table, as ground truth. This is an optional external test of mutational tolerance in cells, with different measurement biology from thermodynamic folding stability.

[FireProtDB 2.0](https://academic.oup.com/nar/article/54/D1/D409/8329105) and its [downloads](https://loschmidt.chemi.muni.cz/fireprotdb/download/) aggregate multiple source datasets, including MegaScale and Domainome. Its millions of records are not millions of independent experiments newly validating this thesis. Separate endpoints, conditions and source publications, and deduplicate against training data. The [FLIP Meltome resource](https://flip.protein.properties/assets/splits/meltome/README.md) is useful for melting-temperature prediction but lacks the dense mutation neighborhoods central to Q3.

## Collections and provenance

[ProteinGym](https://github.com/OATML-Markslab/ProteinGym) is useful for standardized sequences, score direction, supervised splits and baseline comparisons. The fetched substitution reference has 217 assay records. Start with selected benign fluorescence, binding and stability assays rather than the entire suite. [FLIP2](https://flip.protein.properties/) offers useful distribution-shift splits; [SSMuLA](https://github.com/fhalab/SSMuLA) supplies dense-landscape tooling; [MaveDB](https://mavedb.org/docs/mavedb/programmatic-access/api-quickstart.html) provides a broader searchable source of experimental scores.

Maintain a source-experiment identifier in addition to the downloaded dataset name. TrpB appearing in FLIP2 and SSMuLA is one source experiment. MegaScale appearing in ThermoMPNN, ProteinGym and FireProtDB remains overlapping evidence. Published benchmark normalization may use test-set maxima or globally fitted transforms; use original labels or document and evaluate that choice explicitly.

## Match the training target to the dataset

TrpB, GB1 and fluorescence assays associate an experimental phenotype with a sequence within a fixed context. MegaScale also supports a paired target: a mutation effect relative to a reference. For a shared scalar potential, compare an explicitly signed difference E(mutant) − E(reference) with the measured change, or use quality-controlled absolute stability labels where appropriate. Do not silently treat every reference-relative ΔΔG as an absolute label on the mutant sequence. Apply the same target construction to the supervised-only baseline.

A dataset with genuine thermodynamic labels does not make an EqM generative potential a physical folding energy. Conversely, an assay need not measure physical energy to test whether EqM regularization helps predict its phenotype. The core test is an objective comparison under identical inputs, targets and held-out measurements.

## What remains unresolved for Question 4

The proposal's comparative-evolution candidates establish that starting background can matter, but they do not yet constitute a sufficiently large independent starting-point benchmark. [Baier et al.](https://doi.org/10.7554/eLife.40789) compare four starting orthologs. The [AI-redesigned starting-points study](https://doi.org/10.1038/s41586-026-10820-0) also studies four starting points, with repeated campaigns and multiple conditions. Replicates and conditions do not create dozens of independent protein backgrounds. Retain these as literature context rather than the core training benchmark.

Public GFP evolution studies also exist, including [the expression-level study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7613228/) and [the temperature study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10980763/). They vary biological conditions and therefore do not directly isolate selection among many starting protein sequences under one common protocol. They are potentially informative about omitted context, but do not resolve the main Q4 data requirement.

A defensible extension is to ask whether measured neighborhood statistics predict outcomes under a fixed **simulated** mutation-and-selection rule on an experimental landscape. That is an intermediate question between Q3 and experimental Q4, with its endpoint explicitly identified as simulation. It cannot substitute for independently measured evolution outcomes.

## Recommended experiment sequence

- **Q1: Is the scalar EqM formulation correct?**
  - Use synthetic distributions with known structure. Protein assay labels do not provide a known continuous gradient field in embedding space.
- **Q2: Does EqM training improve phenotype prediction?**
  - Pilot on Poelwijk; establish main comparisons on TrpB and GB1; assess held-out-protein stability prediction on curated MegaScale.
  - Compare identical representations and scalar heads under supervised-only and combined objectives. Include simple sequence-based baselines, especially for restricted-site libraries.
  - Fix the link between energy and phenotype before interpreting lower energy as better function.
- **Q3: Does the model reconstruct useful measured neighborhoods?**
  - Evaluate complete/reliably covered neighborhoods in the restricted landscapes and reference-centered stars in MegaScale.
  - Compare predicted and measured functional fraction, beneficial fraction and upper-tail improvement using the same observed neighbor set for every model.
  - Hold out the measurements used to construct each test summary. Track overlapping neighborhoods and group uncertainty by landscape or protein cluster.
  - A random variant split tests recovery of missing labels. A whole-background or separated-region split tests a stronger form of generalization. Report both with distinct interpretations.
- **Intermediate extension: Do neighborhood summaries predict simulated adaptation?**
  - Freeze the prediction model; use held-out experimental measurements to evaluate a declared simulation endpoint. Keep this separate from real evolution.
- **Q4: Do predictions identify better experimental starting proteins?**
  - Require sufficient independent starting backgrounds, matched conditions, independently measured endpoints and an agreed holdout strategy. Otherwise retain this question as conditional and scope the completed claim to local improvement potential.

## First acquisition and audit tasks

1. Import Poelwijk's experimental supplementary tables and reconstruct full sequences using the parent references.
2. Pin ThermoMPNN's curated CSVs, split definitions and cluster metadata; preserve control identities and label signs.
3. Obtain measured-only TrpB and GB1 tables with parent sequences, quality flags and source-landscape identifiers.
4. Produce a coverage table before training: unique sequences, references, allowed mutation choices, missing/censored measurements, label units, replicate availability, and evaluable test neighborhoods.
5. Freeze the main targets, splits and model-comparison budget. Add GFP or CreiLOV only after the core pipeline works.

The small public tables noted above were inspected directly. The multi-gigabyte primary archives were located but not exhaustively downloaded or audited. Exact usable neighborhood counts, dataset-specific licenses, and final label/split choices remain acquisition checks rather than completed experimental results.
