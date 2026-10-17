# Informed Thesis Proposal

**Matt Krasnow**  
**Date:** October 17, 2026  
**Based on:** Thesis Proposal, September 29, 2026

## Title: Scalar Equilibrium Matching for Predicting Protein Evolvability

## Abstract

In my thesis, I will study whether a directly parameterized scalar energy formulation of Equilibrium Matching (EqM) can learn useful protein landscapes and help identify promising starting sequences for directed evolution. The mathematical goal is to determine whether learning a scalar energy through its gradient is an effective alternative to learning an unconstrained vector field. The biological goal is to test whether the predicted mutational neighborhood of a protein provides information about its evolutionary potential beyond its current function.

The project will proceed through a series of questions with measurable intermediate checkpoints. I will first establish whether the scalar formulation learns the intended field and understand its tradeoffs relative to implicit EqM and existing explicit-energy variants. I will then specify how measured protein function enters training and evaluate the resulting model against ordinary supervised predictors using the same protein representations. Next, I will test whether predicted neighborhoods agree with experimentally measured neighborhoods. Finally, if suitable comparative evolution data are available, I will ask whether these neighborhood predictions improve selection of starting sequences for directed evolution.

These stages support progressively stronger claims. Accurate function prediction would establish a sequence-to-function result. Accurate neighborhood prediction would establish a result about local improvement potential. Predicting independently measured outcomes across starting sequences would support the broader evolvability claim. The thesis will distinguish these outcomes rather than treating any one of them as proof of the others.

## Motivation from the Literature

EqM learns an equilibrium gradient field and supports optimization-based sampling. Its explicit-energy variants already match the gradient of a scalar function to the EqM target, but the reported variants lose generation performance relative to implicit EqM [1]. My methodological question is therefore about the effectiveness of a direct scalar parameterization and its training behavior, rather than whether explicit-energy EqM is possible. Prof. Du's suggestion is to parameterize a scalar neural network and train its input gradient against the EqM field target. This differs from regressing scalar outputs onto known energy labels. Earlier work by Du and collaborators also motivates explicit potentials through the operations they enable, including energy-dependent composition and Metropolis-corrected sampling [2]. Energy Matching provides another relevant scalar generative formulation, although it uses a different objective from EqM [3].

There is also substantial precedent for predicting protein properties without EqM. EVmutation uses evolutionary covariation to predict context-dependent mutation effects [4], and UniRep shows that pretrained protein representations support useful supervised property predictors [5]. ProteinGym provides a framework for comparing fitness predictors across experimental assays [6]. These papers make ordinary predictors important baselines: a differentiable supervised predictor already defines a scalar surface and can score nearby mutants. My experiments must establish whether EqM training adds useful information.

The biological motivation comes from evidence that current activity alone does not determine evolutionary potential. Bloom et al. connect stability to mutational tolerance and acquisition of improved functions [7]. Baier et al. show that different starting enzymes can reach different evolutionary outcomes, with initial activity failing to identify the best outcome [8]. Wu et al. demonstrate how epistasis and indirect paths shape accessibility in a combinatorial protein landscape [9], while Starr et al. show how permissive backgrounds change accessible functions [10]. The AI-redesigned starting-points study provides a recent experimental example of redesign improving evolutionary outcomes [11]. Together, these studies motivate asking whether learned neighborhood properties can help predict evolutionary potential; they do not establish that a simple neighborhood statistic will be sufficient.

## Main Questions and Goal Hierarchy

The checkpoints below are intended to produce interpretable results along the way to the full thesis goal. Completing a checkpoint means answering its question with evidence, including a negative result. It does not mean that Scalar EqM must outperform every alternative. Later biological evaluation requires adequate predictive performance and suitable data, but comparisons can continue even if an earlier experiment finds no advantage over a baseline.

- **Overall goal: Determine whether Scalar EqM can learn useful protein landscapes whose local structure helps identify promising starting sequences for directed evolution.**
  - **Question 1: Can a directly parameterized scalar energy learn the EqM target field effectively?**
    - **Checkpoint 1.1: Establish the formulation.**
      - Define the scalar network, corruption process, target field, and gradient-matching objective.
      - Identify the differences from implicit EqM and existing explicit-energy constructions.
      - **Evidence:** A mathematical specification with explicit assumptions and a corresponding implementation.
    - **Checkpoint 1.2: Verify learning on a controlled problem.**
      - Can the model recover a known conservative field and relative energy differences?
      - Does optimization on the learned landscape behave as expected?
      - **Evidence:** Field error, relative-energy comparisons, and visualizations on a synthetic problem with known structure.
    - **Checkpoint 1.3: Establish practical tradeoffs.**
      - Compare direct scalar, implicit EqM, and an existing explicit-energy variant under comparable budgets.
      - Measure approximation quality, training stability, and computational cost.
      - **Evidence:** Repeated-run comparisons that identify where the scalar formulation works well and where it loses performance.
    - **Parent question answered:** Whether the formulation works as intended and what practical tradeoffs it introduces.
  - **Question 2: Can Scalar EqM be adapted to predict measured protein function, and does EqM training add value beyond ordinary prediction?**
    - **Checkpoint 2.1: Define the link between energy and function.**
      - Decide how assay labels enter training and specify the full objective.
      - Separate the EqM component from any conditioning or supervised component.
      - **Evidence:** An explicit objective and a justified interpretation of the scalar output.
    - **Checkpoint 2.2: Establish datasets and evaluation splits.**
      - Select a small set of suitable assays and document label meaning, measurement quality, and sequence coverage.
      - Fix train, validation, and test splits, including a meaningful extrapolation test.
      - **Evidence:** A dataset inventory and evaluation protocol fixed before final model comparison.
    - **Checkpoint 2.3: Establish predictive performance.**
      - Can the model rank held-out variants and enrich for experimentally beneficial variants?
      - How does performance change with mutation distance and training-set size?
      - **Evidence:** Held-out ranking and selection metrics, with uncertainty across runs and assays.
    - **Checkpoint 2.4: Identify the contribution of EqM training.**
      - Compare with linear and nonlinear supervised predictors using the same embeddings.
      - Compare with the same scalar architecture trained only on function labels.
      - If training combines EqM and supervised terms, remove each term separately.
      - **Evidence:** Ablations distinguishing representation, architecture, and objective effects.
    - **Parent question answered:** Whether the biological adaptation predicts function and what EqM contributes to that prediction.
  - **Question 3: Can the learned landscape characterize the mutational opportunities around a starting sequence?**
    - **Checkpoint 3.1: Define testable neighborhood properties.**
      - Specify mutation radius, allowed sites, and the mutation sampling distribution.
      - Distinguish preservation of function from improvement of function.
      - Select a small set of summaries, including functional fraction, beneficial fraction, and accessible improvement magnitude.
      - **Evidence:** Fixed definitions and thresholds applicable to predictions and experimental measurements.
    - **Checkpoint 3.2: Verify neighborhood data coverage.**
      - Identify multiple starting sequences with sufficient measured neighbors.
      - Assess missing measurements, uneven sampling, and overlap with training data.
      - **Evidence:** An audited set of evaluable backgrounds and a protocol that withholds the relevant test measurements.
    - **Checkpoint 3.3: Test neighborhood prediction.**
      - Do predicted neighborhood summaries agree with measured summaries?
      - Can the model distinguish sequences with similar current function but different local improvement potential?
      - **Evidence:** Held-out neighborhood prediction and starting-sequence ranking results.
    - **Checkpoint 3.4: Test the added value of Scalar EqM.**
      - Compute identical neighborhood summaries from ordinary supervised predictors.
      - Compare neighborhood accuracy, including whether any advantage persists at comparable sequence-level prediction accuracy.
      - **Evidence:** A direct comparison of neighborhood information across models.
    - **Parent question answered:** Whether the model captures measurable local opportunities and whether EqM improves their characterization.
  - **Question 4: Does predicted neighborhood structure improve selection of starting sequences for directed evolution?**
    - **Checkpoint 4.1: Establish an independent evolutionary endpoint.**
      - Identify comparable campaigns with multiple starting sequences and document selection conditions and experimental budgets.
      - Choose an endpoint such as improvement after a fixed budget, final activity, or probability of reaching a threshold.
      - **Evidence:** A defined outcome and an assessment of the number of independent backgrounds and replicate campaigns available.
    - **Checkpoint 4.2: Test the premise using measured neighborhoods, where available.**
      - Do measured neighborhood properties predict subsequent evolutionary outcomes?
      - Do they add information beyond initial function and, when available, stability?
      - **Evidence:** An assessment of neighborhood informativeness before adding sequence-model prediction error.
    - **Checkpoint 4.3: Test predicted neighborhoods against outcomes.**
      - Freeze the sequence predictor before evaluating evolutionary outcomes.
      - Compare starting function alone, neighborhood information alone, and their combination on held-out starting backgrounds.
      - **Evidence:** Independent outcome prediction with uncertainty reflecting the experimental grouping.
    - **Checkpoint 4.4: Evaluate starting-point selection.**
      - Under a fixed selection budget, does the model choose starting sequences with better measured outcomes?
      - Does Scalar EqM improve selection relative to ordinary predictors and simple starting-activity baselines?
      - **Evidence:** Held-out selection results, with conclusions limited by the number of independent starting sequences.
    - **Parent question answered:** Whether neighborhood information improves starting-point selection and whether Scalar EqM contributes to that improvement.
  - **Stretch goal: Use the learned landscape to guide sequence discovery.**
    - Compare discrete model-guided mutation search with random mutation and simple score-based search under equal budgets.
    - Evaluate proposals against withheld measurements or new experiments, if available.
    - **Evidence:** Improvement established independently of the proposing model's own scores.

## Data

I will distinguish three kinds of data rather than assume that one dataset supports every question.

1. **Sequence-to-function data:** Protein variants paired with measured activity, stability, binding, or another defined assay outcome. These data support Question 2. I will begin with a small, justified subset of assays rather than attempt a full benchmark immediately.
2. **Measured neighborhood data:** Dense measurements around multiple starting sequences, including combinatorial landscapes where available. These data support Question 3. The four-site GB1 landscape studied by Wu et al. is a candidate, but its coverage is restricted to those sites rather than the entire protein.
3. **Comparative evolution data:** Outcomes from multiple starting sequences evolved under sufficiently comparable conditions. These data support Question 4. The Baier and AI-redesigned starting-points studies are candidates to audit, not datasets whose suitability I will assume from their conclusions.

The initial data audit will document available sequences, assay direction and units, replicates, missing values, mutation coverage, and independent starting backgrounds. Many mutants around one protein do not create many independent tests of starting-point selection. Replicate campaigns improve estimation of a starting sequence's outcome but do not replace additional starting backgrounds.

I will distinguish mutational robustness, local improvement potential, and longer-term evolvability. A measured fraction of functional mutants assesses robustness. A measured fraction or magnitude of beneficial mutants assesses local improvement potential. Outcomes after multiple rounds of mutation and selection assess a longer-term endpoint under a specific experimental protocol. Predicted neighborhood properties are candidate predictors of these outcomes, not their ground truth.

## Method

### Scalar formulation and the connection to function

Let a protein sequence be \(s\), its fixed pretrained embedding be \(z=h(s)\), and its measured assay outcome be \(y\). The core scalar EqM model will parameterize \(E_\theta(z)\). For a corrupted embedding \(z_\gamma\) and the EqM target field \(v_{\mathrm{EqM}}\), the gradient-matching component has the form

\[
\mathcal{L}_{\mathrm{EqM}}
=\mathbb{E}\left[\left\|\nabla_{z_\gamma}E_\theta(z_\gamma)
-v_{\mathrm{EqM}}(z_\gamma;z,\epsilon,\gamma)\right\|^2\right].
\]

The exact corruption schedule, weighting, and target definition will follow the selected EqM specification. This trains a scalar function through its gradient; it does not require known scalar energy labels. I will compare relative energies because gradient matching leaves an arbitrary additive constant.

This objective alone does not establish that lower energy means higher measured function. Checkpoint 2.1 will specify the biological adaptation before large experiments begin. A candidate is an additional supervised or ranking term aligning energy with assay outcomes, but the choice and its weighting remain experimental decisions. If I use a combined objective, I will compare it with the supervised-only and EqM-only versions. I will not describe the unmodified generative objective as fitness supervision.

The learned energy is a model potential. Its relationship to thermodynamic stability, assay function, and sequence probability must be evaluated separately. Raw energy offsets across independently trained models will not be treated as comparable biological measurements.

### Model comparisons and evaluation

I will initially hold the protein embedding model fixed so that comparisons focus on the learned head and objective. The core protein baselines will be a regularized linear predictor, a nonlinear supervised predictor, and the same scalar architecture trained only on function labels. The controlled formulation experiments will additionally include implicit EqM and an existing explicit-energy construction. Energy Matching will provide methodological context; implementing every related generative method is not a prerequisite for the biological tests.

I will use validation data for hyperparameter selection and reserve test data for final comparisons. Random splits can provide a reference result, but I will also use a split appropriate to the intended generalization claim, such as holding out higher mutation distances or whole starting backgrounds. Preprocessing and label scaling will be fit using training data only. Overlapping variants and neighborhoods will be tracked to avoid leakage into the measurements being predicted.

Primary function metrics will include rank correlation and enrichment or precision among top-ranked variants. Numerical error will be reported when assay calibration makes it meaningful. Comparisons will use the same splits, representations, and documented tuning budgets. I will begin with at least three training seeds for the core comparisons and report computational cost alongside predictive results.

### Neighborhood analysis

I will enumerate or sample actual discrete amino-acid mutants, embed them, and score them. This allows evaluation of neighborhoods without first solving continuous embedding-to-sequence decoding. Neighborhood summaries will use consistent mutation budgets and sampling rules across starting sequences.

The first summaries will be the fraction of variants predicted to preserve function, the fraction predicted to improve function, and an upper-tail improvement statistic under a fixed sampling budget. Thresholds will reflect assay precision and biological meaning where possible. I will report performance for absolute neighbor function as well as improvement relative to the starting sequence, since error in the starting score can affect every relative comparison.

I will initially focus on one-step neighborhoods and expand to multi-step neighborhoods or accessible paths only when coverage permits. Continuous gradients may provide additional diagnostics, but embedding directions will not automatically be interpreted as realizable mutations. Ordinary supervised models will receive the same neighborhood analysis.

### Evolvability and independent validation

For Question 4, I will compare models using initial function alone, neighborhood summaries alone, and their combination. Initial stability will be included where measured and comparable. Any fitted mapping from neighborhood features to evolutionary outcomes will be selected and evaluated using held-out backgrounds, rather than training and testing on the same starting sequences.

I will account for mutation budgets, selection conditions, replicate campaigns, and correlated measurements. Uncertainty estimates will respect the relevant grouping; thousands of mutants or repeated campaigns from one starting background will not be counted as thousands of independent backgrounds. Where very few starting points are available, results will be presented as a case study rather than broad evidence of generalization.

## Checkpoint Decisions and Scope

The first decision is data feasibility: can the available evidence support local neighborhood evaluation, longer-term evolution evaluation, or both? This will be resolved early rather than deferred until February.

If the controlled scalar formulation performs poorly, I will first check correctness and optimization under a bounded set of comparisons. I will document a persistent tradeoff rather than make unlimited architecture search part of the thesis. If the protein model predicts function but its neighborhood summaries add no information, that is a substantive answer to the central biological question.

If measured neighborhoods fail to predict later outcomes, this will limit the proposed summaries and their time horizon; it will not show that all landscape information is irrelevant. If comparative evolution data are inadequate, I will explicitly narrow the demonstrated application to local improvement potential and revise the title and claims accordingly.

Sequence generation and new wet-lab experiments are stretch work, not prerequisites for the core evaluation. Gradient-based discrete search methods such as Plug & Play Directed Evolution provide relevant context if the discovery stage becomes feasible [12].

## Proposed Contributions to the Field

### Methodology

I intend to characterize the effectiveness and practical tradeoffs of a directly parameterized scalar EqM model. The contribution will be an evaluated parameterization and biological adaptation, with comparisons that distinguish the effects of the architecture, EqM training, and function supervision. Conservativeness follows from the scalar construction; improved performance or generalization must be demonstrated experimentally.

### Application

I intend to determine whether learned neighborhood properties predict measured local improvement potential and, if the data support it, evolutionary outcomes across starting sequences. The strongest result would show improved starting-point selection beyond current function and ordinary predictors. More limited results about function prediction or neighborhood reconstruction will be reported at their demonstrated scope.

### Reproducibility

The thesis will include dataset provenance, fixed evaluation splits, training objectives, baseline configurations, and reproducible analyses. Positive and negative comparisons will be retained so that conclusions do not depend on selecting only favorable assays or starting sequences.

## Timeline

### October 2026: Establish the formulation and what can be evaluated

- Complete Checkpoints 1.1, 2.1, and 2.2: specify the objectives, audit candidate data, and define splits and endpoints.
- Audit neighborhood coverage and comparative evolution data for Checkpoints 3.2 and 4.1.
- Define the core comparison set and primary metrics before extensive tuning.
- **Deliverable:** A mathematical specification, dataset feasibility report, and experiment plan identifying which biological claims are testable.

### November 2026: Verify the method and establish baselines

- Complete controlled formulation experiments for Checkpoints 1.2 and 1.3.
- Implement ordinary protein prediction baselines and the biological adaptation.
- Run a complete pilot from sequence input to held-out evaluation on one assay.
- **Deliverable:** A working evaluation pipeline, initial formulation comparison, and baseline results.

### December 2026: Evaluate sequence-to-function prediction

- Complete Checkpoints 2.3 and 2.4 across the selected assays and generalization splits.
- Run the core objective and architecture ablations and assess training cost and variability.
- Finalize neighborhood definitions for Checkpoint 3.1.
- Write the Introduction and the formulation and function-prediction Methods sections.
- **Deliverable:** A function-prediction result with a clear account of what EqM training contributes.

### January 2027: Evaluate measured neighborhoods

- Complete Checkpoints 3.3 and 3.4 on held-out backgrounds.
- Test functional fraction, beneficial fraction, and accessible improvement predictions against measurements.
- Where compatible data exist, test measured neighborhoods against later outcomes for Checkpoint 4.2.
- Write the neighborhood Methods and initial Results sections.
- **Deliverable:** A measured-neighborhood result and a decision about the scope of the final evolvability analysis.

### February 2027: Evaluate starting-point selection and consolidate results

- Complete Checkpoints 4.3 and 4.4 if independent comparative evolution data are sufficient.
- Run remaining sensitivity checks for thresholds, mutation radius, and sampling budget.
- Attempt discrete sequence search only if core analyses are complete and independent evaluation is available.
- Set an experimental cutoff and assemble the full Results and Discussion draft.
- **Deliverable:** Final evidence for the strongest supported claim, including limitations and negative findings.

### March 2027: Finish the thesis

- Complete the full draft and revise with Prof. Du.
- Finalize figures, mathematical definitions, data provenance, and reproducibility materials.
- Ensure that the title, abstract, and conclusions match the endpoints actually evaluated.
- Submit according to the department's confirmed deadline.
- **Deliverable:** A completed thesis whose claims can be traced to the question hierarchy and checkpoint evidence.

## References

The papers below are available in the thesis References folder. They motivate the plan; their suitability as sources of usable experimental data will be established by the data audit.

1. Wang and Du (2025). *Equilibrium Matching: Generative Modeling with Implicit Energy-Based Models.* https://arxiv.org/abs/2510.02300
2. Du et al. (2023). *Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC.* https://proceedings.mlr.press/v202/du23a.html
3. Balcerak et al. (2025). *Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling.* https://arxiv.org/abs/2504.10612
4. Hopf et al. (2017). *Mutation effects predicted from sequence co-variation.* https://doi.org/10.1038/nbt.3769
5. Alley et al. (2019). *Unified rational protein engineering with sequence-based deep representation learning.* https://doi.org/10.1038/s41592-019-0598-1
6. Notin et al. (2023). *ProteinGym: Large-Scale Benchmarks for Protein Fitness Prediction and Design.* https://proceedings.neurips.cc/paper_files/paper/2023/file/cac723e5ff29f65e3fcbb0739ae91bee-Paper-Datasets_and_Benchmarks.pdf
7. Bloom et al. (2006). *Protein stability promotes evolvability.* https://doi.org/10.1073/pnas.0510098103
8. Baier et al. (2019). *Cryptic genetic variation shapes the adaptive evolutionary potential of enzymes.* https://doi.org/10.7554/eLife.40789
9. Wu et al. (2016). *Adaptation in protein fitness landscapes is facilitated by indirect paths.* https://doi.org/10.7554/eLife.16965
10. Starr, Picton, and Thornton (2017). *Alternative evolutionary histories in the sequence space of an ancient protein.* https://doi.org/10.1038/nature23902
11. Krasnow et al. (2026). *AI-redesigned starting points and outcomes enhance protein evolution.* https://doi.org/10.1038/s41586-026-10820-0
12. Emami et al. (2023). *Plug & Play Directed Evolution of Proteins with Gradient-based Discrete MCMC.* https://arxiv.org/abs/2212.09925
