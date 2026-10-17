# Thesis Proposal

**Matt Krasnow**  
**Date:** September 29, 2026

## Title: Scalar Equilibrium Matching for Predicting Protein Evolvability

## Abstract

In my thesis, I will study whether a scalar energy-based formulation of Equilibrium Matching (EqM) can be used to model protein fitness landscapes and predict protein evolvability. The statistical and mathematical goal is to develop and evaluate a scalar energy model for sequence-function prediction. The biological goal is to test whether the local structure of a learned fitness landscape can identify protein sequences that are promising starting points for directed evolution.

EqM is a recent generative modeling framework that learns an equilibrium gradient field associated with an implicit energy landscape. The EqM paper introduces explicit-energy variants that degrade performance relative to implicit EqM. This leaves open the question of whether a simpler direct scalar formulation can make EqM behave more like a true energy-based model while preserving model performance.

My thesis will investigate whether my explicit energy based formulation of EqM, dubbed Scalar EqM, can be useful for biological sequence modeling. This method directly parametrizes a scalar energy function. In this setting, the model learns an energy function over protein sequences where lower energy is trained to correspond to higher measured function or fitness. After learning this landscape, I will study whether the local neighborhood around a starting sequence predicts that sequence's evolvability.

## Main Questions to Be Studied

### Question 1: Can my scalar implementation of EqM learn useful energy landscapes, and does it improve over existing approaches?

To answer this question, I will evaluate Scalar EqM against ordinary supervised models on protein embeddings and implicit EqM models. The goal is to demonstrate if the model can predict protein function and to also evaluate directly whether learning a scalar energy provides useful structure beyond a standard predictive model.

### Question 2: Can Scalar EqM learn a relationship between protein sequence and measured function?

The data I will use will use protein variants associated with measured outcomes like activity, stability, or fitness. The model will take a protein sequence and assign it a scalar energy value. This would hopefully lead to a behavior where the scalar energy value of a fit sequence is lower than that of a less fit sequence. The goal of this question is thus to test whether scalar energy is biologically meaningful.

### Question 3: Can the local energy landscape around a starting sequence predict whether that sequence is a good starting point for directed evolution?

A benefit of using an energy based or energy-like model is that we can also evaluate the neighborhoods around sequences. For a given starting sequence, I will also examine nearby mutants and evaluate their predicted energies or predicted functions. A starting sequence will be considered more evolvable if many nearby variants are predicted to preserve or improve function. The goal of this question is to connect the Scalar EqM method directly to the biological problem at hand: predicting which protein sequences have favorable local evolutionary neighborhoods.

## Data

I will need two sets of data:

1. **Sequence-to-function data** for model training
2. **Evolution/neighborhood data** for evaluating evolvability

The first will be protein sequence-function datasets containing sequences paired to their measured fitness. There exists publicly available datasets of this sort that provide enough examples to train and evaluate the model on sequence-function prediction.

The second will be data where multiple starting sequences are evolved or locally mutated, which will allow direct evaluation of evolvability.

## Method

I will train a model on scalar energy over protein sequences. Given a sequence, I will represent it using an embedding from an existing protein language model. The model will then learn energy values associated with given sequence embeddings where lower energy should correspond to higher function.

Instead of simply predicting the label, EqM learns a landscape. This landscape's behavior can be studied. In particular, the gradient of the scalar energy can be used to identify directions in representation space that reduce energy. The slight hiccup is that I would need to convert the continuous landscape's values into discrete protein sequences.

### Questions of Interest

- Does the model rank or predict measured protein function accurately?
- Does the local neighborhood of the learned energy landscape predict which starting sequences lead to better evolutionary outcomes?

## Proposed Contributions to the Field

### Methodology

I intend to develop and evaluate a simpler scalar formulation of EqM that directly learns an explicit energy function. The goal is to address the open issue in EqM: can a true scalar energy formulation retain useful performance while giving access to a meaningful energy landscape?

### Application

My thesis intends to test whether a learned scalar energy landscape can predict protein evolvability. If I am successful in this, this would provide a computational approach for identifying promising starting sequences for directed evolution. My biological claim would be that the learned energy and its local neighborhood are predictive of measured function and evolvability.

## Timeline

### October

- Refine thesis questions
- Identify datasets for sequence-function prediction
- Identify datasets for evolvability evaluation

### November

- Complete literature review on energy-based models, EqM, protein language models, and sequence-function prediction
- EDA and data cleaning of sequence-function datasets
- Define baseline models and evaluation metrics
- Create an experimental decision tree that outlines what experiments need be run

### December

- Implement Scalar EqM for protein sequence-function data
- Train the baseline models
- Run initial fitness-prediction experiments
- Does scalar energy correlate with measured function?
- Write the Introduction and Methods sections

### January

- Define and test evolvability metrics based on local neighborhoods
- Conduct experiments on nearby mutants
- Prep initial results and thesis outline

### February

- Continue running experiments
- Part 2: Evaluate whether local energy neighborhoods predict evolvability
- Conduct ablations
- Write up Results and Discussion sections
- Be ready to write up full thesis

### March

- Finish experiments
- Write up full thesis draft
- Revisions with Prof. Du
- Figures, tables, mathematical formulations
- Submit!