This repository contains code and processed datasets used for the manuscript (in preparation):

**"Prioritization of Molecular Initiating Event Candidates in Adverse Outcome Pathways through Transcriptomics and Statistical Causal Discovery"**

The study develops a data-driven workflow that integrates transcriptomic concentration-response analysis, benchmark dose (BMD) modeling, weighted gene co-expression network analysis (WGCNA), and LiNGAM-based statistical causal discovery (SCD) to prioritize candidate molecular initiating event (MIE)-related genes.

## Overview

The workflow consists of the following major steps:

1. Preprocessing and filtering of RNA-seq count data using **edgeR**
2. Identification of concentration-responsive genes and estimation of gene-level BMDs using **BMDExpress**
3. Reduction of transcriptomic dimensionality using **WGCNA**
4. Statistical causal discovery among exposure concentration and WGCNA module eigengenes using **DirectLiNGAM**
5. Prioritization of candidate upstream modules based on inferred causal relationships and gene-level BMDs
   * Analyses with edgeR and WGCNA can be done using R, while DirectLiNGAM can be run using Python.

The manuscript evaluates this workflow using two transcriptomic case studies:  
- 17α-ethinylestradiol (EE2) exposure in rainbow trout (Alcaraz et al., 2021, ES&T)
- Pyrene exposure in zebrafish embryos  

## Repository contents

### `Data/htseq_zebra_pyrene.tsv`

Expression data of zebrafish exposed to pyrene

### `RCode_forPyrene.Rmd`

R code used for analysis of the zebrafish pyrene RNA-seq dataset

### `PythonCode_forPyrene_9modules.py`

Python code used for statistical causal discovery applied to zebrafish pyrene dataset


