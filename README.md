# Factual Retrieval in LLMs Is a Redundant, Distributed and Non-Contiguous Process

This repository contains the code implementation for the paper **"Factual Retrieval in LLMs Is a Redundant, Distributed and Non-Contiguous Process"**, accepted for presentation at **ACL 2026**.

[![arXiv](https://img.shields.io/badge/arXiv-2606.21345-B31B1B.svg)](https://arxiv.org/abs/2606.21345)

## Project Overview

How do Large Language Models (LLMs) retrieve factual knowledge from their parameters? This project investigates the underlying mechanics through the lens of **Attribute-Computation Paths**—the sequence of computational steps an entity representation undergoes to elicit a specific target attribute.  

By introducing a novel iterative activation-patching protocol, our framework extracts minimal, irreducibly necessary subsets of layers required to recall a given fact.

### Key Findings
* **Multiple layers required:** There isn't just one single layer responsible for fetching a fact; the entity must be processed by multiple layers before the information is fully available.
* **Minimal paths are often sparse and non contiguous:** Only a subset of layers participate in each factual recall process, while others may be skipped. 
* **Backup paths exist:** Minimal paths are not unique. For most facts, the model has alternative paths for retrieving the information. These backup paths are usually longer and deeper in the model than the paths used during a normal, uninterrupted run.

## Repository Map

| File / Folder | Description |
| --- | --- |
| `data/` | Sampled CounterFact subsets and the computed computation-path outputs for each supported model. |
| `data/Llama-3.1-8B/` | Llama-3.1-8B sampled data and saved path-search results. |
| `data/Qwen3-8B/` | Qwen3-8B sampled data and saved path-search results. |
| `src/patching.py` | Core patching functions implementing the `lock` and `isolate` operations. |
| `src/search.py` | Iterative algorithm for discovering minimal computation paths. |
| `src/experiments.py` | Additional experimental variants outside the main path search. |
| `scripts/` | Command-line utilities for CounterFact sampling and path-search execution. |
| `scripts/prepare_dataset.py` | Builds the sampled CounterFact subset used in the experiments. |
| `scripts/run_path_search.py` | Runs computation-path extraction on the prepared dataset. |
| `scripts/run_sufficiency_experiments.py` | Runs sufficiency tests for the constructed entity representation. |

## Dataset

Our experiments are conducted using a curated subset of **2,000 samples** from the **CounterFact dataset** (Meng et al., 2022). These specific samples are packaged directly within this codebase for ease of use and reproducibility.

If you use this data or codebase, please make sure to cite the original CounterFact paper:

```bibtex
@article{meng2022locating,
  title={Locating and Editing Factual Associations in {GPT}},
  author={Kevin Meng and David Bau and Alex Andonian and Yonatan Belinkov},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2022},
  note={arXiv:2202.05262}
}
```

## Citation

```bibtex
@inproceedings{hochman-etal-2026-factual,
    title = "Factual Retrieval in {LLM}s Is a Redundant, Distributed and Non-Contiguous Process",
    author = "Hochman, Hail  and
      Shapira, Natalie  and
      Goldberg, Yoav",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.2168/",
    pages = "46747--46768"
}

