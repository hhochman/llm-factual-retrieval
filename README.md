# Factual Retrieval in LLMs Is a Redundant, Distributed and Non-Contiguous Process

This repository contains the code implementation for the paper **"Factual Retrieval in LLMs Is a Redundant, Distributed and Non-Contiguous Process"**, accepted for presentation at **ACL 2026**.

> **Repository Under Construction** > We are currently cleaning our codebase and uploading it stage-by-stage. Stay tuned for updates!  

---

## Project Overview

How do Large Language Models (LLMs) retrieve factual knowledge from their parameters? This project investigates the underlying mechanics through the lens of **Attribute-Computation Paths**—the sequence of computational steps an entity representation undergoes to elicit a specific target attribute.  

By introducing a novel iterative activation-patching protocol, our framework extracts minimal, irreducibly necessary subsets of layers required to recall a given fact.

### Key Findings
* **Multiple layers required:** There isn't just one single layer responsible for fetching a fact; the entity must be processed by multiple layers before the information is fully available.
* **Minimal paths are often sparse and non contiguous:** Only a subset of layers participate in each factual recall process, while others may be skipped. 
* **Backup paths exist:** Minimal paths are not unique. For most facts, the model has alternative paths for retrieving the information. These backup paths are usually longer and deeper in the model than the paths used during a normal, uninterrupted run.
