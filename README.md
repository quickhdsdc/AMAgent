# AM-Agent

This repository contains the code for our paper **"Data-Driven Meets Knowledge-Driven: An LLM‑Agent Framework for Quality Control in Metal Additive Manufacturing"**.

Paper: [Data-Driven Meets Knowledge-Driven: An LLM-Agent Framework for Quality Control in Metal Additive Manufacturing](https://www.sciencedirect.com/science/article/pii/S2452414X2600124X).

## Abstract

Quality prediction in metal additive manufacturing (AM) has conventionally relied on data-driven models that map process parameters to defect classes or quality metrics. However, these models often fail to generalize across different machines, materials, and process regimes. The emerging knowledge-driven approach based on large language models (LLMs) can interpret literature and expert guidance, yet struggles to deliver quantitative decisions tied to part-specific parameter sets. To bridge this gap, we propose AM-Agent, a neuro-symbolic LLM-agent framework that unifies data-driven and knowledge-driven quality control to enhance adaptivity in part-specific pre-build process planning for Laser Powder Bed Fusion (LPBF). AM-Agent treats both paradigms as independently callable services over a shared defect label space. A Data-driven Prediction Service (DPS) exposes melt-pool regressors and defect classifiers under a model-per-material design, in which the supervising LLM adaptively invokes the predictor matched to the current powder material. A Knowledge Service (KS) grounds its reasoning in process literature via retrieval-augmented generation and in dynamic digital-twin context retrieved from the Asset Administration Shell (AAS). To harmonize the two paradigms’ competing predictions, we formalize the fusion as a Linear Opinion Pool (LOP) with source-intrinsic reliability weights. The DPS weight is an entropy-and-margin calibration proxy modulated by a distribution-shift indicator, and the KS weight is the LLM’s evidence-grounded self-reported reliability. In-domain and out-of-domain experiments on the LPBF benchmark show that AM-Agent preserves DPS performance in-domain and recovers macro-F1 under cross-material distribution shifts. Conflict analysis further demonstrates how the LOP recovers from data-driven errors by deferring to retrieved physics-grounded evidence when the DPS flags uncertainty.

## Framework Overview

![Fig. 1](assets/Fig.%201.png)
**Fig. 1.** Proposed concept of integrating data-driven and knowledge-driven approaches for quality control. Both are containerized as callable services for the AM-Agent.

![Fig. 2](assets/Fig.%202.png)
**Fig. 2.** System architecture of the proposed AM-Agent. The application layer details the agent’s components and workflow. It is decoupled from the infrastructure layer via API calls, enabling that models and DTs can be swapped without changing agent logic. Plug icons mark where the components invoke the corresponding infrastructure APIs.

## Project Structure

- **`AgentApp/`**: Contains the core application logic and WebUI.
- **`AgentExperiments/`**: Benchmarking scripts and experimental setups.
- **`AgentFTLLMs/`**: Code related to fine-tuning LLMs for the AM domain.

The experimental train/test splits and generated result files are not distributed in this repository. The prompt and runner corrections in this release have not been used to regenerate the paper's reported metrics.
