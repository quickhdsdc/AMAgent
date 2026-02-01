# AM-Agent

This repository contains the code for our paper **"Data-Driven Meets Knowledge-Driven: An LLM‑Agent Framework for Quality Control in Metal Additive Manufacturing"**.

## Abstract

Quality prediction in metal additive manufacturing (AM) has conventionally relied on data-driven models that map process parameters to defect classes or quality metrics. However, these models often fail to generalize across different machines, materials, and process regimes. The emerging knowledge-driven approach based on large language models (LLMs) can interpret literature and expert guidance, yet struggles to deliver quantitative, part-specific decisions tied to specific parameter sets. To bridge this gap, we propose AM-Agent, a modular LLM agent framework that unifies data-driven prediction and domain knowledge through Model Context Protocol tools organized as Knowledge Services (KS) and Data-driven Prediction Services (DPS). KS integrates literature-based static knowledge with dynamic digital twin (DT) context powered by Asset Administration Shell, enabling real-time queries of printer status and historical build records through LLM-DT interaction. DPS exposes melt-pool regressors and defect classifiers using a model-per-condition design, where the AM-Agent adaptively selects the appropriate pretrained predictor at runtime based on current working conditions. To harmonize the outputs of DPS and KS, a reliability-weighted fusion strategy is proposed to resolve conflicts by dynamically weighting numeric uncertainty against semantic confidence. The proposed AM-Agent is intended as a part-specific, pre-build decision support system for human operators during process planning. It is able to select suitable printers, forecasts quality risks, and recommends parameter adjustments. Statistical experiments demonstrate that AM-Agent outperforms purely data-driven or knowledge-driven baselines, especially under domain shift, by correcting overfitted predictions through physical reasoning based on retrieved evidences and the fusion logic. This work highlights the potential of hybridizing data-driven models with domain-informed reasoning, representing a promising new direction for achieving generalizable and interpretable quality control in manufacturing.

## Framework Overview

![Fig. 1](assets/Fig.%201.png)
**Fig. 1.** Proposed concept of integrating data-driven and knowledge-driven approaches for quality control. Both are containerized as callable services for the AM-Agent.

![Fig. 2](assets/Fig.%202.png)
**Fig. 2.** System architecture of the proposed AM-Agent. The application layer details the agent’s components and workflow. It is decoupled from the infrastructure layer via API calls, enabling that models and DTs can be swapped without changing agent logic. Plug icons mark where the components invoke the corresponding infrastructure APIs.

## Project Structure

- **`AgentApp/`**: Contains the core application logic and WebUI.
- **`AgentExperiments/`**: Benchmarking scripts and experimental setups.
- **`AgentFTLLMs/`**: Code related to fine-tuning LLMs for the AM domain.
