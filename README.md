# LLM State Glyph

### A minimal prototype for pre-attentive visualization of language model state

---

## Abstract

Current interfaces to large language models expose internal state through fragmented, explicit metrics (e.g. probabilities, attention maps, token lists). While informative, these representations require analytical effort and do not support rapid, intuitive assessment.

Here we present a minimal prototype for representing model state as a **compact visual glyph**, designed to leverage human ability to interpret continuous, multi-signal visual patterns pre-attentively. The glyph encodes hidden-state structure, attention, uncertainty, and temporal evolution in a single consistent representation.

This repository is not a finalized method, but a **working demonstration of a design principle**: model state should be *readable at a glance*, not only inspected numerically.

---

## Motivation

Human observers are highly efficient at interpreting complex internal states from subtle, continuous cues. This is evident in domains such as face perception, where confidence, attention, and intent are inferred rapidly without explicit measurement.

In contrast, most LLM interfaces rely on:

* discrete metrics (entropy, probabilities)
* separated visualizations (attention maps, token streams)
* or textual summaries

This creates a mismatch:

> the system is continuous and high-dimensional, but the interface is fragmented and analytical.

The central hypothesis of this work is:

> **LLM internal state can be compressed into a small set of continuous visual cues that humans can learn to interpret intuitively.**

---

## Concept

We represent the state of a causal language model at each generation step as a **glyph** composed of several layered elements:

| Component          | Signal                        | Visual encoding           |
| ------------------ | ----------------------------- | ------------------------- |
| Latent state       | Final-layer hidden states     | 2D projection (PCA)       |
| Context weighting  | Attention to previous tokens  | Density field             |
| Temporal evolution | State across generation steps | Trajectory                |
| Uncertainty        | Next-token entropy            | Radial ring               |
| Decision structure | Top-k token probabilities     | Radial distribution glyph |

These elements are not independent plots. They are **co-embedded into a single visual object** intended for holistic perception.

---

## Implementation

This prototype uses a small causal model (default: DistilGPT2) and performs the following steps for a given prompt:

1. Generate tokens sequentially (greedy decoding)
2. At each step, extract:

   * final-layer hidden states
   * attention weights
   * next-token probability distribution
3. Fit a shared PCA projection across all steps for that prompt
4. Render a glyph per step with:

   * attention-weighted density field over projected states
   * trajectory of the current token in latent space
   * entropy-scaled uncertainty ring
   * compact radial representation of top-k probabilities

An interactive interface (via Gradio) allows inspection of individual steps and full trajectories.

---

## Design Principles

This prototype follows a small number of explicit constraints:

* **Compactness**
  All relevant signals are encoded in a single glyph.

* **Continuity**
  Signals are represented as continuous visual variables (position, density, size), not discrete labels.

* **Consistency**
  The same mapping is used across all steps, enabling learning over time.

* **Pre-attentive readability**
  The glyph is intended to be interpretable without conscious parsing of metrics.

* **Minimal decoration**
  Visual elements are tied directly to model-derived quantities.

---

## Limitations

This is an exploratory prototype and has several important limitations:

* **Projection instability**
  PCA is fitted per prompt; glyphs are not directly comparable across runs.

* **Interpretability**
  The mapping from latent space geometry to semantic meaning is not guaranteed.

* **No user validation**
  The claim of improved readability is not empirically tested.

* **Model scale**
  The default model is small; larger models may exhibit clearer structure.

---

## Scope and Intent

This repository is not intended as a finished method or framework. Instead, it serves as:

* a **concrete example** of compressing LLM state into a single visual representation
* a **starting point** for alternative encodings
* a **public disclosure** of the underlying design idea

The specific visual choices are not canonical. The core contribution is the principle:

> *Model state can be exposed as a compact, learnable visual grammar rather than a collection of explicit metrics.*

---

## Usage

```bash
pip install -r requirements.txt
python llm_state_icon.py
```

---

## Outlook

Potential directions include:

* alternative projections (e.g. UMAP, learned embeddings)
* systematic evaluation of interpretability
* comparison with traditional dashboards
* adaptation to larger models and different architectures
* exploration of alternative glyph grammars

---

## License

MIT

## Last edited

2026-05-03
