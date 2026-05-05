# LLM State Glyph

### A minimal prototype for pre-attentive visualization of language model state

---

## Abstract

Current interfaces to large language models expose internal state through fragmented, explicit metrics (e.g. probabilities, attention maps, token lists). While informative, these representations require analytical effort and do not support rapid, intuitive assessment.

Here we present a minimal prototype for representing model state as a **compact visual glyph**, designed to leverage human ability to interpret continuous, multi-signal visual patterns pre-attentively. The glyph encodes hidden-state structure, attention, uncertainty, and temporal evolution in a single consistent representation.

In addition, we include a **contrastive prototype** that maps the same underlying signals onto a simple biomorphic structure. This serves as a *cautionary demonstration* of how identical model-derived quantities can be interpreted very differently depending on representation.

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

## Contrastive Prototype: Biomorphic Mapping

In addition to the abstract glyph, this repository includes an experimental prototype that maps the same model-derived signals onto a simple **tardigrade-like biomorph**.

Example mappings:

| Signal      | Biomorphic encoding            |
| ----------- | ------------------------------ |
| Entropy     | Leg twitch / movement speed    |
| Uncertainty | Contraction toward “tun” state |
| Confidence  | Posture extension / openness   |
| Load/Stress | Vibration amplitude            |
| Attention   | Proboscis extension            |
| Stability   | Rotation smoothness vs wobble  |

Importantly:

> **No additional information is introduced. Only the representation changes.**

---

## Cautionary Observation

A key outcome of this prototype is that:

> **minimal structural and dynamical cues are sufficient to induce perception of agency**

Even without:

* eyes
* facial features
* explicit emotional signals

observers tend to interpret the biomorph in terms of:

* hesitation
* confidence
* attention
* intent

This occurs despite the fact that all behavior is derived from simple scalar signals (e.g. entropy, logits, attention weights).

---

## Interpretation Drift

The abstract glyph and biomorphic representation encode the same data but produce qualitatively different interpretations:

| Representation | Typical interpretation  |
| -------------- | ----------------------- |
| Glyph          | Analytical, statistical |
| Biomorph       | Intentional, agent-like |

This demonstrates a broader point:

> **the interface determines the narrative, not just the data**

---

## Implementation

This prototype uses a small causal model (default: DistilGPT2) and performs the following steps for a given prompt:

1. Generate tokens sequentially (greedy decoding)
2. At each step, extract:

   * final-layer hidden states
   * attention weights
   * next-token probability distribution
3. Fit a shared PCA projection across all steps for that prompt
4. Derive scalar signals:

   * entropy (uncertainty proxy)
   * logit margin (confidence proxy)
   * attention concentration
   * latent trajectory stability
5. Render:

   * an abstract glyph
   * a biomorphic mapping of the same signals

An interactive interface (via Gradio) allows inspection of both representations.

---

## Design Principles

This prototype follows a small number of explicit constraints:

* **Compactness**
  All relevant signals are encoded in a single glyph.

* **Continuity**
  Signals are represented as continuous visual variables (position, density, size), not discrete labels.

* **Consistency**
  The same mapping is used across all steps.

* **Pre-attentive readability**
  The glyph is intended to be interpretable without explicit metric parsing.

* **Minimal decoration**
  Visual elements are tied directly to model-derived quantities.

* **Controlled anthropomorphism (biomorphic prototype)**
  The biomorph is deliberately minimal and avoids explicit features (e.g. eyes), while still exposing perceptual biases.

---

## Limitations

This is an exploratory prototype and has several important limitations:

* **Projection instability**
  PCA is fitted per prompt; glyphs are not directly comparable across runs.

* **Interpretability**
  The mapping from latent space geometry to semantic meaning is not guaranteed.

* **No user validation**
  Claims about readability and interpretation are not empirically tested.

* **Model scale**
  The default model is small; larger models may exhibit clearer structure.

* **Behavioral ambiguity**
  The biomorphic representation may induce interpretations not grounded in model internals.

---

## Scope and Intent

This repository is not intended as a finished method or framework. Instead, it serves as:

* a **concrete example** of compressing LLM state into a single visual representation
* a **contrastive demonstration** of how representation shapes interpretation
* a **public disclosure** of both the design idea and its potential pitfalls

The core contribution is the principle:

> *Model state can be exposed as a compact, learnable visual grammar — but representation choices strongly influence how that state is perceived.*

---

## Usage

```bash
pip install -r requirements.txt
python llm_biomorph_app.py
```

---

## Outlook

Potential directions include:

* alternative projections (e.g. UMAP, learned embeddings)
* systematic evaluation of interpretability
* comparison with traditional dashboards
* controlled studies of perception and misinterpretation
* exploration of alternative glyph grammars
* formal analysis of representation-induced bias

---

## License

MIT

## Last edited

2026-05-05
