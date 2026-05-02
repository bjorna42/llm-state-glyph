#!/usr/bin/env python3

"""
Interactive LLM state icon demo.

What this does
--------------
- Loads a small causal language model (default: distilgpt2)
- Lets the user enter a prompt
- Generates a short continuation greedily
- Extracts final-layer hidden states and final-layer mean attention at each step
- Fits a shared PCA across all steps for that prompt
- Renders a simplified, informative-at-a-glance icon for each step
- Exposes the whole thing through a small Gradio UI

Design principles in this version
---------------------------------
- Background field = actual density of projected token states
- Trajectory = path of the current-token state across generation steps
- Current point = current-token state now
- Entropy ring = uncertainty at the current step
- Minimal decoration

Run
---
pip install torch transformers matplotlib scikit-learn gradio imageio pillow
python llm_state_icon_interactive_app.py
"""

from __future__ import annotations

import os
import tempfile
import traceback
from dataclasses import dataclass
from typing import Any

import gradio as gr
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_MODEL = "distilgpt2"
DEFAULT_MAX_NEW_TOKENS = 10
DEFAULT_TOP_K = 24
DEFAULT_ICON_PX = 512
DEFAULT_DPI = 100
DEFAULT_TRAIL_LENGTH = 7
DEFAULT_RANDOM_SEED = 1

# Density settings in normalized icon space
DEFAULT_GRID_SIZE = 300
DEFAULT_DENSITY_SIGMA = 0.16
DEFAULT_DENSITY_ALPHA = 0.42

# Uncertainty ring settings
DEFAULT_RING_RADIUS_MIN = 0.035
DEFAULT_RING_RADIUS_MAX = 0.12

# Rendering colors kept intentionally simple
BG_COLOR = "black"
DENSITY_CMAP = "Blues"


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class StepState:
  step_idx: int
  input_ids: list[int]
  token_texts: list[str]
  hidden_states_last: np.ndarray
  current_attention: np.ndarray
  next_token_probs: np.ndarray
  next_token_logits: np.ndarray
  topk_indices: list[int]
  topk_probs: np.ndarray
  generated_token_id: int | None
  generated_token_text: str | None
  entropy: float


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int) -> None:
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)



def stable_softmax(logits: torch.Tensor) -> torch.Tensor:
  shifted = logits - logits.max()
  exp_vals = torch.exp(shifted)
  return exp_vals / exp_vals.sum()



def tensor_to_numpy(x: torch.Tensor) -> np.ndarray:
  return x.detach().cpu().numpy()



def clean_token_text(text: str) -> str:
  text = text.replace("\n", "\\n").replace("\t", "\\t")
  if text == "":
    return "<EMPTY>"
  if text == " ":
    return "␠"
  return text



def entropy_from_probs(probs: torch.Tensor) -> float:
  p = tensor_to_numpy(probs)
  p = p[p > 0]
  return float(-(p * np.log(p)).sum())



def normalize01(arr: np.ndarray) -> np.ndarray:
  arr = np.asarray(arr, dtype=float)
  mn = arr.min()
  mx = arr.max()
  if np.isclose(mx, mn):
    return np.zeros_like(arr)
  return (arr - mn) / (mx - mn)



def weighted_centroid(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
  w = np.asarray(weights, dtype=float)
  sw = w.sum()
  if np.isclose(sw, 0.0):
    return points.mean(axis=0)
  return (points * w[:, None]).sum(axis=0) / sw



def make_grid(xlim: tuple[float, float], ylim: tuple[float, float], n: int) -> tuple[np.ndarray, np.ndarray]:
  x = np.linspace(xlim[0], xlim[1], n)
  y = np.linspace(ylim[0], ylim[1], n)
  return np.meshgrid(x, y)



def gaussian_field(
  coords: np.ndarray,
  weights: np.ndarray,
  xlim: tuple[float, float],
  ylim: tuple[float, float],
  grid_size: int,
  sigma_frac: float,
) -> np.ndarray:
  xx, yy = make_grid(xlim, ylim, grid_size)
  field = np.zeros_like(xx, dtype=float)

  width = xlim[1] - xlim[0]
  height = ylim[1] - ylim[0]
  sigma = sigma_frac * max(width, height)

  weights = np.asarray(weights, dtype=float)
  if weights.ndim == 0:
    weights = np.full(coords.shape[0], float(weights))

  for (cx, cy), w in zip(coords, weights):
    field += w * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

  return field



def circular_alpha_mask(field: np.ndarray, softness: float = 0.08) -> np.ndarray:
  n_y, n_x = field.shape
  xs = np.linspace(-1, 1, n_x)
  ys = np.linspace(-1, 1, n_y)
  xx, yy = np.meshgrid(xs, ys)
  rr = np.sqrt(xx ** 2 + yy ** 2)

  edge0 = 1.0 - softness
  mask = np.clip((1.0 - rr) / (1.0 - edge0), 0, 1)
  mask[rr <= edge0] = 1.0
  mask[rr >= 1.0] = 0.0
  return field * mask


# -----------------------------------------------------------------------------
# Model wrapper
# -----------------------------------------------------------------------------

class LLMStateIconApp:
  def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
    self.model_name = model_name
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(
      model_name,
      attn_implementation="eager",
    )
    self.model.to(self.device)
    self.model.eval()

  def run_model_step(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    top_k: int,
  ) -> dict[str, Any]:
    with torch.no_grad():
      outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
      )

    if outputs.attentions is None or len(outputs.attentions) == 0:
      raise RuntimeError("No attentions returned. This app expects eager attention.")
    if outputs.hidden_states is None or len(outputs.hidden_states) == 0:
      raise RuntimeError("No hidden states returned.")

    logits = outputs.logits[0, -1, :]
    probs = stable_softmax(logits)
    entropy = entropy_from_probs(probs)

    top_probs, top_idx = torch.topk(probs, k=top_k)

    last_hidden = outputs.hidden_states[-1][0]
    last_attn = outputs.attentions[-1][0]
    mean_attn = last_attn.mean(dim=0)
    current_attention = mean_attn[-1, :]

    return {
      "logits": tensor_to_numpy(logits),
      "probs": tensor_to_numpy(probs),
      "entropy": entropy,
      "top_idx": top_idx.tolist(),
      "top_probs": tensor_to_numpy(top_probs),
      "last_hidden": tensor_to_numpy(last_hidden),
      "current_attention": tensor_to_numpy(current_attention),
    }

  def collect_generation_trace(
    self,
    prompt: str,
    max_new_tokens: int,
    top_k: int,
  ) -> list[StepState]:
    enc = self.tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(self.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
      attention_mask = attention_mask.to(self.device)

    states: list[StepState] = []

    for step_idx in range(max_new_tokens + 1):
      info = self.run_model_step(
        input_ids=input_ids,
        attention_mask=attention_mask,
        top_k=top_k,
      )

      token_ids = input_ids[0].tolist()
      token_texts = [clean_token_text(self.tokenizer.decode([tid])) for tid in token_ids]

      generated_token_id = None
      generated_token_text = None
      if step_idx < max_new_tokens:
        next_id = int(np.argmax(info["probs"]))
        generated_token_id = next_id
        generated_token_text = clean_token_text(self.tokenizer.decode([next_id]))

      states.append(
        StepState(
          step_idx=step_idx,
          input_ids=token_ids,
          token_texts=token_texts,
          hidden_states_last=info["last_hidden"],
          current_attention=info["current_attention"],
          next_token_probs=info["probs"],
          next_token_logits=info["logits"],
          topk_indices=info["top_idx"],
          topk_probs=info["top_probs"],
          generated_token_id=generated_token_id,
          generated_token_text=generated_token_text,
          entropy=info["entropy"],
        )
      )

      if step_idx < max_new_tokens:
        next_token_tensor = torch.tensor([[generated_token_id]], device=self.device)
        input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
        if attention_mask is not None:
          next_mask = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=self.device,
          )
          attention_mask = torch.cat([attention_mask, next_mask], dim=1)

    return states


# -----------------------------------------------------------------------------
# Projection and rendering
# -----------------------------------------------------------------------------


def fit_shared_pca(states: list[StepState], n_components: int = 2) -> PCA:
  all_vectors = np.vstack([s.hidden_states_last for s in states])
  pca = PCA(n_components=n_components)
  pca.fit(all_vectors)
  return pca



def project_states(states: list[StepState], pca: PCA) -> list[np.ndarray]:
  return [pca.transform(s.hidden_states_last) for s in states]



def draw_distribution_glyph(ax, top_probs: np.ndarray) -> None:
  probs = np.asarray(top_probs, dtype=float)
  if probs.sum() > 0:
    probs = probs / probs.sum()

  gx, gy = 0.83, 0.18
  r = 0.12
  outer = Circle((gx, gy), r, transform=ax.transAxes, fill=False,
                 linewidth=1.0, edgecolor=(1, 1, 1, 0.20))
  ax.add_patch(outer)

  inner = 0.20 * r
  theta = np.linspace(0, 2 * np.pi, len(probs), endpoint=False)
  maxp = max(probs.max(), 1e-8)

  for ang, p in zip(theta, probs):
    rr = inner + (r - inner) * (p / maxp)
    x0 = gx + inner * np.cos(ang)
    y0 = gy + inner * np.sin(ang)
    x1 = gx + rr * np.cos(ang)
    y1 = gy + rr * np.sin(ang)
    ax.plot(
      [x0, x1], [y0, y1],
      transform=ax.transAxes,
      color=(1, 1, 1, 0.25 + 0.40 * (p / maxp)),
      linewidth=1.0,
    )



def render_informative_icon(
  state: StepState,
  coords: np.ndarray,
  frontier_history: list[np.ndarray],
  xlim: tuple[float, float],
  ylim: tuple[float, float],
  entropy_min: float,
  entropy_max: float,
  outpath: str,
  icon_px: int,
  dpi: int,
  trail_length: int,
  grid_size: int,
  density_sigma: float,
  density_alpha: float,
  ring_radius_min: float,
  ring_radius_max: float,
) -> None:
  figsize = icon_px / dpi
  fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
  ax = fig.add_axes([0, 0, 1, 1])

  ax.set_xticks([])
  ax.set_yticks([])
  for spine in ax.spines.values():
    spine.set_visible(False)
  fig.patch.set_facecolor(BG_COLOR)
  ax.set_facecolor(BG_COLOR)

  # Normalize projected coordinates to icon space [-1, 1]
  x_center = (xlim[0] + xlim[1]) / 2
  y_center = (ylim[0] + ylim[1]) / 2
  scale = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) / 2

  coords_icon = np.column_stack([
    (coords[:, 0] - x_center) / scale,
    (coords[:, 1] - y_center) / scale,
  ])

  history_icon = np.array([
    [(pt[0] - x_center) / scale, (pt[1] - y_center) / scale]
    for pt in frontier_history
  ])

  att = np.asarray(state.current_attention, dtype=float)
  att_norm = normalize01(att)
  density_weights = 0.25 + 0.75 * att_norm

  field = gaussian_field(
    coords=coords_icon,
    weights=density_weights,
    xlim=(-1, 1),
    ylim=(-1, 1),
    grid_size=grid_size,
    sigma_frac=density_sigma,
  )
  field = circular_alpha_mask(field, softness=0.08)

  boundary = Circle((0, 0), 0.96, fill=False, linewidth=1.2,
                    edgecolor=(1, 1, 1, 0.16))
  ax.add_patch(boundary)

  ax.imshow(
    field,
    extent=[-1, 1, -1, 1],
    origin="lower",
    interpolation="bilinear",
    cmap=DENSITY_CMAP,
    alpha=density_alpha,
  )

  # Trajectory of the current-token state
  if len(history_icon) > 1:
    trail = history_icon[-trail_length:]
    nseg = len(trail) - 1
    for i in range(nseg):
      frac = i / max(nseg - 1, 1)
      alpha = 0.15 + 0.65 * frac
      lw = 1.2 + 4.2 * frac
      ax.plot(
        trail[i:i + 2, 0],
        trail[i:i + 2, 1],
        color=(1, 1, 1, alpha),
        linewidth=lw,
        solid_capstyle="round",
      )

  # Current point and previous -> current arrow-like final segment
  cur = coords_icon[-1]
  prev = history_icon[-2] if len(history_icon) > 1 else cur
  ax.plot([prev[0], cur[0]], [prev[1], cur[1]], color=(1, 1, 1, 0.75), linewidth=4.8,
          solid_capstyle="round")
  ax.scatter([cur[0]], [cur[1]], s=70, c=[(1, 1, 1, 0.98)], linewidths=0)

  # Attention-weighted centroid as a faint anchor
  centroid = weighted_centroid(coords_icon, density_weights)
  ax.scatter([centroid[0]], [centroid[1]], s=28, c=[(1, 1, 1, 0.30)], linewidths=0)
  ax.plot([centroid[0], cur[0]], [centroid[1], cur[1]], color=(1, 1, 1, 0.14), linewidth=1.4)

  # Local uncertainty ring around current point
  if np.isclose(entropy_max, entropy_min):
    entropy_scaled = 0.5
  else:
    entropy_scaled = (state.entropy - entropy_min) / (entropy_max - entropy_min)
  ring_radius = ring_radius_min + (ring_radius_max - ring_radius_min) * entropy_scaled
  uncertainty_ring = Circle(
    (cur[0], cur[1]),
    ring_radius,
    fill=False,
    linewidth=1.3,
    edgecolor=(1, 1, 1, 0.28),
  )
  ax.add_patch(uncertainty_ring)

  draw_distribution_glyph(ax, state.topk_probs)

  ax.set_xlim(-1, 1)
  ax.set_ylim(-1, 1)
  ax.set_aspect("equal")

  plt.savefig(outpath, dpi=dpi, facecolor=fig.get_facecolor())
  plt.close(fig)


# -----------------------------------------------------------------------------
# Interactive app logic
# -----------------------------------------------------------------------------

APP = LLMStateIconApp()



def run_demo(
  prompt: str,
  max_new_tokens: int,
  top_k: int,
  step_to_show: int,
  icon_px: int,
  trail_length: int,
  density_sigma: float,
) -> tuple[str, str, str, str]:
  prompt = prompt.strip()
  if not prompt:
    raise gr.Error("Please enter a prompt.")

  set_seed(DEFAULT_RANDOM_SEED)

  states = APP.collect_generation_trace(
    prompt=prompt,
    max_new_tokens=max_new_tokens,
    top_k=top_k,
  )

  pca = fit_shared_pca(states)
  projected = project_states(states, pca)

  all_xy = np.vstack(projected)
  xmin, ymin = all_xy.min(axis=0)
  xmax, ymax = all_xy.max(axis=0)
  xr = xmax - xmin
  yr = ymax - ymin
  pad_x = 0.10 * xr if xr > 0 else 1.0
  pad_y = 0.10 * yr if yr > 0 else 1.0
  xlim = (xmin - pad_x, xmax + pad_x)
  ylim = (ymin - pad_y, ymax + pad_y)

  entropies = [s.entropy for s in states]
  entropy_min = min(entropies)
  entropy_max = max(entropies)

  step_idx = min(max(step_to_show, 0), len(states) - 1)
  frontier_history = [coords[-1].copy() for coords in projected[: step_idx + 1]]

  tmpdir = tempfile.mkdtemp(prefix="llm_state_icon_")
  png_path = os.path.join(tmpdir, "current_icon.png")
  gif_path = os.path.join(tmpdir, "animation.gif")

  render_informative_icon(
    state=states[step_idx],
    coords=projected[step_idx],
    frontier_history=frontier_history,
    xlim=xlim,
    ylim=ylim,
    entropy_min=entropy_min,
    entropy_max=entropy_max,
    outpath=png_path,
    icon_px=icon_px,
    dpi=DEFAULT_DPI,
    trail_length=trail_length,
    grid_size=DEFAULT_GRID_SIZE,
    density_sigma=density_sigma,
    density_alpha=DEFAULT_DENSITY_ALPHA,
    ring_radius_min=DEFAULT_RING_RADIUS_MIN,
    ring_radius_max=DEFAULT_RING_RADIUS_MAX,
  )

  # Also render a full animation for the current prompt.
  frame_paths: list[str] = []
  frontier_hist_all: list[np.ndarray] = []
  for i, (state, coords) in enumerate(zip(states, projected)):
    frontier_hist_all.append(coords[-1].copy())
    frame_path = os.path.join(tmpdir, f"frame_{i:03d}.png")
    render_informative_icon(
      state=state,
      coords=coords,
      frontier_history=frontier_hist_all,
      xlim=xlim,
      ylim=ylim,
      entropy_min=entropy_min,
      entropy_max=entropy_max,
      outpath=frame_path,
      icon_px=icon_px,
      dpi=DEFAULT_DPI,
      trail_length=trail_length,
      grid_size=DEFAULT_GRID_SIZE,
      density_sigma=density_sigma,
      density_alpha=DEFAULT_DENSITY_ALPHA,
      ring_radius_min=DEFAULT_RING_RADIUS_MIN,
      ring_radius_max=DEFAULT_RING_RADIUS_MAX,
    )
    frame_paths.append(frame_path)

  images = [imageio.imread(fp) for fp in frame_paths]
  imageio.mimsave(gif_path, images, duration=0.85)

  final_text = "".join(states[-1].token_texts)
  step_summary = (
    f"Step {step_idx} / {len(states) - 1}\n"
    f"Current context: {''.join(states[step_idx].token_texts)}\n"
    f"Next token from this step: {states[step_idx].generated_token_text!r}\n"
    f"Entropy: {states[step_idx].entropy:.4f}"
  )
  token_summary = "Tokens:\n" + " | ".join(states[step_idx].token_texts)

  return png_path, gif_path, final_text, step_summary + "\n\n" + token_summary


# -----------------------------------------------------------------------------
# Gradio UI
# -----------------------------------------------------------------------------

with gr.Blocks(title="Interactive LLM State Icon") as demo:
  gr.Markdown(
    "# Interactive LLM State Icon\n"
    "Type a prompt, choose a generation step, and inspect a simplified but still informative state icon."
  )

  with gr.Row():
    with gr.Column(scale=1):
      prompt_box = gr.Textbox(
        label="Prompt",
        lines=4,
        value="In Sweden, the capital is",
      )
      max_new_tokens_slider = gr.Slider(
        label="Max new tokens",
        minimum=1,
        maximum=20,
        step=1,
        value=DEFAULT_MAX_NEW_TOKENS,
      )
      top_k_slider = gr.Slider(
        label="Top-k for distribution glyph",
        minimum=4,
        maximum=64,
        step=1,
        value=DEFAULT_TOP_K,
      )
      step_slider = gr.Slider(
        label="Step to show",
        minimum=0,
        maximum=20,
        step=1,
        value=DEFAULT_MAX_NEW_TOKENS,
      )
      icon_px_slider = gr.Slider(
        label="Icon size (px)",
        minimum=256,
        maximum=768,
        step=64,
        value=DEFAULT_ICON_PX,
      )
      trail_slider = gr.Slider(
        label="Trail length",
        minimum=2,
        maximum=12,
        step=1,
        value=DEFAULT_TRAIL_LENGTH,
      )
      sigma_slider = gr.Slider(
        label="Density sigma",
        minimum=0.06,
        maximum=0.30,
        step=0.01,
        value=DEFAULT_DENSITY_SIGMA,
      )
      run_button = gr.Button("Render")

    with gr.Column(scale=1):
      icon_output = gr.Image(label="Current icon", type="filepath")
      gif_output = gr.Image(label="Animation", type="filepath")
      final_text_output = gr.Textbox(label="Generated continuation", lines=4)
      debug_text_output = gr.Textbox(label="Step summary", lines=10)

  def sync_step_max(max_new_tokens: int) -> gr.Slider:
    return gr.Slider(maximum=max_new_tokens, value=max_new_tokens)

  max_new_tokens_slider.change(
    fn=sync_step_max,
    inputs=max_new_tokens_slider,
    outputs=step_slider,
  )

  run_button.click(
    fn=run_demo,
    inputs=[
      prompt_box,
      max_new_tokens_slider,
      top_k_slider,
      step_slider,
      icon_px_slider,
      trail_slider,
      sigma_slider,
    ],
    outputs=[icon_output, gif_output, final_text_output, debug_text_output],
  )


if __name__ == "__main__":
  try:
    demo.launch()
  except Exception:
    traceback.print_exc()
    raise
