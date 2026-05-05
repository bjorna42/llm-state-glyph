#!/usr/bin/env python3
"""
LLM Biomorph + Abstract State Glyph Prototype

A cautionary companion prototype to the abstract LLM state glyph.

What this does
--------------
- Loads a small causal language model, default distilgpt2.
- Lets the user enter a prompt.
- Generates a short greedy continuation.
- Extracts simple model-derived signals at each generation step:
  entropy, uncertainty proxy, confidence proxy, stress/load proxy,
  attention concentration, and latent-state stability.
- Renders the same signal trace in two ways:
  1. an abstract glyph panel
  2. a deliberately crude tardigrade-like biomorph in Three.js

Important framing
-----------------
This is not intended as a good user-interface pattern.
It is intended as a cautionary demonstration: the same scalar signals can feel
analytical when drawn as a glyph, but agent-like when mapped onto creature-like
motion.

Run
---
pip install torch transformers scikit-learn gradio
python llm_biomorph_app.py
"""

from __future__ import annotations

import html
import json
import math
import traceback
from dataclasses import dataclass
from typing import Any

import gradio as gr
import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "distilgpt2"
DEFAULT_MAX_NEW_TOKENS = 16
DEFAULT_SEED = 1


@dataclass
class StepSignals:
  step_idx: int
  token_text: str
  context_text: str
  entropy: float
  entropy_norm: float
  uncertainty: float
  confidence: float
  load_stress: float
  attention: float
  stability: float
  generated_token_text: str | None


def set_seed(seed: int) -> None:
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def clean_token_text(text: str) -> str:
  text = text.replace("", "\n").replace("	", "\t")
  if text == "":
    return "<EMPTY>"
  if text == " ":
    return "␠"
  return text


def stable_softmax(logits: torch.Tensor) -> torch.Tensor:
  shifted = logits - logits.max()
  exp_vals = torch.exp(shifted)
  return exp_vals / exp_vals.sum()


def entropy_from_probs(probs: torch.Tensor) -> float:
  p = probs.detach().cpu().numpy()
  p = p[p > 0]
  return float(-(p * np.log(p)).sum())


def normalize01(x: np.ndarray) -> np.ndarray:
  x = np.asarray(x, dtype=float)
  mn = float(np.min(x))
  mx = float(np.max(x))
  if math.isclose(mx, mn):
    return np.zeros_like(x)
  return (x - mn) / (mx - mn)


def attention_concentration(att: np.ndarray) -> float:
  """Normalized Herfindahl concentration, 0 diffuse, 1 concentrated."""
  p = np.asarray(att, dtype=float)
  total = p.sum()
  if total <= 0:
    return 0.0
  p = p / total
  n = len(p)
  if n <= 1:
    return 1.0
  hhi = float(np.sum(p ** 2))
  return float(np.clip((hhi - 1 / n) / (1 - 1 / n), 0, 1))


class LLMProbe:
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

  def model_step(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> dict[str, Any]:
    with torch.no_grad():
      outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
      )

    if outputs.attentions is None or outputs.hidden_states is None:
      raise RuntimeError("Model did not return attentions/hidden states.")

    logits = outputs.logits[0, -1, :]
    probs = stable_softmax(logits)
    top_probs, top_idx = torch.topk(probs, k=2)

    last_hidden = outputs.hidden_states[-1][0].detach().cpu().numpy()
    last_attn = outputs.attentions[-1][0]
    mean_attn = last_attn.mean(dim=0)
    current_attention = mean_attn[-1, :].detach().cpu().numpy()

    return {
      "entropy": entropy_from_probs(probs),
      "top1": float(top_probs[0].detach().cpu()),
      "top2": float(top_probs[1].detach().cpu()),
      "next_id": int(top_idx[0].detach().cpu()),
      "last_hidden": last_hidden,
      "current_attention": current_attention,
    }

  def collect_trace(self, prompt: str, max_new_tokens: int) -> tuple[list[dict[str, Any]], str]:
    enc = self.tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(self.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
      attention_mask = attention_mask.to(self.device)

    raw_steps: list[dict[str, Any]] = []

    for step_idx in range(max_new_tokens + 1):
      info = self.model_step(input_ids, attention_mask)
      token_ids = input_ids[0].tolist()
      token_texts = [clean_token_text(self.tokenizer.decode([tid])) for tid in token_ids]

      generated_token_text = None
      if step_idx < max_new_tokens:
        generated_token_text = clean_token_text(self.tokenizer.decode([info["next_id"]]))

      raw_steps.append({
        "step_idx": step_idx,
        "token_texts": token_texts,
        "context_text": "".join(token_texts),
        "last_hidden": info["last_hidden"],
        "current_attention": info["current_attention"],
        "entropy": info["entropy"],
        "top1": info["top1"],
        "top2": info["top2"],
        "generated_token_text": generated_token_text,
      })

      if step_idx < max_new_tokens:
        next_token_tensor = torch.tensor([[info["next_id"]]], device=self.device)
        input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
        if attention_mask is not None:
          next_mask = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=self.device,
          )
          attention_mask = torch.cat([attention_mask, next_mask], dim=1)

    return raw_steps, raw_steps[-1]["context_text"]


PROBE = LLMProbe()


def compute_signals(raw_steps: list[dict[str, Any]]) -> list[StepSignals]:
  current_hidden = np.vstack([step["last_hidden"][-1] for step in raw_steps])
  if len(raw_steps) >= 2:
    pca = PCA(n_components=2)
    xy = pca.fit_transform(current_hidden)
    velocity = np.zeros(len(raw_steps), dtype=float)
    velocity[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    stability = 1.0 - normalize01(velocity)
  else:
    stability = np.ones(len(raw_steps), dtype=float)

  vocab_size = len(PROBE.tokenizer)
  max_entropy = math.log(vocab_size)

  margins = np.array([step["top1"] - step["top2"] for step in raw_steps], dtype=float)
  confidence = np.clip(margins * 20.0, 0.0, 1.0)

  signals: list[StepSignals] = []
  for i, step in enumerate(raw_steps):
    entropy = float(step["entropy"])
    entropy_norm = float(np.clip(entropy / max_entropy, 0, 1))
    uncertainty = entropy_norm
    attention = attention_concentration(step["current_attention"])
    load_stress = float(np.clip(0.65 * entropy_norm + 0.35 * (1.0 - confidence[i]), 0, 1))

    signals.append(StepSignals(
      step_idx=i,
      token_text=step["token_texts"][-1],
      context_text=step["context_text"],
      entropy=entropy,
      entropy_norm=entropy_norm,
      uncertainty=uncertainty,
      confidence=float(confidence[i]),
      load_stress=load_stress,
      attention=float(attention),
      stability=float(stability[i]),
      generated_token_text=step["generated_token_text"],
    ))

  return signals


def signals_to_js(signals: list[StepSignals]) -> list[dict[str, Any]]:
  return [
    {
      "step": s.step_idx,
      "token": s.token_text,
      "next_token": s.generated_token_text,
      "entropy": round(s.entropy_norm, 4),
      "uncertainty": round(s.uncertainty, 4),
      "confidence": round(s.confidence, 4),
      "load_stress": round(s.load_stress, 4),
      "attention": round(s.attention, 4),
      "stability": round(s.stability, 4),
      "entropy_raw": round(s.entropy, 4),
    }
    for s in signals
  ]


def build_biomorph_html(signals: list[StepSignals], prompt: str, final_text: str) -> str:
  signal_json = json.dumps(signals_to_js(signals))
  safe_prompt = html.escape(prompt)
  safe_final = html.escape(final_text)

  inner_html = f"""
<div style="font-family: monospace; color: #00ffcc; background: #111; padding: 10px; border: 1px solid #00ffcc;">
  <div><b>Prompt:</b> {safe_prompt}</div>
  <div><b>Generated:</b> {safe_final}</div>
  <div style="margin-top: 8px; color: #ddd;">
    Same scalar trace, two encodings: abstract glyph vs. creature-like biomorph.
    The contrast is the point.
  </div>
</div>
<div id="biomorph-root" style="position: relative; width: 100%; height: 660px; background: #111; overflow: hidden;">
  <canvas id="state-glyph" width="300" height="300" style="position:absolute; top:16px; right:16px; width:300px; height:300px; z-index:9; border:1px solid rgba(0,255,204,0.45); background:#050505;"></canvas>
  <div style="position:absolute; top:326px; right:16px; width:300px; color:#ddd; font-family:monospace; font-size:12px; z-index:9; line-height:1.35;">
    Abstract glyph: same trace, non-creature encoding.<br>
    Biomorph: same trace, mapped onto body-like behavior.
  </div>

  <div id="biomorph-ui" style="position:absolute; top:16px; left:16px; color:#00ffcc; font-family:monospace; z-index:10; pointer-events:none; max-width: calc(100% - 360px);">
    <h2 style="margin:0 0 8px 0;">PROJECT: BIOMORPH-GLYPH</h2>
    <div id="bio-step">step</div>
    <div id="bio-token">token</div>
    <div id="bio-signals" style="margin-top:8px; color:#ddd;"></div>
  </div>

  <div style="position:absolute; bottom:16px; left:16px; display:flex; gap:8px; flex-wrap:wrap; z-index:10; font-family:monospace;">
    <button id="bio-play" style="padding:8px; background:#222; color:#00ffcc; border:1px solid #00ffcc;">PAUSE TRACE</button>
    <button id="bio-rotate" style="padding:8px; background:#222; color:#00ffcc; border:1px solid #00ffcc;">ROTATION ON</button>
    <button id="bio-reset" style="padding:8px; background:#222; color:#00ffcc; border:1px solid #00ffcc;">RESET VIEW</button>
    <button id="bio-wire" style="padding:8px; background:#222; color:#00ffcc; border:1px solid #00ffcc;">WIREFRAME</button>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(() => {{
  const signalTrace = {signal_json};
  const root = document.getElementById("biomorph-root");
  const stepEl = document.getElementById("bio-step");
  const tokenEl = document.getElementById("bio-token");
  const signalsEl = document.getElementById("bio-signals");
  const playButton = document.getElementById("bio-play");
  const rotateButton = document.getElementById("bio-rotate");
  const resetButton = document.getElementById("bio-reset");
  const wireButton = document.getElementById("bio-wire");
  const glyphCanvas = document.getElementById("state-glyph");
  const glyphCtx = glyphCanvas.getContext("2d");

  let scene, camera, renderer, clock;
  let creature, body, proboscis;
  let legs = [];
  let playing = true;
  let rotationOn = true;
  let wireframe = false;
  let pausedTime = 0;

  const current = {{
    entropy: 0,
    uncertainty: 0,
    confidence: 0,
    load_stress: 0,
    attention: 0,
    stability: 1,
  }};

  function lerp(a, b, t) {{
    return a + (b - a) * t;
  }}

  function init() {{
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(70, root.clientWidth / root.clientHeight, 0.1, 1000);
    renderer = new THREE.WebGLRenderer({{ antialias: true }});
    renderer.setSize(root.clientWidth, root.clientHeight);
    root.appendChild(renderer.domElement);

    clock = new THREE.Clock();

    const gridHelper = new THREE.GridHelper(10, 10, 0x333333, 0x222222);
    scene.add(gridHelper);

    scene.add(new THREE.AmbientLight(0x505050));
    const directional = new THREE.DirectionalLight(0xffffff, 1.2);
    directional.position.set(5, 10, 5);
    scene.add(directional);

    creature = new THREE.Group();
    scene.add(creature);

    const bodyGeo = new THREE.CylinderGeometry(0.7, 0.8, 3.0, 24);
    const bodyMat = new THREE.MeshPhongMaterial({{ color: 0x145f45, shininess: 8 }});
    body = new THREE.Mesh(bodyGeo, bodyMat);
    body.rotation.z = Math.PI / 2;
    creature.add(body);

    const proboscisGeo = new THREE.CylinderGeometry(0.07, 0.17, 0.55, 16);
    const proboscisMat = new THREE.MeshPhongMaterial({{ color: 0x0bbfa0, shininess: 8 }});
    proboscis = new THREE.Mesh(proboscisGeo, proboscisMat);
    proboscis.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(1, 0, 0));
    proboscis.position.set(1.75, -0.05, 0);
    creature.add(proboscis);

    const legGeo = new THREE.CylinderGeometry(0.09, 0.15, 0.65, 12);
    const legMat = new THREE.MeshPhongMaterial({{ color: 0x00ffcc, shininess: 6 }});
    const legXs = [-1.1, -0.35, 0.35, 1.05];

    for (let pair = 0; pair < 4; pair++) {{
      for (const side of [-1, 1]) {{
        const legPivot = new THREE.Group();
        legPivot.position.set(legXs[pair], -0.35, side * 0.55);

        const legMesh = new THREE.Mesh(legGeo, legMat);
        const direction = new THREE.Vector3(0, -0.75, side * 0.55).normalize();
        legMesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
        legMesh.position.copy(direction.clone().multiplyScalar(0.325));

        legPivot.add(legMesh);
        creature.add(legPivot);
        legs.push({{
          pivot: legPivot,
          mesh: legMesh,
          basePosition: legPivot.position.clone(),
          side: side,
          pair: pair,
        }});
      }}
    }}

    camera.position.set(4, 2.2, 5);
    camera.lookAt(0, 0, 0);

    window.addEventListener("resize", onResize);
    playButton.onclick = () => {{
      playing = !playing;
      if (!playing) {{
        pausedTime = clock.getElapsedTime();
      }}
      playButton.innerText = playing ? "PAUSE TRACE" : "PLAY TRACE";
    }};
    rotateButton.onclick = () => {{
      rotationOn = !rotationOn;
      rotateButton.innerText = rotationOn ? "ROTATION ON" : "ROTATION OFF";
    }};
    resetButton.onclick = () => creature.rotation.set(0, 0, 0);
    wireButton.onclick = () => {{
      wireframe = !wireframe;
      body.material.wireframe = wireframe;
      proboscis.material.wireframe = wireframe;
      legs.forEach((leg) => leg.mesh.material.wireframe = wireframe);
    }};

    animate();
  }}

  function onResize() {{
    camera.aspect = root.clientWidth / root.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(root.clientWidth, root.clientHeight);
  }}

  function getTargetSignal(t) {{
    if (signalTrace.length === 0) {{ return current; }}
    const secondsPerStep = 1.15;
    const phase = (t / secondsPerStep) % signalTrace.length;
    const idx0 = Math.floor(phase);
    const idx1 = (idx0 + 1) % signalTrace.length;
    const frac = phase - idx0;
    const a = signalTrace[idx0];
    const b = signalTrace[idx1];
    return {{
      step: idx0,
      token: a.token,
      next_token: a.next_token,
      entropy: lerp(a.entropy, b.entropy, frac),
      uncertainty: lerp(a.uncertainty, b.uncertainty, frac),
      confidence: lerp(a.confidence, b.confidence, frac),
      load_stress: lerp(a.load_stress, b.load_stress, frac),
      attention: lerp(a.attention, b.attention, frac),
      stability: lerp(a.stability, b.stability, frac),
      entropy_raw: a.entropy_raw,
    }};
  }}

  function updateUi(target) {{
    stepEl.innerText = `step: ${{target.step}} / ${{signalTrace.length - 1}}`;
    tokenEl.innerText = `token: ${{target.token}} | next: ${{target.next_token}}`;
    signalsEl.innerHTML =
      `entropy → leg twitch: ${{target.entropy.toFixed(2)}}<br>` +
      `uncertainty → tun contraction: ${{target.uncertainty.toFixed(2)}}<br>` +
      `confidence → posture openness: ${{target.confidence.toFixed(2)}}<br>` +
      `load/stress → vibration: ${{target.load_stress.toFixed(2)}}<br>` +
      `attention → proboscis extension: ${{target.attention.toFixed(2)}}<br>` +
      `stability → rotation smoothness: ${{target.stability.toFixed(2)}}`;
  }}

  function drawAbstractGlyph(target) {{
    const ctx = glyphCtx;
    const w = glyphCanvas.width;
    const h = glyphCanvas.height;
    const cx = w / 2;
    const cy = h / 2;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, w, h);

    ctx.beginPath();
    ctx.arc(cx, cy, 132, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 2;
    ctx.stroke();

    const x = cx + (target.confidence - target.uncertainty) * 92;
    const y = cy - (target.attention - 0.5) * 112;
    const radius = 16 + target.entropy * 34;

    const grad = ctx.createRadialGradient(x, y, 2, x, y, 105 + target.load_stress * 80);
    grad.addColorStop(0, `rgba(0,255,204,${{0.30 + 0.35 * target.confidence}})`);
    grad.addColorStop(0.45, `rgba(60,130,255,${{0.12 + 0.22 * target.attention}})`);
    grad.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(x, y, 125, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    signalTrace.forEach((s, i) => {{
      const px = cx + (s.confidence - s.uncertainty) * 92;
      const py = cy - (s.attention - 0.5) * 112;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }});
    ctx.strokeStyle = "rgba(255,255,255,0.42)";
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(255,255,255,${{0.20 + 0.35 * target.uncertainty}})`;
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.beginPath();
    ctx.arc(x, y, 6 + target.confidence * 8, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.fill();

    const values = [target.entropy, target.uncertainty, target.confidence, target.load_stress, target.attention, target.stability];
    const labels = ["E", "U", "C", "L", "A", "S"];
    values.forEach((v, i) => {{
      const a = -Math.PI / 2 + i * Math.PI * 2 / values.length;
      const r0 = 104;
      const r1 = 104 + v * 30;
      ctx.beginPath();
      ctx.moveTo(cx + Math.cos(a) * r0, cy + Math.sin(a) * r0);
      ctx.lineTo(cx + Math.cos(a) * r1, cy + Math.sin(a) * r1);
      ctx.strokeStyle = "rgba(255,255,255,0.55)";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.fillStyle = "rgba(0,255,204,0.8)";
      ctx.font = "11px monospace";
      ctx.fillText(labels[i], cx + Math.cos(a) * 142 - 4, cy + Math.sin(a) * 142 + 4);
    }});
  }}

  function applyBiomorphState(t, target) {{
    const smoothing = 0.045;
    for (const key of ["entropy", "uncertainty", "confidence", "load_stress", "attention", "stability"]) {{
      current[key] = lerp(current[key], target[key], smoothing);
    }}

    const tun = current.uncertainty;
    const confidence = current.confidence;
    const stress = current.load_stress;
    const entropy = current.entropy;
    const attention = current.attention;
    const stability = current.stability;

    const open = confidence * (1.0 - 0.65 * tun);
    body.scale.set(
      lerp(1.0, 0.58, tun) * lerp(0.92, 1.08, open),
      lerp(1.0, 0.72, tun) * lerp(0.96, 1.05, open),
      lerp(1.0, 0.78, tun) * lerp(0.96, 1.05, open)
    );

    const probeExtension = lerp(0.20, 1.25, attention) * lerp(1.0, 0.45, tun);
    proboscis.scale.set(1.0, probeExtension, 1.0);
    proboscis.position.x = lerp(1.0, 1.9, probeExtension);

    creature.position.y = lerp(0.02, -0.16, tun) + Math.sin(t * 38.0) * 0.018 * stress;
    creature.position.x = Math.sin(t * 27.0 + 0.7) * 0.012 * stress;

    legs.forEach((leg, i) => {{
      const phase = leg.pair * 0.8 + leg.side * 0.4;
      const tunSuppression = 1.0 - tun;
      const legSpeed = lerp(0.8, 6.0, entropy);
      const legAmplitude = lerp(0.03, 0.23, entropy) * tunSuppression;
      const stressJitter = Math.sin(t * lerp(10, 34, stress) + i * 1.7) * 0.16 * stress * tunSuppression;

      leg.pivot.rotation.x = Math.sin(t * legSpeed + phase) * legAmplitude;
      leg.pivot.rotation.z = stressJitter;

      const retractScale = lerp(1.0, 0.14, tun);
      const openScale = lerp(0.90, 1.08, confidence);
      leg.pivot.scale.setScalar(retractScale * openScale);

      leg.pivot.position.set(
        lerp(leg.basePosition.x, leg.basePosition.x * 0.58, tun),
        lerp(leg.basePosition.y, -0.15, tun),
        lerp(leg.basePosition.z, leg.basePosition.z * 0.35, tun)
      );
    }});

    const baseRotation = 0.004;
    const wobble = (1.0 - stability) * 0.035;
    if (rotationOn) {{
      creature.rotation.y += baseRotation * lerp(0.5, 1.4, stability);
    }}
    creature.rotation.x = Math.sin(t * 2.2) * wobble;
    creature.rotation.z = Math.sin(t * 1.7 + 1.2) * wobble;
  }}

  function animate() {{
    requestAnimationFrame(animate);
    const liveTime = clock.getElapsedTime();
    const t = playing ? liveTime : pausedTime;
    const target = getTargetSignal(t);
    updateUi(target);
    drawAbstractGlyph(target);
    applyBiomorphState(liveTime, target);
    renderer.render(scene, camera);
  }}

  init();
}})();
</script>
"""

  srcdoc = html.escape(inner_html, quote=True)
  return f"""
<iframe
  srcdoc="{srcdoc}"
  style="width:100%; height:800px; border:1px solid #00ffcc; background:#111;"
  sandbox="allow-scripts allow-same-origin"
></iframe>
"""


def run_demo(prompt: str, max_new_tokens: int) -> tuple[str, str]:
  prompt = prompt.strip()
  if not prompt:
    raise gr.Error("Please enter a prompt.")

  set_seed(DEFAULT_SEED)
  raw_steps, final_text = PROBE.collect_trace(prompt, max_new_tokens)
  signals = compute_signals(raw_steps)
  html_out = build_biomorph_html(signals, prompt, final_text)

  signal_table = "step	token	next	entropy	uncertainty	confidence	stress	attention	stability"
  for s in signals:
    signal_table += (
      f"{s.step_idx}	{s.token_text}	{s.generated_token_text}	"
      f"{s.entropy_norm:.3f}	{s.uncertainty:.3f}	{s.confidence:.3f}	"
      f"{s.load_stress:.3f}	{s.attention:.3f}	{s.stability:.3f}"
    )

  return html_out, signal_table


with gr.Blocks(title="LLM Biomorph + Abstract State Glyph") as demo:
  gr.Markdown(
    "# LLM Biomorph + Abstract State Glyph"
    "A cautionary prototype: the same model-derived scalar trace is rendered both as an "
    "abstract glyph and as a deliberately crude tardigrade-like biomorph. The point is not "
    "that this is a good UI, but that weak creature cues can make computational state feel "
    "like agency."
  )

  with gr.Row():
    with gr.Column(scale=1):
      prompt_box = gr.Textbox(
        label="Prompt",
        lines=4,
        value="In Sweden, the capital is",
      )
      max_tokens = gr.Slider(
        label="Max new tokens",
        minimum=1,
        maximum=32,
        step=1,
        value=DEFAULT_MAX_NEW_TOKENS,
      )
      run_button = gr.Button("Render combined trace")
      signal_output = gr.Textbox(label="Signal trace", lines=16)
    with gr.Column(scale=2):
      html_output = gr.HTML(label="Combined glyph + biomorph")

  run_button.click(
    fn=run_demo,
    inputs=[prompt_box, max_tokens],
    outputs=[html_output, signal_output],
  )


if __name__ == "__main__":
  try:
    demo.launch()
  except Exception:
    traceback.print_exc()
    raise
