#!/usr/bin/env python3
"""
SUITS-VLM-Bench — Evaluation Script
--------------------------------------
Reads augmented images + manifest.json from output/augmented/
Calls Gemini API for each image, scores predictions, saves results JSON.

Usage:
    python evaluate.py                          # reads key from .env
    python evaluate.py --key YOUR_API_KEY       # or pass directly
    python evaluate.py --model gemini-2.0-flash
    python evaluate.py --limit 10              # test on 10 images first
    python evaluate.py --phase NAVIGATION      # one phase only

Setup:
    pip install Pillow numpy google-genai python-dotenv

API key (recommended):
    Create a .env file in the repo root:
        GEMINI_API_KEY=AIza...
    The script will load it automatically.
"""

import os
import sys
import json
import time
import argparse
import random
from datetime import datetime
from pathlib import Path

# ── Dependency check ─────────────────────────────────────────────────────────

missing = []
try:
    from PIL import Image
except ImportError:
    missing.append("Pillow")

try:
    from google import genai
    from google.genai import types
except ImportError:
    missing.append("google-genai")

try:
    import numpy as np
except ImportError:
    missing.append("numpy")

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env from current working directory or any parent
except ImportError:
    pass  # python-dotenv is optional — just means .env won't auto-load

if missing:
    print("Missing dependencies. Install with:")
    print(f"  pip install {' '.join(missing)} python-dotenv")
    sys.exit(1)


# ── CLI ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="SUITS-VLM-Bench evaluation pipeline")
    p.add_argument("--key",     default=None,                        help="Gemini API key (or set GEMINI_API_KEY in .env)")
    p.add_argument("--model",   default="gemini-2.0-flash",          help="Gemini model (default: gemini-2.0-flash)")
    p.add_argument("--input",   default="output/augmented",          help="Augmented images folder with manifest.json")
    p.add_argument("--output",  default="output/results",            help="Folder to write results JSON")
    p.add_argument("--label",   default=None,                        help="Run label (default: model + timestamp)")
    p.add_argument("--limit",   type=int,   default=None,            help="Max images to evaluate (for testing)")
    p.add_argument("--delay",   type=float, default=0.5,             help="Seconds between API calls (default: 0.5)")
    p.add_argument("--shuffle", action="store_true",                 help="Shuffle image order before evaluation")
    p.add_argument("--phase",   default=None,
                   choices=["EGRESS", "NAVIGATION", "LTV_REPAIR"],   help="Evaluate one phase only")
    p.add_argument("--thresh-low",  type=float, default=0.50,        help="Low confidence threshold (default: 0.50)")
    p.add_argument("--thresh-med",  type=float, default=0.70,        help="Medium confidence threshold (default: 0.70)")
    p.add_argument("--thresh-high", type=float, default=0.90,        help="High confidence threshold (default: 0.90)")
    p.add_argument("--prompt",  default=None,                        help="Path to custom prompt .txt file")
    return p.parse_args()


# ── Prompt ───────────────────────────────────────────────────────────────────

DEFAULT_PROMPT = """You are evaluating NASA SUITS EVA (Extravehicular Activity) operation imagery.

Your task is to classify which EVA phase is shown in the image.

The three possible phases are:
- EGRESS: Astronaut exiting airlock, initial positioning, suit and equipment checks
- NAVIGATION: Crew traversing the surface or worksite, navigation panel interaction, ANAV/LiDAR/comms systems visible
- LTV_REPAIR: Lunar Terrain Vehicle maintenance, fuse box access, PDD or science management hardware interaction

Respond ONLY with a valid JSON object in exactly this format:
{
  "phase": "EGRESS" | "NAVIGATION" | "LTV_REPAIR",
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<one sentence explanation>"
}

Do not include any text, markdown, or explanation outside the JSON object."""


# ── Gemini call ──────────────────────────────────────────────────────────────

def call_gemini(client, model_name: str, image_path: Path, prompt: str) -> dict:
    """Send image to Gemini, return parsed { phase, confidence, reasoning }."""

    img = Image.open(image_path).convert("RGB")

    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, img],
        config=types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=512,
        ),
    )

    text = response.text.strip()

    # Strip markdown fences if model wraps output
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first line (```json or ```) and last line (```)
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(inner).strip()

    parsed = json.loads(text)

    valid_phases = {"EGRESS", "NAVIGATION", "LTV_REPAIR"}
    if parsed.get("phase") not in valid_phases:
        raise ValueError(f"Invalid phase returned: {parsed.get('phase')!r}")
    if not isinstance(parsed.get("confidence"), (int, float)):
        raise ValueError(f"Invalid confidence: {parsed.get('confidence')!r}")

    parsed["confidence"] = float(parsed["confidence"])
    return parsed


# ── Scoring helpers ──────────────────────────────────────────────────────────

PHASES = ["EGRESS", "NAVIGATION", "LTV_REPAIR"]


def build_confusion_matrix(results: list) -> dict:
    cm = {t: {p: 0 for p in PHASES} for t in PHASES}
    for r in results:
        if r["predicted_phase"] and r["true_phase"] in cm:
            pred = r["predicted_phase"]
            if pred in cm[r["true_phase"]]:
                cm[r["true_phase"]][pred] += 1
    return cm


def build_phase_accuracy(results: list) -> dict:
    pa = {}
    for p in PHASES:
        pr = [r for r in results if r["true_phase"] == p]
        correct = sum(1 for r in pr if r["correct"])
        confs = [r["confidence"] for r in pr if r["confidence"] is not None]
        pa[p] = {
            "total": len(pr),
            "correct": correct,
            "accuracy": correct / len(pr) if pr else None,
            "avg_confidence": sum(confs) / len(confs) if confs else None,
        }
    return pa


def threshold_accuracy(results: list, threshold: float) -> dict:
    above = [r for r in results if r["confidence"] is not None and r["confidence"] >= threshold]
    correct = sum(1 for r in above if r["correct"])
    return {
        "threshold": threshold,
        "n_above": len(above),
        "n_correct": correct,
        "accuracy": correct / len(above) if above else None,
        "coverage": len(above) / len(results) if results else None,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Resolve API key: CLI → env var → error
    api_key = args.key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("No API key found.")
        print("Either pass --key YOUR_KEY, or create a .env file containing:")
        print("  GEMINI_API_KEY=AIza...")
        sys.exit(1)

    # Init Gemini client (new SDK)
    client = genai.Client(api_key=api_key)
    print(f"Model  : {args.model}")

    # Load prompt
    if args.prompt:
        prompt_path = Path(args.prompt)
        if not prompt_path.exists():
            print(f"Prompt file not found: {args.prompt}")
            sys.exit(1)
        prompt = prompt_path.read_text().strip()
        print(f"Prompt : {args.prompt}")
    else:
        prompt = DEFAULT_PROMPT
        print("Prompt : default")

    # Load manifest
    input_root = Path(args.input)
    manifest_path = input_root / "manifest.json"

    if not manifest_path.exists():
        print(f"\nmanifest.json not found in {input_root}")
        print("Run augment.py first to generate augmented images.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    images = manifest.get("images", [])
    print(f"Images : {len(images)} in manifest")

    # Filter / shuffle / limit
    if args.phase:
        images = [img for img in images if img["phase"] == args.phase]
        print(f"Filter : phase={args.phase} → {len(images)} images")

    if not images:
        print("No images to evaluate.")
        sys.exit(0)

    if args.shuffle:
        random.shuffle(images)
        print("Order  : shuffled")

    if args.limit:
        images = images[:args.limit]
        print(f"Limit  : {args.limit} images")

    # Output setup
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    run_label = args.label or f"{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    total = len(images)
    results = []

    print(f"\nEvaluating {total} images — run: {run_label}\n")
    print(f"{'#':<5} {'File':<45} {'True':<14} {'Predicted':<14} {'Conf':>6}  Result")
    print("─" * 95)

    for i, item in enumerate(images):
        img_path = input_root / item["path"]
        true_phase = item["phase"]

        if not img_path.exists():
            print(f"{i+1:<5} {item['filename']:<45} {true_phase:<14} {'NOT FOUND':<14} {'—':>6}  ✗")
            results.append({
                "filename": item["filename"],
                "true_phase": true_phase,
                "predicted_phase": None,
                "confidence": None,
                "correct": False,
                "reasoning": "File not found",
                "error": True,
            })
            continue

        try:
            pred = call_gemini(client, args.model, img_path, prompt)
            correct = pred["phase"] == true_phase
            conf_str = f"{pred['confidence'] * 100:.0f}%"
            mark = "✓" if correct else "✗"
            print(f"{i+1:<5} {item['filename']:<45} {true_phase:<14} {pred['phase']:<14} {conf_str:>6}  {mark}")

            results.append({
                "filename": item["filename"],
                "true_phase": true_phase,
                "predicted_phase": pred["phase"],
                "confidence": pred["confidence"],
                "correct": correct,
                "reasoning": pred.get("reasoning", ""),
                "error": False,
            })

        except Exception as e:
            print(f"{i+1:<5} {item['filename']:<45} {true_phase:<14} {'ERROR':<14} {'—':>6}  ✗  [{e}]")
            results.append({
                "filename": item["filename"],
                "true_phase": true_phase,
                "predicted_phase": None,
                "confidence": None,
                "correct": False,
                "reasoning": f"Error: {e}",
                "error": True,
            })

        if i < total - 1:
            time.sleep(args.delay)

    # ── Build output ──────────────────────────────────────────────────────────

    valid = [r for r in results if not r["error"]]
    correct_count = sum(1 for r in valid if r["correct"])
    confs = [r["confidence"] for r in valid if r["confidence"] is not None]

    output_data = {
        "run_label": run_label,
        "model": args.model,
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "config": {
            "limit": args.limit,
            "phase_filter": args.phase,
            "delay": args.delay,
            "shuffle": args.shuffle,
        },
        "summary": {
            "total": total,
            "evaluated": len(valid),
            "errors": total - len(valid),
            "correct": correct_count,
            "accuracy": correct_count / len(valid) if valid else None,
            "avg_confidence": sum(confs) / len(confs) if confs else None,
        },
        "phase_accuracy": build_phase_accuracy(valid),
        "confusion_matrix": build_confusion_matrix(valid),
        "threshold_analysis": {
            "low":    threshold_accuracy(valid, args.thresh_low),
            "medium": threshold_accuracy(valid, args.thresh_med),
            "high":   threshold_accuracy(valid, args.thresh_high),
        },
        "thresholds": {
            "low": args.thresh_low,
            "medium": args.thresh_med,
            "high": args.thresh_high,
        },
        "results": results,
    }

    out_file = output_root / f"results_{run_label}.json"
    with open(out_file, "w") as f:
        json.dump(output_data, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────

    pa = output_data["phase_accuracy"]
    cm = output_data["confusion_matrix"]

    print("\n" + "─" * 95)
    print(f"\nRun   : {run_label}")
    print(f"Model : {args.model}")

    print(f"\n── Accuracy ──────────────────────────────────────────")
    if valid:
        print(f"  Overall      {correct_count}/{len(valid)} = {output_data['summary']['accuracy']*100:.1f}%")
    for phase in PHASES:
        s = pa[phase]
        acc = f"{s['accuracy']*100:.1f}%" if s["accuracy"] is not None else "—"
        avg = f"{s['avg_confidence']*100:.0f}%" if s["avg_confidence"] is not None else "—"
        print(f"  {phase:<15} {s['correct']}/{s['total']} = {acc}  (avg conf {avg})")

    print(f"\n── Confidence Thresholds ─────────────────────────────")
    for tier, s in output_data["threshold_analysis"].items():
        acc = f"{s['accuracy']*100:.1f}%" if s["accuracy"] is not None else "—"
        cov = f"{s['coverage']*100:.0f}%" if s["coverage"] is not None else "—"
        print(f"  {tier.upper():<8} ≥{s['threshold']:.2f}  acc={acc}  coverage={cov}  n={s['n_above']}")

    print(f"\n── Confusion Matrix ──────────────────────────────────")
    header = f"  {'':>15} " + "  ".join(f"{p[:9]:>9}" for p in PHASES)
    print(header)
    for true_p in PHASES:
        row = f"  {true_p:>15} " + "  ".join(f"{cm[true_p].get(pred_p, 0):>9}" for pred_p in PHASES)
        print(row)

    print(f"\nSaved : {out_file}")
    print(f"\nNext  : copy this file to results/ and push to GitHub to update the dashboard.\n")


if __name__ == "__main__":
    main()