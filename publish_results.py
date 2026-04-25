#!/usr/bin/env python3
"""
SUITS-VLM-Bench — Publish Results
-----------------------------------
Copies a results JSON file from output/results/ to results/
and updates results/index.json so the GitHub Pages site auto-loads it.

Usage:
    python publish_results.py                          # publishes all files in output/results/
    python publish_results.py output/results/results_gemini-2.5-flash_20260424.json
"""

import sys
import json
import shutil
from pathlib import Path

RESULTS_DIR = Path("results")
OUTPUT_DIR  = Path("output/results")

def update_manifest(filename: str):
    manifest_path = RESULTS_DIR / "index.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"runs": []}

    if filename not in manifest["runs"]:
        manifest["runs"].append(filename)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"  Updated results/index.json → {manifest['runs']}")
    else:
        print(f"  {filename} already in index.json")

def publish(src: Path):
    if not src.exists():
        print(f"  Not found: {src}")
        return
    dest = RESULTS_DIR / src.name
    shutil.copy2(src, dest)
    print(f"  Copied {src.name} → results/")
    update_manifest(src.name)

RESULTS_DIR.mkdir(exist_ok=True)

if len(sys.argv) > 1:
    # Specific files passed as arguments
    for arg in sys.argv[1:]:
        publish(Path(arg))
else:
    # Publish everything in output/results/
    if not OUTPUT_DIR.exists():
        print(f"No output/results/ folder found. Run evaluate.py first.")
        sys.exit(1)
    files = sorted(OUTPUT_DIR.glob("results_*.json"))
    if not files:
        print("No result files found in output/results/")
        sys.exit(1)
    for f in files:
        publish(f)

print("\nDone. Now run:")
print("  git add results/")
print("  git commit -m 'add evaluation results'")
print("  git push")
print("\nThe site will auto-load the new results on next page visit.")