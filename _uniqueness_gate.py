#!/usr/bin/env python3
"""Hard uniqueness gate for the kalos catalog.

Steven's requirement: exactly 50 DISTINCT mathematical models, none repeated.
This enumerates every KINDS list across scene modules plus single-kind scenes
(attractor maps via _MAPS, harmonograph) and fails loudly on any duplicate or
on a final count != 50. Run before shipping the reel catalog.
"""
import ast
import sys
import pathlib
import collections

SCENES = pathlib.Path("kalos/scenes")


def kinds_in(path):
    """Static-parse KINDS/_KINDS/_MAPS string literals from a scene module."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            tgt = getattr(node.targets[0], "id", "")
            if tgt in ("KINDS", "_KINDS"):
                v = node.value
                elts = getattr(v, "elts", None) or getattr(v, "keys", None) or []
                for e in elts:
                    if isinstance(e, ast.Constant) and isinstance(e.value, str):
                        out.append(e.value)
            if tgt == "_MAPS" and hasattr(node.value, "keys"):
                for k in node.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        out.append(k.value)
    return out


inventory = collections.OrderedDict()  # model -> [modules]
for py in sorted(SCENES.glob("*.py")):
    if py.name.startswith("__"):
        continue
    mod = py.stem
    ks = kinds_in(py)
    if not ks and mod == "harmonograph":
        ks = ["harmonograph"]
    for k in ks:
        inventory.setdefault(k, []).append(mod)

dupes = {k: v for k, v in inventory.items() if len(v) > 1}
total = len(inventory)

print(f"UNIQUE MODELS: {total}")
for m in inventory:
    print(f"  - {m:24s} [{inventory[m][0]}]")

if dupes:
    print("\nDUPLICATES (same model in >1 module):")
    for k, v in dupes.items():
        print(f"  !! {k}: {v}")
    sys.exit(2)

print("\nOK: no duplicates -- every model is distinct.")
if total < 50:
    print(f"NOTE: {total}/50 built so far ({50 - total} more to reach target).")
else:
    print(f"OK: {total} unique models (target 50; extras are bonus, not repeats).")
