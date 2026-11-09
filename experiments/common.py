"""Shared utilities for the experiment scripts."""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))

FIGURES = os.path.join(ROOT, "figures")
os.makedirs(FIGURES, exist_ok=True)


def fit_order(hs, errs):
    hs = np.asarray(hs, float)
    errs = np.asarray(errs, float)
    good = errs > 0
    p = np.polyfit(np.log(hs[good]), np.log(errs[good]), 1)
    resid = np.log(errs[good]) - np.polyval(p, np.log(hs[good]))
    return float(p[0]), float(np.abs(resid).max())


def pairwise_orders(errs):
    out = [None]
    for a, b in zip(errs[:-1], errs[1:]):
        out.append(float(np.log2(a / b)) if b > 0 else None)
    return out


def save_json(name, obj):
    path = os.path.join(FIGURES, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
    return path


def load_results():
    path = os.path.join(FIGURES, "results.json")
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def merge_results(update):
    res = load_results()
    res.update(update)
    save_json("results.json", res)
    return res
