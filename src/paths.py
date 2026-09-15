"""Single source of truth for paths in this repository.

``REPO_ROOT`` is derived from the location of this file, so a fresh clone works
with no edits. Do not hardcode absolute paths anywhere else in the repo.

From a notebook anywhere in the repo, the first cell finds the repo root, puts
src/ on the import path, and changes into the repo root so relative paths such
as ``data/i24/i24_11_30`` resolve::

    import os, sys
    _root = os.path.abspath(os.getcwd())
    while not os.path.exists(os.path.join(_root, "src", "paths.py")):
        ...

Run ``python src/paths.py`` to print every resolved path and whether it exists.
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
DATA_ROOT = os.path.join(REPO_ROOT, "data")
RAW_I24_DATA = os.path.join(REPO_ROOT, "raw_i24_data")
CALIBRATION_ROOT = os.path.join(REPO_ROOT, "calibration")
FIGS_ROOT = os.path.join(REPO_ROOT, "figs")

I24_DATA = os.path.join(DATA_ROOT, "i24")
I24_CALIBRATION = os.path.join(CALIBRATION_ROOT, "i24")


def _i24_id(date):
    """Accept "11_30" or "i24_11_30" and return "i24_11_30"."""
    return date if date.startswith("i24_") else f"i24_{date}"


def i24_data(date):
    """Input data folder for one I-24 day, e.g. i24_data("11_30")."""
    return os.path.join(I24_DATA, _i24_id(date))


def i24_calibration(date, *parts):
    """Calibration output folder for one I-24 day, e.g.
    i24_calibration("11_30", "calibration_dynamic_smooth")."""
    return os.path.join(I24_CALIBRATION, _i24_id(date), *parts)


def fig(*parts):
    """Path under figs/, e.g. fig("itsc2026", "FD_comparison.png")."""
    return os.path.join(FIGS_ROOT, *parts)


def i24_dates():
    """Sorted I-24 day identifiers present in data/i24/."""
    if not os.path.isdir(I24_DATA):
        return []
    return sorted(d for d in os.listdir(I24_DATA) if d.startswith("i24_"))


if __name__ == "__main__":
    for name in ["REPO_ROOT", "SRC_ROOT", "DATA_ROOT", "RAW_I24_DATA", "CALIBRATION_ROOT",
                 "FIGS_ROOT", "I24_DATA", "I24_CALIBRATION"]:
        path = globals()[name]
        print(f"{name:18s} {path}  {'(exists)' if os.path.exists(path) else '(missing)'}")
    print("I-24 days:", ", ".join(i24_dates()))
