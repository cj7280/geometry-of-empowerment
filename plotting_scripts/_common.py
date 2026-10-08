"""Shared paths, command-line options, and plot-data records for the figure generators."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "figures" / "paper-figures"


def parse_args(description, *, recompute=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="output directory (relative paths are resolved from the repository root)",
    )
    if recompute:
        parser.add_argument(
            "--recompute-fig4",
            action="store_true",
            help="ignore and replace Figure 4's cached capacity field",
        )
    return parser.parse_args()


def prepare_output(out_dir):
    """Select the Agg backend and create out_dir, resolving relative paths from ROOT."""
    import matplotlib

    matplotlib.use("Agg")
    out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def write_plot_data(out_dir, stem, record):
    """Write the ``<stem>_plot_data.json`` record that accompanies each figure."""
    with (Path(out_dir) / f"{stem}_plot_data.json").open("w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
