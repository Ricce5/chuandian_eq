from pathlib import Path

import matplotlib as mpl
from matplotlib import cycler


def resolve_project_root() -> Path:
    """Find project root by locating the nearest directory containing src/."""
    cwd = Path.cwd().resolve()
    for candidate in (cwd, cwd.parent):
        if (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Cannot find project root containing 'src' directory.")


def apply_publication_style() -> None:
    """Apply a compact publication-friendly Matplotlib style."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.1,
            "lines.markersize": 4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.25,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
            "axes.prop_cycle": cycler(
                color=[
                    "#4E79A7",
                    "#F28E2B",
                    "#59A14F",
                    "#E15759",
                    "#76B7B2",
                    "#B07AA1",
                    "#EDC948",
                    "#9C755F",
                    "#BAB0AC",
                ]
            ),
        }
    )


def save_pub_figure(fig, output_path, dpi: int = 300, file_format: str = "pdf") -> None:
    """Save figure in publication-friendly defaults.

    file_format supports 'pdf' or 'png'.
    """
    output_path = Path(output_path)
    file_format = str(file_format).lower().lstrip(".")
    if file_format not in {"pdf", "png"}:
        raise ValueError("file_format must be 'pdf' or 'png'")

    target_path = output_path.with_suffix(f".{file_format}")
    save_kwargs = {"bbox_inches": "tight"}
    if file_format == "png":
        save_kwargs["dpi"] = dpi
    fig.savefig(target_path, format=file_format, **save_kwargs)


def unwrap_compiled_model(model):
    """Return original model when torch.compile wrapper is present."""
    return model._orig_mod if hasattr(model, "_orig_mod") else model
