"""Chart style shared by the figures (dark 1920x1080 frames, validated categorical palette).

    from style import use, SLOTS, ink, titles, source_line
    use("dark")            # video default; "light" for README/web figures
"""
from __future__ import annotations

import matplotlib as mpl

TOKENS = {
    "dark": {
        "surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "baseline": "#383835",
        "slots": ["#3987e5", "#d95926", "#199e70", "#c98500",
                  "#d55181", "#008300", "#9085e9", "#e66767"],
    },
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "baseline": "#c3c2b7",
        "slots": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                  "#e87ba4", "#008300", "#6250d6", "#e34948"],
    },
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

_mode = "dark"
SLOTS = TOKENS[_mode]["slots"]


def ink(role: str = "ink") -> str:
    return TOKENS[_mode][role]


def use(mode: str = "dark", scale: float = 1.0) -> None:
    """Apply tokens. scale=1 targets a 1920x1080 frame at dpi 100 (fig 19.2x10.8 in)."""
    global _mode, SLOTS
    _mode, t = mode, TOKENS[mode]
    SLOTS = t["slots"]
    px = lambda v: v * 0.72 * scale  # frame pixels -> points at dpi 100
    mpl.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "savefig.facecolor": t["surface"], "figure.dpi": 100, "savefig.dpi": 100,
        "font.family": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "axes.titlecolor": t["ink"],
        "axes.titlesize": px(56), "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.labelsize": px(36), "xtick.labelsize": px(32), "ytick.labelsize": px(32),
        "legend.fontsize": px(32), "legend.frameon": False,
        "xtick.color": t["muted"], "ytick.color": t["muted"],
        "axes.edgecolor": t["baseline"], "axes.linewidth": 1,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": t["grid"],
        "grid.linewidth": 1, "grid.linestyle": "-", "axes.axisbelow": True,
        "lines.linewidth": px(4), "lines.solid_capstyle": "round",
        "lines.markersize": px(14), "lines.markeredgewidth": px(3),
        "lines.markeredgecolor": t["surface"],
        "axes.prop_cycle": mpl.cycler(color=t["slots"]),
        "figure.figsize": (19.2, 10.8),
    })


def titles(fig, ax, title: str, subtitle: str | None = None) -> None:
    """Title, then subtitle under it, left-aligned with the plot area (axes top at 0.80)."""
    x = ax.get_position().x0
    fig.text(x, 0.92, title, color=ink("ink"), fontsize=mpl.rcParams["axes.titlesize"],
             fontweight="semibold", ha="left", va="baseline")
    if subtitle:
        fig.text(x, 0.865, subtitle, color=ink("ink2"), fontsize=mpl.rcParams["axes.labelsize"],
                 ha="left", va="baseline")


def source_line(fig, text: str) -> None:
    """Muted footer: hardware · engine version · date · measured|estimate."""
    fig.text(0.05, 0.03, text, color=ink("muted"), fontsize=mpl.rcParams["xtick.labelsize"] * 0.8,
             ha="left", va="bottom")


use("dark")
