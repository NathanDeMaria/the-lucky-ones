"""
Enough SVG to draw the validation charts, and no more.

There is no plotting dependency in this repo and this is why there doesn't
need to be: the charts in `docs/` are a scatter, a few lines and a bar
chart, and the whole vocabulary for those fits in one file. matplotlib is
~60MB to draw eleven shapes.

Everything here writes a standalone `<svg>` with an explicit light
background. That matters for where these end up: GitHub renders an SVG in a
markdown file against whatever theme the reader has chosen, and a chart with
a transparent ground is dark text on dark for half the audience.

Coordinates are data coordinates everywhere in the public API. `Axes` owns
the one transform from those to pixels, so nothing that draws has to know
where the plot area is.
"""

from dataclasses import dataclass, field
from typing import Iterable, Sequence

# A palette that survives both themes and holds up in greyscale, which is
# what a chart in a pull request diff gets read in.
INK = "#1c1c1c"
MUTED = "#6b6b6b"
GRID = "#e3e3e3"
GROUND = "#fbfbfa"
BLUE = "#2f6f9f"
ORANGE = "#c8642a"
GREEN = "#3f7d54"
RED = "#a8342c"
PURPLE = "#7a5495"
SERIES = (BLUE, ORANGE, GREEN, PURPLE, RED)

FONT = (
    "ui-sans-serif, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', "
    "Arial, sans-serif"
)


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


@dataclass
class Axes:
    """
    A plot area, and the transform from data coordinates into it.

    `pad` is (left, right, top, bottom) in pixels: the room the tick labels
    and the axis titles live in, outside the plot area itself.
    """

    width: int
    height: int
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    pad: tuple[int, int, int, int] = (56, 18, 34, 44)
    parts: list[str] = field(default_factory=list)

    @property
    def left(self) -> int:
        return self.pad[0]

    @property
    def right(self) -> int:
        return self.width - self.pad[1]

    @property
    def top(self) -> int:
        return self.pad[2]

    @property
    def bottom(self) -> int:
        return self.height - self.pad[3]

    def x(self, value: float) -> float:
        lo, hi = self.xlim
        span = (hi - lo) or 1.0
        return self.left + (value - lo) / span * (self.right - self.left)

    def y(self, value: float) -> float:
        lo, hi = self.ylim
        span = (hi - lo) or 1.0
        # SVG's y grows downward, so the axis is flipped here and nowhere else.
        return self.bottom - (value - lo) / span * (self.bottom - self.top)

    # --- primitives -----------------------------------------------------

    def raw(self, markup: str) -> None:
        self.parts.append(markup)

    def line(
        self,
        points: Sequence[tuple[float, float]],
        color: str = INK,
        width: float = 1.8,
        dash: str | None = None,
        opacity: float = 1.0,
    ) -> None:
        if len(points) < 2:
            return
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{self.x(px):.2f},{self.y(py):.2f}"
            for i, (px, py) in enumerate(points)
        )
        stroke = f' stroke-dasharray="{dash}"' if dash else ""
        self.raw(
            f'<path d="{path}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round" '
            f'stroke-linecap="round" opacity="{opacity}"{stroke}/>'
        )

    def band(
        self,
        points: Sequence[tuple[float, float, float]],
        color: str,
        opacity: float = 0.16,
    ) -> None:
        """A filled ribbon from `(x, low, high)` triples -- a confidence band."""
        if len(points) < 2:
            return
        top = " ".join(f"{self.x(px):.2f},{self.y(hi):.2f}" for px, _, hi in points)
        bottom = " ".join(
            f"{self.x(px):.2f},{self.y(lo):.2f}" for px, lo, _ in reversed(points)
        )
        self.raw(
            f'<polygon points="{top} {bottom}" fill="{color}" '
            f'opacity="{opacity}" stroke="none"/>'
        )

    def dots(
        self,
        points: Iterable[tuple[float, float]],
        color: str = BLUE,
        radius: float = 3.2,
        opacity: float = 0.9,
        edge: str | None = None,
    ) -> None:
        stroke = f' stroke="{edge}" stroke-width="1"' if edge else ""
        for px, py in points:
            self.raw(
                f'<circle cx="{self.x(px):.2f}" cy="{self.y(py):.2f}" '
                f'r="{radius}" fill="{color}" opacity="{opacity}"{stroke}/>'
            )

    def bars(
        self,
        values: Sequence[tuple[float, float]],
        width: float,
        color: str = BLUE,
        opacity: float = 0.85,
        baseline: float = 0.0,
    ) -> None:
        """`(centre, height)` pairs, `width` in data units."""
        for centre, height in values:
            x0, x1 = self.x(centre - width / 2), self.x(centre + width / 2)
            y0, y1 = self.y(baseline), self.y(height)
            top, bottom = min(y0, y1), max(y0, y1)
            if bottom - top < 0.4:
                continue
            self.raw(
                f'<rect x="{x0:.2f}" y="{top:.2f}" width="{max(x1 - x0, 0.6):.2f}" '
                f'height="{bottom - top:.2f}" fill="{color}" opacity="{opacity}"/>'
            )

    def hline(self, value: float, color: str = MUTED, dash: str = "4 3") -> None:
        self.raw(
            f'<line x1="{self.left}" y1="{self.y(value):.2f}" x2="{self.right}" '
            f'y2="{self.y(value):.2f}" stroke="{color}" stroke-width="1" '
            f'stroke-dasharray="{dash}"/>'
        )

    def vline(self, value: float, color: str = MUTED, dash: str = "4 3") -> None:
        self.raw(
            f'<line x1="{self.x(value):.2f}" y1="{self.top}" '
            f'x2="{self.x(value):.2f}" y2="{self.bottom}" stroke="{color}" '
            f'stroke-width="1" stroke-dasharray="{dash}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        label: str,
        size: float = 11,
        color: str = INK,
        anchor: str = "start",
        weight: str = "400",
        pixels: bool = False,
        italic: bool = False,
    ) -> None:
        px, py = (x, y) if pixels else (self.x(x), self.y(y))
        style = ' font-style="italic"' if italic else ""
        self.raw(
            f'<text x="{px:.2f}" y="{py:.2f}" font-family="{FONT}" '
            f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{style}>{_escape(label)}</text>'
        )

    # --- furniture ------------------------------------------------------

    def frame(
        self,
        xticks: Sequence[float],
        yticks: Sequence[float],
        xlabel: str = "",
        ylabel: str = "",
        xformat: str = "{:g}",
        yformat: str = "{:g}",
        xtick_labels: Sequence[str] | None = None,
    ) -> None:
        """Gridlines, ticks and axis titles. Called before anything is drawn."""
        for value in yticks:
            y = self.y(value)
            self.raw(
                f'<line x1="{self.left}" y1="{y:.2f}" x2="{self.right}" '
                f'y2="{y:.2f}" stroke="{GRID}" stroke-width="1"/>'
            )
            self.text(
                self.left - 8,
                y + 3.5,
                yformat.format(value),
                size=10.5,
                color=MUTED,
                anchor="end",
                pixels=True,
            )
        labels = xtick_labels or [xformat.format(value) for value in xticks]
        for value, label in zip(xticks, labels):
            x = self.x(value)
            self.raw(
                f'<line x1="{x:.2f}" y1="{self.top}" x2="{x:.2f}" '
                f'y2="{self.bottom}" stroke="{GRID}" stroke-width="1"/>'
            )
            self.text(
                x,
                self.bottom + 16,
                label,
                size=10.5,
                color=MUTED,
                anchor="middle",
                pixels=True,
            )
        self.raw(
            f'<line x1="{self.left}" y1="{self.bottom}" x2="{self.right}" '
            f'y2="{self.bottom}" stroke="{MUTED}" stroke-width="1"/>'
        )
        if xlabel:
            self.text(
                (self.left + self.right) / 2,
                self.height - 8,
                xlabel,
                size=11,
                color=INK,
                anchor="middle",
                pixels=True,
            )
        if ylabel:
            cx, cy = 15, (self.top + self.bottom) / 2
            self.raw(
                f'<text x="{cx}" y="{cy}" font-family="{FONT}" font-size="11" '
                f'fill="{INK}" text-anchor="middle" '
                f'transform="rotate(-90 {cx} {cy})">{_escape(ylabel)}</text>'
            )

    def legend(self, entries: Sequence[tuple[str, str]], x: float, y: float) -> None:
        """`(label, color)` pairs, laid out downwards from a pixel position."""
        for index, (label, color) in enumerate(entries):
            row = y + index * 16
            self.raw(
                f'<rect x="{x}" y="{row - 7}" width="10" height="10" rx="2" '
                f'fill="{color}"/>'
            )
            self.text(x + 15, row + 1.5, label, size=10.5, color=INK, pixels=True)

    def render(self, title: str = "", subtitle: str = "") -> str:
        head = []
        if title:
            head.append(
                f'<text x="{self.left - 40}" y="16" font-family="{FONT}" '
                f'font-size="12.5" font-weight="600" fill="{INK}">'
                f"{_escape(title)}</text>"
            )
        if subtitle:
            head.append(
                f'<text x="{self.left - 40}" y="{16 + (14 if title else 0)}" '
                f'font-family="{FONT}" font-size="10.5" fill="{MUTED}">'
                f"{_escape(subtitle)}</text>"
            )
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}" '
            f'role="img" aria-label="{_escape(title or "chart")}">'
            f'<rect width="{self.width}" height="{self.height}" fill="{GROUND}"/>'
            + "".join(head)
            + "".join(self.parts)
            + "</svg>\n"
        )


def nice_ticks(lo: float, hi: float, count: int = 6) -> list[float]:
    """
    Round tick values covering `[lo, hi]`.

    Steps from the 1/2/5 family, which is what makes ticks read as numbers a
    person would have chosen rather than as `hi - lo` divided by six.
    """
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / max(count - 1, 1)
    magnitude = 10.0 ** int(_floor_log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * magnitude
        if step >= raw:
            break
    start = step * int(lo / step) - (step if lo < 0 and lo % step else 0)
    ticks, value = [], start
    while value <= hi + step * 1e-9:
        if value >= lo - step * 1e-9:
            ticks.append(round(value, 10))
        value += step
    return ticks or [lo, hi]


def _floor_log10(value: float) -> int:
    from math import floor, log10

    return int(floor(log10(abs(value)))) if value else 0
