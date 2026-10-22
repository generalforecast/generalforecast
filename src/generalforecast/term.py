# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Terminal presentation shared by every gf command: one palette, one layout.

Results go to stdout and diagnostics to stderr. Color follows NO_COLOR,
CLICOLOR_FORCE / FORCE_COLOR, CLICOLOR=0, and otherwise whether the stream is
a terminal, so piped output is always plain.
"""

import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from typing import IO

VERB_WIDTH = 12
_CODES = {
    "bold": "1",
    "dim": "2",
    "underline": "4",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "blue": "34",
    "cyan": "36",
}


def color_enabled(stream: IO[str], environ: Mapping[str, str] | None = None) -> bool:
    """Whether to style `stream`, following the NO_COLOR and CLICOLOR conventions."""
    env = os.environ if environ is None else environ
    if env.get("NO_COLOR", "") != "":
        return False
    for force in ("CLICOLOR_FORCE", "FORCE_COLOR"):
        value = env.get(force, "")
        if value in ("0", "false"):
            return False
        if value:
            return True
    if env.get("CLICOLOR") == "0" or env.get("TERM") == "dumb":
        return False
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


class Output:
    """Styled writers for one invocation; quiet keeps only errors."""

    def __init__(
        self,
        *,
        quiet: bool = False,
        verbose: bool = False,
        out: IO[str] | None = None,
        err: IO[str] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Binds stdout/stderr and decides color for each separately."""
        self.out = sys.stdout if out is None else out
        self.err = sys.stderr if err is None else err
        self.quiet, self.verbose = quiet, verbose
        self.color_out = color_enabled(self.out, environ)
        self.color_err = color_enabled(self.err, environ)

    def style(self, text: str, *names: str, stream: IO[str] | None = None) -> str:
        """Wraps text in ANSI styles when the target stream wants color."""
        enabled = self.color_err if stream is self.err else self.color_out
        if not enabled or not names or not text:
            return text
        return f"\x1b[{';'.join(_CODES[n] for n in names)}m{text}\x1b[0m"

    def _write(self, line: str, stream: IO[str]) -> None:
        stream.write(line + "\n")
        stream.flush()

    def line(self, text: str = "") -> None:
        """A plain result line on stdout."""
        if not self.quiet:
            self._write(text, self.out)

    def verb(
        self, verb: str, text: str = "", detail: str = "", *, err: bool = True
    ) -> None:
        """A cargo-style status line: right-aligned bold green verb, then text."""
        if self.quiet:
            return
        stream = self.err if err else self.out
        head = self.style(verb.rjust(VERB_WIDTH), "bold", "green", stream=stream)
        tail = f" {self.style(detail, 'dim', stream=stream)}" if detail else ""
        self._write(f"{head} {text}{tail}".rstrip(), stream)

    def _label(
        self, label: str, colors: Sequence[str], text: str, bold_text: bool
    ) -> None:
        stream = self.err
        head = self.style(label + ":", *colors, "bold", stream=stream)
        body = self.style(text, "bold", stream=stream) if bold_text else text
        self._write(f"{head} {body}", stream)

    def error(self, text: str) -> None:
        """`error: message` in red, always shown."""
        self._label("error", ("red",), text, True)

    def warning(self, text: str) -> None:
        """`warning: message` in yellow."""
        if not self.quiet:
            self._label("warning", ("yellow",), text, True)

    def hint(self, text: str, label: str = "hint") -> None:
        """An indented cyan `hint:` or `help:` follow-up to a diagnostic."""
        stream = self.err
        head = self.style(f"{label}:", "cyan", "bold", stream=stream)
        self._write(f"  {head} {text}", stream)

    def cause(self, text: str) -> None:
        """A dim `Caused by:` line under an error."""
        stream = self.err
        self._write(
            f"  {self.style('Caused by:', 'dim', stream=stream)} {text}", stream
        )

    def say(self, text: str = "", *, force: bool = False) -> None:
        """A plain status line on stderr; force shows it even with --quiet."""
        if force or not self.quiet:
            self._write(text, self.err)

    def note(self, text: str) -> None:
        """Secondary information on stderr, dimmed."""
        if not self.quiet:
            self._write(self.style(text, "dim", stream=self.err), self.err)

    def debug(self, text: str) -> None:
        """Requests and timings, only with --verbose."""
        if self.verbose:
            stream = self.err
            self._write(
                self.style(f"{'debug'.rjust(VERB_WIDTH)} {text}", "dim", stream=stream),
                stream,
            )

    def link(self, url: str, stream: IO[str] | None = None) -> str:
        """An underlined blue URL."""
        return self.style(url, "blue", "underline", stream=stream or self.err)

    def code(self, text: str, stream: IO[str] | None = None) -> str:
        """An inline command or value, in backticks for plain output."""
        return self.style(f"`{text}`", "bold", stream=stream or self.err)

    def pointer(
        self, path: str, line: int, column: int, source: str, label: str, width: int = 1
    ) -> None:
        """A cargo-style source pointer under an error for a gf.yaml location."""
        stream, gutter = self.err, " " * len(str(line))
        bar = self.style("|", "blue", "bold", stream=stream)
        arrow = self.style("-->", "blue", "bold", stream=stream)
        number = self.style(str(line), "blue", "bold", stream=stream)
        mark = self.style(
            "^" * max(width, 1) + " " + label, "red", "bold", stream=stream
        )
        for text in (
            f"{gutter}{arrow} {path}:{line}:{column}",
            f"{gutter} {bar}",
            f"{number} {bar} {source}",
            f"{gutter} {bar} {' ' * (column - 1)}{mark}",
            f"{gutter} {bar}",
        ):
            self._write(text, stream)

    def table(self, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
        """Left-aligned columns, two spaces apart, with dim uppercase headers."""
        if self.quiet:
            return
        cells = [[_cell(v) for v in row] for row in rows]
        widths = [
            max([len(h)] + [len(r[i]) for r in cells]) for i, h in enumerate(headers)
        ]

        def render(values: Sequence[str]) -> str:
            return "  ".join(
                v.ljust(w) for v, w in zip(values, widths, strict=True)
            ).rstrip()

        self._write(self.style(render([h.upper() for h in headers]), "dim"), self.out)
        for row in cells:
            self._write(render(row), self.out)

    def fields(self, pairs: Iterable[tuple[str, object]], indent: int = 0) -> None:
        """Aligned `label  value` pairs on stdout with dim labels."""
        if self.quiet:
            return
        items = [(k, _cell(v)) for k, v in pairs if v is not None]
        width = max((len(k) for k, _ in items), default=0)
        for key, value in items:
            label = self.style(key.ljust(width), "dim")
            self._write(f"{' ' * indent}{label} {value}", self.out)


def _cell(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(_cell(v) for v in value)
    return str(value)


def seconds(value: float) -> str:
    """A compact duration like 0.41s, 12s or 3m 05s."""
    if value < 10:
        return f"{value:.2f}s"
    if value < 60:
        return f"{value:.0f}s"
    minutes, rest = divmod(int(value), 60)
    return f"{minutes}m {rest:02d}s"


def suggest(word: str, choices: Iterable[str]) -> str | None:
    """The closest known name to a typo, if any is close enough."""
    import difflib

    match = difflib.get_close_matches(word, list(choices), n=1, cutoff=0.6)
    return match[0] if match else None
