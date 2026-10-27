# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Terminal output for the `gf` command.

One renderer draws every surface the CLI shows: status lines with a
right-aligned verb, lock change lines, the three error shapes, tables, and
the interactive prompts `gf init` uses. Results go to stdout and only
`generalforecast.render` writes them; status lines, prompts and errors go to
stderr. Colour is on only when the stream is a terminal and `NO_COLOR` is
unset.
"""

from __future__ import annotations

import os
import re
import select
import sys
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import IO, Any

from generalforecast.errors import Error, InvalidInput

# Status verbs are right-aligned in this many columns.
VERB_WIDTH = 12
# Collapsed prompt labels are padded to this width so values start at column 14.
LABEL_WIDTH = 13
# Ends every frame a raw prompt draws. The cursor stays hidden while a frame is
# up, and whatever reads the terminal knows the frame is whole once it arrives.
FRAME_END = "\x1b[?25l"
# The rendering size when the terminal does not report one.
WIDTH = 100
HEIGHT = 24
# Lock change keys (model, input, selection, target, context) share one column.
LOCK_KEY_WIDTH = len("selection")
# Seconds a raw prompt waits for the rest of an escape sequence.
ESCAPE_SECONDS = 0.05

_CODES = {
    "bold": "1",
    "dim": "2",
    "red": "31",
    "green": "32",
    "yellow": "33",
    "cyan": "36",
}


def color_enabled(
    stream: IO[str] | None, environ: Mapping[str, str] | None = None
) -> bool:
    """Whether `stream` should get ANSI colour.

    `NO_COLOR` set to anything turns colour off; `FORCE_COLOR` set to a value
    other than `0` turns it on; otherwise colour follows whether the stream is
    a terminal and `TERM` is not `dumb`.
    """
    env = os.environ if environ is None else environ
    if env.get("NO_COLOR"):
        return False
    force = env.get("FORCE_COLOR")
    if force is not None:
        return force != "0"
    if env.get("TERM") == "dumb":
        return False
    return bool(stream is not None and getattr(stream, "isatty", lambda: False)())


def seconds(value: float) -> str:
    """Formats a duration as the status lines show it: `0.21s`, `6.8s`, `41s`."""
    if value < 1:
        return f"{value:.2f}s"
    if value < 10:
        return f"{value:.1f}s"
    return f"{value:.0f}s"


def count(value: int) -> str:
    """Formats a count with thousands separators."""
    return f"{value:,}"


def plural(value: int, noun: str) -> str:
    """`1 row`, `2,052 rows`."""
    return f"{count(value)} {noun}{'' if value == 1 else 's'}"


def size(stream: IO[str] | None = None) -> os.terminal_size:
    """The terminal's columns and lines, or `WIDTH` by `HEIGHT` when unknown."""
    try:
        return os.get_terminal_size((stream or sys.stderr).fileno())
    except (OSError, ValueError, AttributeError):
        return os.terminal_size((WIDTH, HEIGHT))


_ANSI = re.compile(r"\x1b\[[0-9;]*m|\x1b\]8;;[^\x1b]*\x1b\\")


def _width(character: str) -> int:
    if unicodedata.combining(character) or unicodedata.category(character) == "Cf":
        return 0
    return 2 if unicodedata.east_asian_width(character) in ("W", "F") else 1


def visible(text: str) -> int:
    """The columns text takes on a terminal, ANSI codes aside."""
    return sum(_width(character) for character in _ANSI.sub("", text))


def pad(text: str, width: int) -> str:
    """Text padded with spaces to `width` printed columns."""
    return text + " " * max(width - visible(text), 0)


def clip(text: str, width: int) -> str:
    """Text cut to at most `width` printed columns, its ANSI codes kept whole."""
    shown, used, at = [], 0, 0
    for match in _ANSI.finditer(text):
        for character in text[at : match.start()]:
            used += _width(character)
            if used > width:
                return "".join(shown) + ("\x1b[0m" if "\x1b[" in text else "")
            shown.append(character)
        shown.append(match.group(0))
        at = match.end()
    for character in text[at:]:
        used += _width(character)
        if used > width:
            return "".join(shown) + ("\x1b[0m" if "\x1b[" in text else "")
        shown.append(character)
    return "".join(shown)


def suggest(word: str, choices: Iterable[str]) -> str | None:
    """The closest choice to a typo, or None when nothing is close."""
    import difflib

    matches = difflib.get_close_matches(word, list(choices), n=1, cutoff=0.6)
    return matches[0] if matches else None


@dataclass(frozen=True)
class Pointer:
    """Where in a file an error points, for the `-->` error shape."""

    path: str
    line: int
    column: int
    source: str
    label: str = ""
    width: int = 1


class Aborted(Error):
    """The user declined, pressed Escape or left a question unanswered; nothing changed."""

    code = "Aborted"


@dataclass(frozen=True)
class Option:
    """One row of a selector: the value, the shown text and a quiet detail."""

    value: Any
    text: str
    detail: str = ""
    shown: str = ""


class Output:
    """Writes the CLI's human output to `out` (results) and `err` (everything else)."""

    def __init__(
        self,
        *,
        quiet: bool = False,
        verbose: bool = False,
        out: IO[str] | None = None,
        err: IO[str] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Binds the renderer to its streams and colour setting."""
        self.quiet, self.verbose = quiet, verbose
        self.out = out if out is not None else sys.stdout
        self.err = err if err is not None else sys.stderr
        self._environ = environ
        self._color = {
            id(self.out): color_enabled(self.out, environ),
            id(self.err): color_enabled(self.err, environ),
        }

    # -- primitives --------------------------------------------------------

    def colored(self, stream: IO[str] | None = None) -> bool:
        """Whether a stream (default stderr) gets colour."""
        stream = self.err if stream is None else stream
        return self._color.get(id(stream), color_enabled(stream, self._environ))

    def style(self, text: str, *styles: str, stream: IO[str] | None = None) -> str:
        """Wraps text in ANSI styles when the stream is coloured."""
        if not text or not styles or not self.colored(stream):
            return text
        codes = ";".join(_CODES[name] for name in styles)
        return f"\x1b[{codes}m{text}\x1b[0m"

    def quiet_text(self, text: str, stream: IO[str] | None = None) -> str:
        """Text in the quiet colour."""
        return self.style(text, "dim", stream=stream)

    def _write(self, text: str, stream: IO[str]) -> None:
        stream.write(text + "\n")
        stream.flush()

    def line(self, text: str = "") -> None:
        """One line of a result on stdout; only `generalforecast.render` calls it."""
        self._write(text, self.out)

    def say(self, text: str = "", *, force: bool = False) -> None:
        """One line of commentary on stderr, dropped under `-q` unless forced."""
        if force or not self.quiet:
            self._write(text, self.err)

    def debug(self, text: str) -> None:
        """A quiet line shown only under `-v`."""
        if self.verbose:
            self._write(self.quiet_text(text), self.err)

    def link(self, url: str, text: str | None = None) -> str:
        """A clickable URL on terminals that support OSC 8 hyperlinks."""
        shown = text or url
        if not self.colored():
            return shown
        return f"\x1b]8;;{url}\x1b\\{shown}\x1b]8;;\x1b\\"

    # -- status ------------------------------------------------------------

    def status(self, verb: str, message: str, *, seconds: float | None = None) -> None:
        """A status line: the verb right-aligned in 12 columns, a space, the message.

        A duration ends the line as `in 0.21s`. Dropped under `-q`.
        """
        if self.quiet:
            return
        head = self.style(verb.rjust(VERB_WIDTH), "bold", "green")
        tail = "" if seconds is None else f" in {globals()['seconds'](seconds)}"
        self._write(f"{head} {message}{tail}", self.err)

    def verb(self, verb: str, text: str, detail: str | None = None) -> None:
        """`status` with an optional two-space-separated detail."""
        self.status(verb, text if detail is None else f"{text}  {detail}")

    def changes(self, items: Sequence[tuple[str, str, Any, Any]]) -> None:
        """Lock change lines, keys padded to the longest key in the block."""
        if not items or self.quiet:
            return
        width = max(LOCK_KEY_WIDTH, *(len(key) for _, key, _, _ in items))
        for op, key, old, new in items:
            self.change(op, key, old, new, width=width)

    def change(
        self, op: str, key: str, old: Any, new: Any, *, width: int | None = None
    ) -> None:
        """One lock change: ` + key  value`, ` - key  value` or ` ~ key  old → new`."""
        if self.quiet:
            return
        colour = {"+": "green", "-": "red", "~": "yellow"}.get(op, "bold")
        shown = key.ljust(width or len(key))
        if op == "+" or old is None:
            value = f"{new}"
        elif op == "-" or new is None:
            value = f"{old}"
        else:
            value = f"{old} → {new}"
        self._write(f" {self.style(op, 'bold', colour)} {shown}  {value}", self.err)

    def collapsed(self, label: str, value: str) -> None:
        """A finished prompt: the label padded to column 14, then the value."""
        self._write(f"{label.ljust(LABEL_WIDTH)}{value}", self.err)

    # -- errors ------------------------------------------------------------

    def _labelled(self, label: str, text: str, *styles: str) -> None:
        head = self.style(f"{label}:", *styles)
        indent = " " * (2 + len(label) + 2)
        lines = text.split("\n")
        self._write(f"  {head} {lines[0]}", self.err)
        for rest in lines[1:]:
            self._write(f"{indent}{rest}", self.err)

    def error(
        self,
        message: str,
        *,
        hint: str | None = None,
        because: str | None = None,
        tip: str | None = None,
        pointer: Pointer | None = None,
        request_id: str | None = None,
    ) -> None:
        """An error in one of the three shapes.

        Shape one is `error:` plus `hint:`. Shape two adds a `-->` pointer into a
        file between them. Shape three adds a `because:` line before the hint.
        `tip:` names a near-miss on a flag or a name. The request id, when there
        is one, is the last line.
        """
        head = self.style("error:", "bold", "red")
        lines = message.split("\n")
        self._write(f"{head} {lines[0]}", self.err)
        for rest in lines[1:]:
            self._write(f"       {rest}", self.err)
        if pointer is not None:
            self.pointer(pointer)
        if because:
            self._labelled("because", because, "bold")
        if hint:
            self._labelled("hint", hint, "bold")
        if tip:
            self._labelled("tip", tip, "bold")
        if request_id:
            self._write(self.quiet_text(f"  request {request_id}"), self.err)

    def pointer(self, where: Pointer) -> None:
        """The `-->` block: location, a bar, the source line and a caret line."""
        number = str(where.line)
        gutter = " " * (len(number) + 1)
        quiet = self.quiet_text
        self._write(
            f"{gutter}{quiet('-->')} {where.path}:{where.line}:{where.column}", self.err
        )
        self._write(f"{gutter} {quiet('|')}", self.err)
        self._write(
            f"{quiet(number.rjust(len(number) + 1))} {quiet('|')} {where.source}",
            self.err,
        )
        carets = self.style("^" * max(where.width, 1), "bold", "red")
        label = f" {where.label}" if where.label else ""
        lead = " " * visible(where.source[: max(where.column - 1, 0)])
        self._write(f"{gutter} {quiet('|')} {lead}{carets}{label}", self.err)

    def warning(self, text: str) -> None:
        """A `warning:` line."""
        self._write(f"{self.style('warning:', 'bold', 'yellow')} {text}", self.err)

    def hint(self, text: str) -> None:
        """A `hint:` line on its own."""
        self._labelled("hint", text, "bold")

    # -- tables ------------------------------------------------------------

    def table(
        self,
        rows: Sequence[Sequence[str]],
        *,
        indent: int = 2,
        gap: int = 2,
        align: Sequence[str] | None = None,
        header: Sequence[str] | None = None,
        stream: IO[str] | None = None,
        force: bool = False,
    ) -> None:
        """Rows of cells, each column padded to its widest cell.

        `align` gives `<` or `>` per column (default left). A header row is
        aligned like the values beneath it. A table on stderr is commentary,
        dropped under `-q` unless forced; one on stdout is a result.
        """
        stream = self.err if stream is None else stream
        if stream is self.err and self.quiet and not force:
            return
        every = [list(header), *map(list, rows)] if header else list(map(list, rows))
        if not every:
            return
        ncol = max(len(row) for row in every)
        widths = [
            max(visible(row[i]) if i < len(row) else 0 for row in every)
            for i in range(ncol)
        ]
        kinds = list(align or []) + ["<"] * (ncol - len(align or []))
        for row in every:
            cells = []
            for i in range(ncol):
                cell = row[i] if i < len(row) else ""
                fill = " " * (widths[i] - visible(cell))
                cells.append(fill + cell if kinds[i] == ">" else cell + fill)
            self._write(" " * indent + (" " * gap).join(cells).rstrip(), stream)

    def fields(self, pairs: Sequence[tuple[str, str]], *, indent: int = 2) -> None:
        """A two-column key/value table on stdout."""
        self.table(pairs, indent=indent, stream=self.out)


class Prompter:
    """Asks `gf init`'s questions one at a time, and asks to confirm a change.

    On a terminal the prompt is a selector drawn in raw mode: `>` marks the
    cursor, up and down move it, space toggles a multi-select row, Enter
    confirms and Escape aborts. Without a terminal the prompt falls back to
    numbered lines read from stdin. `yes` takes the first option everywhere
    and `no_input` raises instead of asking.
    """

    def __init__(
        self,
        out: Output,
        *,
        stdin: IO[str] | None = None,
        yes: bool = False,
        no_input: bool = False,
        interactive: bool | None = None,
    ) -> None:
        """Binds the prompter to the renderer and the input stream."""
        self.out = out
        self.stdin = stdin if stdin is not None else sys.stdin
        self.yes, self.no_input = yes, no_input
        if interactive is None:
            interactive = (
                getattr(self.stdin, "isatty", lambda: False)()
                and getattr(out.err, "isatty", lambda: False)()
                and os.name == "posix"
            )
        self.interactive = interactive and not yes and not no_input

    def _unavailable(self, label: str, flag: str) -> Error:
        return InvalidInput(
            f"{label.lower()} could not be inferred and no terminal is attached",
            hint=f"pass '{flag}', or run without --no-input in a terminal",
            param=flag.lstrip("-"),
        )

    def choose(
        self, label: str, options: Sequence[Option], *, flag: str, default: int = 0
    ) -> Any:
        """One value from `options`, with the row at `default` pre-selected."""
        if not options:
            raise InvalidInput(
                f"no column can be the {label.lower()}",
                hint=f"pass '{flag}' with a column of the table",
                param=flag.lstrip("-"),
            )
        if self.yes:
            chosen = options[default]
        elif not self.interactive:
            if self.no_input:
                raise self._unavailable(label, flag)
            chosen = self._plain(label, options, default)
        else:
            chosen = options[self._raw(label, options, default, multi=None)[0]]
        self.out.collapsed(label, chosen.shown or chosen.text)
        return chosen.value

    def choose_many(
        self,
        label: str,
        options: Sequence[Option],
        *,
        flag: str,
        selected: Iterable[int] = (),
    ) -> list[Any]:
        """Several values from `options`, with `selected` rows toggled on first.

        With no options there is nothing to ask, and the answer is none.
        """
        picked = set(selected)
        if not options:
            indexes: list[int] = []
        elif self.yes:
            indexes = sorted(picked)
        elif not self.interactive:
            if self.no_input:
                raise self._unavailable(label, flag)
            indexes = self._plain_many(label, options, picked)
        else:
            indexes = self._raw(label, options, 0, multi=picked)
        values = [options[i] for i in indexes]
        self.out.collapsed(label, ", ".join(option.text for option in values) or "none")
        return [option.value for option in values]

    def ask(
        self,
        label: str,
        default: str,
        *,
        flag: str,
        note: Callable[[str], str] | None = None,
        check: Callable[[str], str | None] | None = None,
    ) -> str:
        """One typed line, pre-filled with `default`.

        `check` names what is wrong with a value, or returns None for a good
        one: a terminal shows the problem beside the line and keeps editing,
        and typed input raises it. `note` is a quiet aside on a good value.
        """

        def problem(value: str) -> str | None:
            return check(value) if check is not None else None

        def aside(value: str) -> str:
            return note(value) if note is not None and problem(value) is None else ""

        if self.yes:
            value = default
        elif not self.interactive:
            if self.no_input:
                raise self._unavailable(label, flag)
            value = self._answer(
                f"{label} [{default}]: ",
                Aborted(
                    f"{label.lower()} was not answered",
                    hint=f"pass '{flag}', or answer on stdin",
                ),
            )
            value = value or default
            wrong = problem(value)
            if wrong is not None:
                raise InvalidInput(
                    f"{label.lower()} {wrong}",
                    hint=f"pass '{flag}'",
                    param=label.lower(),
                )
        else:
            value = self._edit(label, default, aside, problem)
        self.out.collapsed(label, f"{value}  {aside(value)}".rstrip())
        return value

    def confirm(self, question: str, *, flag: str = "--yes") -> None:
        """Asks `question [y/N]` on stderr; raises `Aborted` unless the answer is yes."""
        if self.yes:
            return
        unanswered = Aborted(
            f"`{question}` needs an answer",
            hint=f"pass {flag} to go ahead without asking",
        )
        if self.no_input:
            raise unanswered
        answer = self._answer(f"{question} [y/N] ", unanswered)
        if answer.lower() not in ("y", "yes"):
            raise Aborted("cancelled; nothing was changed")

    # -- plain fallback ----------------------------------------------------

    def _answer(self, prompt: str, unanswered: Aborted) -> str:
        """One line read from stdin after `prompt`; `unanswered` when stdin ends.

        The prompt's line is ended either way, so what follows starts a line.
        """
        self.out.err.write(prompt)
        self.out.err.flush()
        read = self.stdin.readline()
        if not read:
            self.out.err.write("\n")
            self.out.err.flush()
            raise unanswered
        return read.strip()

    def _plain(self, label: str, options: Sequence[Option], default: int) -> Option:
        err = self.out.err
        err.write(f"{label}\n")
        for i, option in enumerate(options, 1):
            err.write(
                f"  {i}. {option.text}{'  ' + option.detail if option.detail else ''}\n"
            )
        text = self._answer(
            f"Choose [{default + 1}]: ",
            Aborted(
                f"{label.lower()} was not chosen", hint="answer on stdin, or pass --yes"
            ),
        )
        if not text:
            return options[default]
        if text.isdigit() and 1 <= int(text) <= len(options):
            return options[int(text) - 1]
        for option in options:
            if option.text == text:
                return option
        raise InvalidInput(f"`{text}` is not one of the {label.lower()} choices")

    def _plain_many(
        self, label: str, options: Sequence[Option], picked: set[int]
    ) -> list[int]:
        err = self.out.err
        err.write(f"{label}\n")
        for i, option in enumerate(options, 1):
            mark = "x" if i - 1 in picked else " "
            err.write(f"  {i}. [{mark}] {option.text}\n")
        text = self._answer(
            "Choose (comma-separated numbers or names, Enter keeps the marked rows): ",
            Aborted(
                f"{label.lower()} were not chosen",
                hint="answer on stdin, or pass --yes",
            ),
        )
        if not text:
            return sorted(picked)
        names = {option.text: i for i, option in enumerate(options)}
        chosen = []
        for part in [p.strip() for p in text.split(",") if p.strip()]:
            if part.isdigit() and 1 <= int(part) <= len(options):
                chosen.append(int(part) - 1)
            elif part in names:
                chosen.append(names[part])
            else:
                raise InvalidInput(
                    f"`{part}` is not one of the {label.lower()} choices"
                )
        return sorted(set(chosen))

    # -- raw terminal ------------------------------------------------------

    def _byte(self, fd: int, *, wait: bool) -> bytes:
        if not wait and not select.select([fd], [], [], ESCAPE_SECONDS)[0]:
            return b""
        return os.read(fd, 1)

    def _key(self) -> str:
        """One key: a character, `up`, `down`, `escape` or `other`.

        A character is read whole, however many UTF-8 bytes it takes. An
        escape sequence is read to its final byte, so keys such as Delete or
        Home leave nothing behind.
        """
        fd = self.stdin.fileno()
        first = self._byte(fd, wait=True)
        if first != b"\x1b":
            lead = first[0] if first else 0
            extra = (
                3 if lead >= 0xF0 else 2 if lead >= 0xE0 else 1 if lead >= 0xC0 else 0
            )
            data = first
            for _ in range(extra):
                data += self._byte(fd, wait=False)
            return data.decode("utf-8", "replace")
        kind = self._byte(fd, wait=False)
        if not kind:
            return "escape"
        sequence = kind
        if kind in (b"[", b"O"):
            while True:
                byte = self._byte(fd, wait=False)
                sequence += byte
                if not byte or 0x40 <= byte[0] <= 0x7E:
                    break
        return {b"[A": "up", b"OA": "up", b"[B": "down", b"OB": "down"}.get(
            sequence, "other"
        )

    def _raw_session(
        self, draw: Callable[[int, int], list[str]], handle: Callable[[str], bool]
    ) -> None:
        import termios
        import tty

        fd = self.stdin.fileno()
        saved = termios.tcgetattr(fd)
        err = self.out.err
        try:
            tty.setcbreak(fd)
            while True:
                columns, lines = size(err)
                # Every line fits the width, so the cursor climbs exactly
                # as many lines as the frame took.
                frame = [
                    clip(line, max(columns - 1, 1)) for line in draw(columns, lines)
                ]
                err.write("".join(f"{line}\n" for line in frame) + FRAME_END)
                err.flush()
                key = self._key()
                err.write(f"\x1b[{len(frame)}A\r\x1b[J" if frame else "\r\x1b[J")
                if key == "\x03":
                    raise KeyboardInterrupt
                if key in ("escape", "\x04"):
                    raise Aborted("cancelled; nothing was written")
                if handle(key):
                    break
        finally:
            err.write("\x1b[?25h")
            err.flush()
            termios.tcsetattr(fd, termios.TCSADRAIN, saved)

    def _raw(
        self,
        label: str,
        options: Sequence[Option],
        default: int,
        *,
        multi: set[int] | None,
    ) -> list[int]:
        width = max(visible(option.text) for option in options)
        state = {"cursor": default}

        def draw(columns: int, lines: int) -> list[str]:
            # The rows that fit under the label, scrolled to keep the cursor in view.
            room = max(lines - 2, 1)
            first = min(
                max(state["cursor"] - room // 2, 0), max(len(options) - room, 0)
            )
            frame = [label]
            for i in range(first, min(first + room, len(options))):
                option = options[i]
                cursor = ">" if i == state["cursor"] else " "
                mark = "" if multi is None else ("[x] " if i in multi else "[ ] ")
                text = pad(f"{mark}{option.text}", width + len(mark))
                detail = (
                    f"    {self.out.quiet_text(option.detail)}" if option.detail else ""
                )
                frame.append(f"  {cursor} {text}{detail}")
            return frame

        def handle(key: str) -> bool:
            if key in ("up", "k"):
                state["cursor"] = (state["cursor"] - 1) % len(options)
            elif key in ("down", "j", "\t"):
                state["cursor"] = (state["cursor"] + 1) % len(options)
            elif key == " " and multi is not None:
                multi.symmetric_difference_update({state["cursor"]})
            elif key in ("\r", "\n"):
                return True
            return False

        self._raw_session(draw, handle)
        return sorted(multi) if multi is not None else [state["cursor"]]

    def _edit(
        self,
        label: str,
        default: str,
        aside: Callable[[str], str],
        problem: Callable[[str], str | None],
    ) -> str:
        state = {"text": default}

        def draw(columns: int, lines: int) -> list[str]:
            wrong = problem(state["text"])
            shown = (
                self.out.style(wrong, "red", stream=self.out.err)
                if wrong is not None
                else self.out.quiet_text(aside(state["text"]))
            )
            tail = f"  {shown}" if shown else ""
            return [f"{label.ljust(LABEL_WIDTH)}{state['text']}{tail}"]

        def handle(key: str) -> bool:
            if key in ("\r", "\n"):
                return problem(state["text"]) is None
            if key in ("\x7f", "\b"):
                state["text"] = state["text"][:-1]
            elif len(key) == 1 and key.isprintable():
                state["text"] += key
            return False

        self._raw_session(draw, handle)
        return state["text"]
