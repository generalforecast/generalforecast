# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pages: the row representations a table's fit and embeds make, kept in host slots and joined for encodes."""

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from generalforecast._core import CacheKey
from generalforecast.worker.records import Fit, Page

if TYPE_CHECKING:
    from generalforecast.worker.runner import Worker

logger = logging.getLogger(__name__)


# A page's row representations, on the device or the host, and its targets.
Representations = tuple[Any, np.ndarray]


@dataclass
class _Page:
    """Where a page's representations are in host slots, and its targets."""

    slots: list[int]
    shape: tuple[int, ...]
    dtype: np.dtype
    targets: np.ndarray


class Pages:
    """How a worker fits tables and embeds, keeps and reads their pages."""

    def _fit(
        self: "Worker", fits: Sequence[Fit], pages: Sequence[Page]
    ) -> dict[int, Representations]:
        """Fits tables, and splits the rows of fits that fill pages into them."""
        work = [
            (
                fit.table,
                (
                    *self._padded(fit.features, fit.targets),
                    self._tables[fit.table].categorical,
                ),
            )
            for fit in fits
        ]
        # A fit filling pages embeds its rows in the call that fits its state;
        # one past the rows a call takes computes the state alone, and each
        # page is embedded once a plan reads it.
        fused = [index for index, fit in enumerate(fits) if fit.pages]
        alone = [index for index, fit in enumerate(fits) if not fit.pages]
        for index, state in zip(
            alone, self._call("state", [work[index] for index in alone]), strict=True
        ):
            self._tables[fits[index].table].state = state
            self._warn_crowded(fits[index].table, state)
            self._warm(fits[index])
        filled: dict[int, Representations] = {}
        for index, (state, representations) in zip(
            fused, self._call("fit", [work[index] for index in fused]), strict=True
        ):
            fit = fits[index]
            self._tables[fit.table].state = state
            self._warn_crowded(fit.table, state)
            self._warm(fit)
            # Split on the host: slicing on the device compiles once per page
            # length, and pages change length with every write.
            with self.stats.timed("fetch"):
                on_host = np.asarray(representations)
            offset = 0
            for index in fit.pages:
                count = len(pages[index].rows)
                targets = np.asarray(fit.targets[offset : offset + count])
                filled[index] = (on_host[offset : offset + count], targets)
                self._keep(pages[index], filled[index])
                offset += count
        return filled

    def _warn_crowded(self: "Worker", table: int, state: Any) -> None:
        """Logs once per table that categorical columns held more categories than kept.

        Args:
            table: The table a state was fit for, on the whole table or a unit.
            state: The fitted state.
        """
        entry = self._tables[table]
        count = self._models[entry.model].functions.crowded_categories
        if entry.crowded or entry.categorical is None or count is None:
            return
        crowded = count(state)
        if crowded:
            entry.crowded = True
            columns = "column holds" if crowded == 1 else "columns hold"
            logger.warning(
                "table %d: %d categorical %s more categories than the model "
                "keeps, so their ranks are interpolated",
                table,
                crowded,
                columns,
            )

    def _keep(self: "Worker", page: Page, representations: Representations) -> None:
        """Caches a computed page in the host slots the node kept for it, if any."""
        if page.action != "encode":
            return
        values, targets = representations
        # Targets can be views of the whole frame the step came in; a copy keeps
        # the cached page from holding that frame, and its shared memory, alive.
        targets = np.array(targets, copy=True)
        data = np.ascontiguousarray(values).view(np.uint8).reshape(-1)
        capacity = len(page.slots) * self._slot_bytes
        if data.size > capacity:
            raise RuntimeError(
                f"a page takes {data.size} bytes, but the node kept {capacity}"
            )
        # Only the page's bytes: a page may fill a fraction of a large slot.
        with self.stats.timed("store"):
            self._host.write_bytes(list(page.slots), data)
        self._pages[page.key] = _Page(
            list(page.slots), values.shape, values.dtype, targets
        )

    def _cached(self: "Worker", key: CacheKey) -> Representations:
        """A cached page's representations and targets, read from its host slots."""
        page = self._pages[key]
        size = int(np.prod(page.shape)) * page.dtype.itemsize
        data = self._host.read_bytes(page.slots, size)
        return data.view(page.dtype).reshape(page.shape), page.targets

    def _embed_pages(
        self: "Worker", pages: Sequence[Page], filled: dict[int, Representations]
    ) -> dict[int, Representations]:
        """Reads cached pages and embeds the rest, one vectorized call per signature.

        A page with sources copies their rows, representations and targets,
        and embeds only its own. Those pad to their own bucket, not the page
        size: a page written a row at a time then embeds a few rows, or none.
        """
        found: dict[int, Representations] = {}
        work: list[tuple[int, tuple]] = []
        embedding: list[tuple[int, list[int]]] = []
        partial: dict[int, Representations] = {}
        for index, page in enumerate(pages):
            if index in filled:
                continue
            if page.action == "reuse":
                found[index] = self._cached(page.key)
                continue
            if page.features is None or page.targets is None:
                raise RuntimeError(
                    f"a page of table {page.table} to embed came without its rows"
                )
            missing = list(range(len(page.rows)))
            if page.copies:
                partial[index] = self._copied(page)
                copied = {to for _, positions in page.copies for _, to in positions}
                missing = [at for at in missing if at not in copied]
            if not missing:
                continue
            at_least = 1 if page.copies else self._tables[page.table].page_size
            features, targets, _ = self._padded(page.features, page.targets, at_least)
            work.append(
                (page.table, (self._tables[page.table].state, features, targets))
            )
            embedding.append((index, missing))
        for (index, missing), representations in zip(
            embedding, self._call("embed", work), strict=True
        ):
            page = pages[index]
            with self.stats.timed("fetch"):
                embedded = np.asarray(representations)[: len(missing)]
            if index in partial:
                values, targets = partial[index]
                values[missing] = embedded
                targets[missing] = np.asarray(page.targets)
            else:
                partial[index] = (embedded, np.asarray(page.targets))
            self.stats.rows_embedded += len(missing)
        for index, computed in partial.items():
            self._keep(pages[index], computed)
            found[index] = computed
        return found

    def _copied(self: "Worker", page: Page) -> Representations:
        """A page's representations and targets with its sources' rows copied in."""
        values: np.ndarray | None = None
        targets: np.ndarray | None = None
        for source, positions in page.copies:
            held, labels = self._cached(source)
            if values is None or targets is None:
                values = np.empty((len(page.rows), *held.shape[1:]), held.dtype)
                targets = np.empty((len(page.rows), *labels.shape[1:]), labels.dtype)
            at, to = (list(part) for part in zip(*positions, strict=True))
            values[to] = held[at]
            targets[to] = labels[at]
            self.stats.rows_copied += len(positions)
        if values is None or targets is None:
            raise RuntimeError(f"a page of table {page.table} copies from no source")
        return values, targets

    @staticmethod
    def _tables_rows(
        pages: Sequence[Page], computed: dict[int, Representations]
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Each table's page rows sorted, with their representations and targets.

        Built once per step, so units gather from them without joining pages
        again. They join on the host: on the device, every new number of rows
        would compile its own join and gather, and writes change it each time.
        """
        by_table: dict[int, list[int]] = defaultdict(list)
        for index, page in enumerate(pages):
            by_table[page.table].append(index)
        joined = {}
        for table, indices in by_table.items():
            rows = np.sort(
                np.concatenate([np.asarray(pages[index].rows) for index in indices])
            )
            # Each page's rows go straight to their place, so the step holds
            # one copy of the representations beside the pages'.
            first_values, first_targets = (
                np.asarray(part) for part in computed[indices[0]]
            )
            values = np.empty((len(rows), *first_values.shape[1:]), first_values.dtype)
            targets = np.empty(
                (len(rows), *first_targets.shape[1:]), first_targets.dtype
            )
            for index in indices:
                at = np.searchsorted(rows, np.asarray(pages[index].rows))
                values[at], targets[at] = computed[index]
            joined[table] = (rows, values, targets)
        return joined
