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
from generalforecast.runtime.device import HOST_MEMORY, available_memory
from generalforecast.worker.records import Fit, Page, Unit

if TYPE_CHECKING:
    from generalforecast.worker.runner import Worker

logger = logging.getLogger(__name__)


# A page's row representations, on the device or the host, and its targets.
Representations = tuple[Any, np.ndarray]

# A put-off page whose rows the step's units read fewer than this share of
# embeds only those: a context drawn across a large table, such as a core
# every query shares, reads a few rows of each of its pages.
SPARSE_SHARE = 0.5


class HostMemoryError(RuntimeError):
    """A step's row representations would take more host memory than is free."""


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
        # A program that keeps only the state rounds its sums apart from one
        # that also embeds the rows, as a context fit with its queries does.
        # A fit within the rows such a call takes runs that program, and
        # drops the rows it embeds, so its state is the same to the bit.
        whole = [index for index in alone if self._embeds_whole(fits[index])]
        apart = [index for index in alone if index not in whole]
        states = zip(
            whole, self._call("fit", [work[index] for index in whole]), strict=True
        )
        for index, (state, _) in states:
            self._took_state(fits[index], state)
        for index, state in zip(
            apart, self._call("state", [work[index] for index in apart]), strict=True
        ):
            self._took_state(fits[index], state)
        filled: dict[int, Representations] = {}
        for index, (state, representations) in zip(
            fused, self._call("fit", [work[index] for index in fused]), strict=True
        ):
            fit = fits[index]
            self._took_state(fit, state)
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

    def _embeds_whole(self: "Worker", fit: Fit) -> bool:
        """Whether a fit's rows are within those a call that fits its state and embeds them takes.

        Within `fit_bound`, which takes no compile, surely; past it, as the
        measure the server asked for to plan the fit says.
        """
        table = self._tables[fit.table]
        rows = self._size(len(fit.features))
        if rows <= self.fit_bound(table.model, table.features):
            return True
        _, fused_rows = self.fit_rows(table.model, table.features)
        return rows <= fused_rows

    def _took_state(self: "Worker", fit: Fit, state: Any) -> None:
        """Keeps a newly fit state for its table, and compiles its page embeds ahead."""
        self._tables[fit.table].state = state
        self._warn_crowded(fit.table, state)
        self._warm(fit)

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
        self: "Worker",
        pages: Sequence[Page],
        filled: dict[int, Representations],
        units: Sequence[Unit] = (),
        subsets: dict[int, np.ndarray] | None = None,
    ) -> dict[int, Representations]:
        """Reads cached pages and embeds the rest, one vectorized call per signature.

        A page with sources copies their rows, representations and targets,
        and embeds only its own. Those pad to their own bucket, not the page
        size: a page written a row at a time then embeds a few rows, or none.
        So does a put-off page, kept in no slots, of which `units` read fewer
        than `SPARSE_SHARE` of the rows: it embeds only those, and
        `subsets` gets their rows, which its representations then hold.

        Raises:
            HostMemoryError: If the representations the step would hold take
                more host memory than is free, so the worker fails the step
                rather than its container running out.
        """
        sparse = self._sparse_pages(pages, filled, units)
        self._check_host(pages, filled, sparse)
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
            if index in sparse:
                read = sparse[index]
                if subsets is not None:
                    subsets[index] = np.asarray(page.rows)[read]
                features, targets, _ = self._padded(
                    np.asarray(page.features)[read], np.asarray(page.targets)[read]
                )
                work.append(
                    (page.table, (self._tables[page.table].state, features, targets))
                )
                embedding.append((index, read.tolist()))
                continue
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

        # Each page comes to the host as its call finishes, so the device holds
        # one call's pages at a time, not every page the step embeds: a wide
        # table's representations of all its pages can fill the device.
        def fetched(position: int, representations: Any) -> None:
            index, missing = embedding[position]
            page = pages[index]
            with self.stats.timed("fetch"):
                embedded = np.asarray(representations)[: len(missing)]
            if index in sparse:
                partial[index] = (embedded, np.asarray(page.targets)[sparse[index]])
            elif index in partial:
                values, targets = partial[index]
                values[missing] = embedded
                targets[missing] = np.asarray(page.targets)
            else:
                partial[index] = (embedded, np.asarray(page.targets))
            self.stats.rows_embedded += len(missing)

        self._call("embed", work, consume=fetched)
        for index, computed in partial.items():
            self._keep(pages[index], computed)
            found[index] = computed
        return found

    def _sparse_pages(
        self: "Worker",
        pages: Sequence[Page],
        filled: dict[int, Representations],
        units: Sequence[Unit],
    ) -> dict[int, np.ndarray]:
        """The put-off pages of which `units` read fewer than `SPARSE_SHARE` of the rows, with the positions they read.

        Only pages to embed in full count: a kept page is cached whole, and
        one with sources copies most of its rows.
        """
        candidates = [
            index
            for index, page in enumerate(pages)
            if index not in filled
            and page.action == "transient"
            and not page.copies
            and page.features is not None
        ]
        if not candidates:
            return {}
        parts: dict[int, list[np.ndarray]] = defaultdict(list)
        for unit in units:
            if unit.action not in ("reuse", "fuse"):
                parts[unit.table].append(np.asarray(unit.rows))
        read = {table: np.unique(np.concatenate(rows)) for table, rows in parts.items()}
        sparse: dict[int, np.ndarray] = {}
        for index in candidates:
            page = pages[index]
            if page.table not in read:
                continue
            rows = np.asarray(page.rows)
            positions = np.flatnonzero(np.isin(rows, read[page.table]))
            if len(positions) < SPARSE_SHARE * len(rows):
                sparse[index] = positions
        return sparse

    def _check_host(
        self: "Worker",
        pages: Sequence[Page],
        filled: dict[int, Representations],
        sparse: dict[int, np.ndarray],
    ) -> None:
        """Raises `HostMemoryError` if the representations a step would hold take more host memory than is free.

        It holds each page's that it embeds or reads from host slots, once
        (see `_tables_rows`); a fit's pages are already held. Where free
        memory cannot be told, nothing is checked.
        """
        needed: dict[int, int] = defaultdict(int)
        for index, page in enumerate(pages):
            if index in filled:
                continue
            rows = len(sparse[index]) if index in sparse else len(page.rows)
            table = self._tables[page.table]
            needed[page.table] += (
                rows * self.profile(table.model, table.features).row_bytes
            )
        total = sum(needed.values())
        if not total:
            return
        free = available_memory()
        if free is None or total <= free:
            return
        table = max(needed, key=needed.__getitem__)
        raise HostMemoryError(
            f"a step's row representations take {total >> 20} MiB of host "
            f"memory ({needed[table] >> 20} MiB for table {table}), but "
            f"{free >> 20} MiB is free; use a strategy whose contexts read "
            "fewer rows, a host with more memory, or set "
            f"{HOST_MEMORY} if the container's limit is larger than found"
        )

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
        pages: Sequence[Page],
        computed: dict[int, Representations],
        subsets: dict[int, np.ndarray] | None = None,
    ) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Each table's page rows sorted, with their representations and targets.

        Built once per step, so units gather from them without joining pages
        again. They join on the host: on the device, every new number of rows
        would compile its own join and gather, and writes change it each time.
        Each page is taken out of `computed` as it joins, so the step holds
        one copy of its pages' representations, not theirs and the join's. A
        page in `subsets` holds only those of its rows.
        """
        subsets = subsets or {}

        def rows_of(index: int) -> np.ndarray:
            return subsets[index] if index in subsets else np.asarray(pages[index].rows)

        by_table: dict[int, list[int]] = defaultdict(list)
        for index, page in enumerate(pages):
            by_table[page.table].append(index)
        joined = {}
        for table, indices in by_table.items():
            rows = np.sort(np.concatenate([rows_of(index) for index in indices]))
            # Each page's rows go straight to their place, so the step holds
            # one copy of the representations beside the pages'.
            first_values, first_targets = (
                np.asarray(part) for part in computed[indices[0]]
            )
            values = np.empty((len(rows), *first_values.shape[1:]), first_values.dtype)
            targets = np.empty(
                (len(rows), *first_targets.shape[1:]), first_targets.dtype
            )
            del first_values, first_targets
            for index in indices:
                at = np.searchsorted(rows, rows_of(index))
                values[at], targets[at] = computed.pop(index)
            joined[table] = (rows, values, targets)
        return joined
