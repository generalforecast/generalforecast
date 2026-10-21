# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

from generalforecast._core import bucket_size
from generalforecast.runtime.decoding import Context, Decoding, Group, alike, packed


def _decoding(queries: int, slots: int) -> Decoding:
    """A decoding of `queries` queries against one context of `slots` slots."""
    context = Context(cast(Any, None), slots, list(range(slots)))
    return Decoding(None, list(range(queries)), [context], None)


def test_queries_with_contexts_of_their_own_decode_apart() -> None:
    # Four queries reading a context of their own, then two sharing one.
    groups: list[Group] = [(unit, [unit]) for unit in range(4)]
    groups.append((4, [4, 5]))
    slots = dict.fromkeys(range(5), 4)

    # Each decoding reads at most twice its largest group's slots, not all of
    # them, and no more than the window holds.
    assert packed(groups, slots, window=64) == [groups[:2], groups[2:4], groups[4:]]
    assert packed(groups, slots, window=4) == [[group] for group in groups]


def test_decodings_pad_only_to_their_own_size() -> None:
    many, few, other, wide = (
        _decoding(32, 4),
        _decoding(1, 4),
        _decoding(1, 4),
        _decoding(1, 16),
    )

    groups = alike([many, few, wide, other], 1, lambda count: bucket_size(count))

    # A call pads to its largest decoding, so each size calls apart.
    assert [[id(decoding) for decoding in group] for group in groups] == [
        [id(many)],
        [id(few), id(other)],
        [id(wide)],
    ]
