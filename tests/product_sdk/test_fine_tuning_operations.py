# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The fine-tuning operations' steps, driven without a service."""

import pyarrow as pa
import pytest

from generalforecast import _operations as ops
from generalforecast.service.contracts import ServiceError

TABLE = pa.table({"x": [1.0, 2.0, 3.0], "y": [0, 1, 0]})


def _drive(operation, replies):
    """Runs an operation, answering each step with the next reply; returns the steps.

    A Pause takes no reply. A reply of "run" runs the Work step it answers.
    """
    steps, value = [], None
    try:
        while True:
            step = operation.send(value)
            steps.append(step)
            if isinstance(step, ops.Pause):
                value = None
            elif replies[0] == "run":
                replies.pop(0)
                value = step.function(*step.args, **step.kwargs)
            else:
                value = replies.pop(0)
    except StopIteration as stop:
        return steps, stop.value


def test_a_chunk_is_a_raw_post_with_its_content_range():
    prepared = ops.prepare("acme", ops.Chunk("credit-2026-01", 8, b"abcd", 20))
    assert prepared.method == "POST"
    assert prepared.path == "/v1/projects/acme/snapshots/credit-2026-01/bytes"
    assert prepared.body == b"abcd"
    assert prepared.headers["Content-Range"] == "bytes 8-11/20"
    assert prepared.headers["Content-Type"] == "application/octet-stream"
    assert prepared.request.resource.kind == "snapshot"


def test_an_upload_the_service_takes_is_sent_in_chunks_then_awaited():
    body = ops.parquet_bytes(TABLE)
    opened = {"name": "t", "url": None, "chunk_bytes": 1024, "state": "open"}
    chunks = -(-len(body) // 1024)
    replies = ["run", opened, *[{}] * chunks, {"state": "ready", "digest": "d"}]
    steps, result = _drive(
        ops.training_upload(
            "t", TABLE, {"target": "y", "task": "classification"}, wait=True
        ),
        replies,
    )
    sent = [s for s in steps if isinstance(s, ops.Chunk)]
    assert len(sent) == chunks and b"".join(s.data for s in sent) == body
    assert [s.start for s in sent] == list(range(0, len(body), 1024))
    # The metadata opens the upload, and nothing completes it: the service
    # verifies it once every byte is stored, and the client reads it back.
    opening = steps[1]
    assert opening.path == "/snapshots"
    assert opening.resource == "t"
    assert opening.payload == {
        "name": "t",
        "bytes": len(body),
        "format": "parquet",
        "metadata": {"target": "y", "task": "classification"},
    }
    assert not [s for s in steps if "complete" in getattr(s, "path", "")]
    assert steps[-1].method == "GET" and steps[-1].path == "/snapshots/t"
    assert result["state"] == "ready"


def test_an_arrow_snapshot_is_named_in_its_query():
    prepared = ops.prepare(
        "acme",
        ops.Upload(
            "fraud-5fb6a4c6f24b", b"arrow", {"target": "y", "task": "regression"}
        ),
    )
    assert prepared.path == (
        "/v1/projects/acme/snapshots?name=fraud-5fb6a4c6f24b&target=y&task=regression"
    )
    assert prepared.request.resource.kind == "snapshot"
    assert prepared.request.resource.name == "fraud-5fb6a4c6f24b"
    assert ops.snapshot_name("fraud", "sha256:" + "ab" * 32) == "fraud-abababababab"


def test_an_upload_to_storage_puts_its_chunks_at_the_url():
    opened = {
        "name": "t",
        "url": "https://storage/session",
        "chunk_bytes": 1 << 20,
        "state": "open",
    }
    replies = ["run", opened, 200, {"state": "open"}, {"state": "ready"}]
    steps, result = _drive(
        ops.training_upload(
            "t", TABLE, {"target": "y", "task": "regression"}, wait=True
        ),
        replies,
    )
    puts = [s for s in steps if isinstance(s, ops.Work) and s.function is ops.put_chunk]
    assert len(puts) == 1 and puts[0].args[0] == "https://storage/session"
    # It waited for verification.
    assert sum(isinstance(s, ops.Pause) for s in steps) == 2
    assert result["state"] == "ready"


def test_a_rejected_upload_raises_its_reason():
    opened = {"name": "t", "url": None, "chunk_bytes": 1 << 20, "state": "open"}
    rejected = {"state": "rejected", "error": "the table has no column y"}
    with pytest.raises(ServiceError, match="no column y"):
        _drive(
            ops.training_upload(
                "t", TABLE, {"target": "y", "task": "regression"}, wait=True
            ),
            ["run", opened, {}, rejected],
        )


def test_waiting_for_a_run_times_out():
    with pytest.raises(ServiceError) as waited:
        _drive(
            ops.run_wait("r1", timeout=0, poll=0.0),
            [{"state": "training"}, {"state": "training"}],
        )
    assert waited.value.code == "WaitTimeout"


def test_payloads_leave_out_what_was_not_given():
    assert ops.upload_metadata("y", "regression") == {
        "target": "y",
        "task": "regression",
    }
    assert ops.run_spec("r1", seed=None, plan={"steps": 5}) == {
        "name": "r1",
        "plan": {"steps": 5},
    }


def test_a_plan_is_its_steps_and_the_trainers_settings():
    from generalforecast import _api

    one = _api.Run.FIELDS[[f.key for f in _api.Run.FIELDS].index("plan")]
    plan = {"steps": 200, "learning_rate": 5e-5, "episode_sampling": {"rows": 1024}}
    decoded = one.decode(plan, "plan")
    assert isinstance(decoded, _api.RunPlan) and decoded.steps == 200
    assert decoded.episode_sampling == _api.EpisodeSampling(rows=1024)
    saved = one.decode({"steps": 10, "save_steps": 5, "save_total_limit": 2}, "plan")
    assert (saved.save_steps, saved.save_total_limit) == (5, 2)
    for refused in (
        {"steps": 10, "phases": [{"steps": 1}, {"steps": 1}]},
        {"phases": [{"steps": 1}, {"steps": 1}]},
        {"learning_rate": 1e-4},
        {"steps": 10, "checkpoint_policy": {"every": 5}},
    ):
        with pytest.raises(_api.PayloadError):
            one.decode(refused, "plan")
