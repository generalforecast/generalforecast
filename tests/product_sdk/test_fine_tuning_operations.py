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
    prepared = ops.prepare("acme", ops.Chunk("up-1", 8, b"abcd", 20))
    assert prepared.method == "POST"
    assert prepared.path == "/v1/projects/acme/uploads/up-1/bytes"
    assert prepared.body == b"abcd"
    assert prepared.headers["Content-Range"] == "bytes 8-11/20"
    assert prepared.headers["Content-Type"] == "application/octet-stream"
    assert prepared.request.resource.kind == "corpus"


def test_an_upload_the_service_takes_is_sent_in_chunks_then_completed():
    body = ops.parquet_bytes(TABLE)
    opened = {"id": "up-1", "url": None, "chunk_bytes": 1024, "state": "open"}
    chunks = -(-len(body) // 1024)
    replies = ["run", opened, *[{}] * chunks, {"state": "verified", "digest": "d"}]
    steps, result = _drive(
        ops.training_upload(
            TABLE, {"target": "y", "task": "classification"}, wait=True
        ),
        replies,
    )
    sent = [s for s in steps if isinstance(s, ops.Chunk)]
    assert len(sent) == chunks and b"".join(s.data for s in sent) == body
    assert [s.start for s in sent] == list(range(0, len(body), 1024))
    complete = steps[-1]
    assert complete.path == "/uploads/up-1/complete"
    assert complete.payload == {"metadata": {"target": "y", "task": "classification"}}
    assert result["state"] == "verified"


def test_an_upload_to_storage_puts_its_chunks_at_the_url():
    opened = {"id": "up-1", "url": "https://storage/session", "chunk_bytes": 1 << 20}
    replies = ["run", opened, 200, {"state": "verifying"}, {"state": "verified"}]
    steps, result = _drive(
        ops.training_upload(TABLE, {"target": "y", "task": "regression"}, wait=True),
        replies,
    )
    puts = [s for s in steps if isinstance(s, ops.Work) and s.function is ops.put_chunk]
    assert len(puts) == 1 and puts[0].args[0] == "https://storage/session"
    # It waited once for verification.
    assert any(isinstance(s, ops.Pause) for s in steps)
    assert result["state"] == "verified"


def test_a_rejected_upload_raises_its_reason():
    opened = {"id": "up-1", "url": None, "chunk_bytes": 1 << 20}
    rejected = {"state": "rejected", "error": "the table has no column y"}
    with pytest.raises(ServiceError, match="no column y"):
        _drive(
            ops.training_upload(
                TABLE, {"target": "y", "task": "regression"}, wait=True
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
    assert ops.run_spec("r1", seed=None, plan={"phases": 5}) == {
        "name": "r1",
        "plan": {"phases": 5},
    }
