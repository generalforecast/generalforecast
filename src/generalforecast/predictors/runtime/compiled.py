# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A compiled plan: an execution plan bound to its model pin and observed schema.

`compile_plan` and `compile_lock` build one in memory; it is never written,
sent or stored. `gf.lock` is the only document a predictor has, and a
compiled plan names the lock it came from by `lock_digest`, the one digest
revision identity, `gf serve` and gf-server's fingerprint all use.
"""

from dataclasses import dataclass, replace
from typing import Literal

from generalforecast.predictors.models import CatalogResolver, ModelPin, ModelResolver
from generalforecast.predictors.runtime.plan import ExecutionPlan
from generalforecast.predictors.runtime.schema import TableSchema
from generalforecast.predictors.validation import digest, ensure


@dataclass(frozen=True)
class CompiledPlan:
    """A plan, its pinned model, its schema and their digests.

    Attributes:
        spec: The execution plan.
        model: The pinned model and its capabilities, from the trusted catalog.
        task: The compiled task.
        schema: The physical schema the context rows were observed to have.
        source_digest: The digest of the plan.
        data_digest: The bound snapshot rows' digest, for a snapshot context.
        pin_digest: The digest of the model pin's identity.
        plan_digest: The digest of all of the above; a cache key, never an
            identity a user or a revision is known by.
        outputs: The outputs predictions carry.
        context_rows: The bound snapshot rows.
        cpu: The CPUs a worker requests.
        memory: The memory a worker requests.
        lock_digest: The digest of the `gf.lock` compiled, if one was.
    """

    spec: ExecutionPlan
    model: ModelPin
    task: str
    schema: TableSchema
    source_digest: str
    data_digest: str | None
    pin_digest: str
    plan_digest: str
    outputs: tuple[str, ...]
    context_rows: int | None
    cpu: int
    memory: str
    lock_digest: str | None = None

    @property
    def kind(self) -> Literal["catalog"]:
        """A catalog model's plan."""
        return "catalog"

    @property
    def identity(self) -> str:
        """The admitted model's content-addressed reference."""
        return self.model.resolved

    def verify_model(
        self, resolver: ModelResolver | None = None, *, offline: bool = True
    ) -> ModelPin:
        """Authenticates the pin's capabilities against the admitted catalog.

        The model's identity must equal the catalog's, and the serving policy
        the plan was compiled under must stay within what the catalog admits
        now.
        """
        trusted = (resolver or CatalogResolver()).resolve(
            self.model.resolved, offline=offline
        )
        locked = self.model
        ensure(
            trusted.identity() == locked.identity()
            and locked.memory == trusted.memory
            and all(
                (getattr(locked, key) is None) == (getattr(trusted, key) is None)
                and (getattr(locked, key) or 0) <= (getattr(trusted, key) or 0)
                for key in ("max_context_rows", "max_features", "max_horizon", "cpu")
            ),
            "locked model capabilities differ from admitted catalog",
            "ModelMismatch",
        )
        return trusted

    def admit(
        self, resolver: ModelResolver | None = None, *, offline: bool = True
    ) -> ModelPin:
        """Verifies the model, then the shared capability and policy admission."""
        from generalforecast.predictors.runtime.capabilities import admit

        trusted = self.verify_model(resolver, offline=offline)
        admit(self.spec, self.model, self.task, self.outputs)
        return trusted

    def to_dict(self) -> dict:
        """The plan's content, as `plan_digest` hashes it."""
        return {
            "source_digest": self.source_digest,
            "data_digest": self.data_digest,
            "pin_digest": self.pin_digest,
            "spec": self.spec.to_dict(),
            "model": self.model.to_dict(),
            "task": self.task,
            "schema": self.schema.to_dict(),
            "outputs": list(self.outputs),
            "context_rows": self.context_rows,
            "compute": {"cpu": self.cpu, "memory": self.memory},
            "lock_digest": self.lock_digest,
        }

    @classmethod
    def build(
        cls,
        *,
        spec: ExecutionPlan,
        model: ModelPin,
        task: str,
        schema: TableSchema,
        data_digest: str | None,
        outputs: tuple[str, ...],
        context_rows: int | None,
        lock_digest: str | None = None,
    ) -> "CompiledPlan":
        """Computes the plan, pin and content digests."""
        draft = cls(
            spec,
            model,
            task,
            schema,
            digest({"spec": spec.to_dict()}),
            data_digest,
            digest(model.identity()),
            "",
            outputs,
            context_rows,
            model.cpu,
            model.memory,
            lock_digest,
        )
        return replace(draft, plan_digest=digest(draft.to_dict()))


__all__ = ["CompiledPlan"]
