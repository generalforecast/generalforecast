"""Every built-in model once: artifact identity, intrinsic capability and qualification.

This module is metadata only, standard library alone, so offline tools such as
`gf check` read it without the native core, JAX or a network. Loading lives in
`generalforecast.models.registry`, which verifies bytes against these entries first.

Three things stay apart:

- An `Artifact` is the immutable content a model is made of: a repository at an
  upstream commit, with the SHA-256 of each file it needs, including license
  evidence. Families converted at load name their upstream bytes and how they
  convert, and the converted directory carries its own derived manifest.
- A `Capability` is what the weights can do, whatever a service allows.
- A `ServingPolicy` is what a service allows: request bounds, compute and the
  most `@K` codes and `*N` estimators it runs. Locks freeze the v1 policy so a
  service changing its bounds never changes an old lock's identity.

`Gates` say, per surface, what the evidence supports. Being in the catalog only
qualifies a model for trusted local use; packages and remote serving need
pinned bytes, license evidence and runtime qualification with real weights.
"""

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, overload

from generalforecast.serialization import content_digest

Task = Literal["classification", "regression", "forecast"]
Surface = Literal["local", "package", "remote", "self-hosted"]

# The Hugging Face organization that holds the pinned converted checkpoints.
ORGANIZATION = "general-forecast"
# The most classes a classifier serves through `@K` error-correcting codes;
# `generalforecast.models.many_class.MAX_CLASSES` reads this.
MAX_CLASSES = 1000
# Levels Chronos-2 publishes, the last axis of its forecasts.
CHRONOS2_QUANTILES = (
    0.01,
    0.05,
    *(round(0.1 + 0.05 * step, 2) for step in range(17)),
    0.95,
    0.99,
)
TABULAR_TYPES = ("boolean", "int64", "float64", "string", "date", "timestamp")
SERIES_TYPES = ("int64", "float64")
LICENSE_FILES = ("LICENSE", "NOTICE", "README.md")
# The derived manifest a converted checkpoint directory records itself in.
DERIVED_MANIFEST = "gf-manifest.json"
# Licenses whose terms permit General Forecast to serve the weights to others.
# Timestral's modified MIT caps free use at a monthly revenue; the owner accepted it.
HOSTED_LICENSES = frozenset(
    {"Apache-2.0", "BSD-3-Clause", "MIT", "OpenMDW-1.1", "LicenseRef-timestral-license"}
)


@dataclass(frozen=True)
class Artifact:
    """Immutable content: files hashed at an upstream commit.

    Attributes:
        repository: The Hugging Face repository the bytes come from.
        revision: The commit they are pinned at.
        files: SHA-256 of every file a load reads, including license evidence,
            or empty where no hashes have been recorded; such a model only
            loads for trusted local use.
        license: The license identifier its evidence declares, if any.
        license_source: The file that declares it.
        convert: The architecture its upstream files convert to at load, or
            None if the files load as they are.
    """

    repository: str
    revision: str
    files: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    license: str | None = None
    license_source: str | None = None
    convert: str | None = None

    @property
    def digest(self) -> str | None:
        """The content digest of the file manifest, or None without one."""
        return content_digest(dict(self.files)) if self.files else None

    @property
    def license_files(self) -> tuple[str, ...]:
        """The pinned files that carry license evidence."""
        return tuple(name for name in self.files if name in LICENSE_FILES)


@dataclass(frozen=True)
class Capability:
    """What a model's weights can do, independent of any service's bounds.

    Attributes:
        task: Classification, regression or forecast.
        class_slots: Classes a classifier serves natively, without `@K`.
        context_rows: Most context rows it was trained on, if known.
        context_features: Most features it was trained on, if known.
        context_classes: Most classes it was trained on, if known.
        quantiles: A forecaster's fixed output levels, if known without weights.
        quantile_step: The spacing of a regressor's output quantile grid.
        past_inputs: Whether a forecaster reads past-only covariates.
        future_inputs: Whether a forecaster reads known-future covariates.
    """

    task: Task
    class_slots: int | None = None
    context_rows: int | None = None
    context_features: int | None = None
    context_classes: int | None = None
    quantiles: tuple[float, ...] = ()
    quantile_step: float | None = None
    past_inputs: bool = False
    future_inputs: bool = False

    @property
    def tabular(self) -> bool:
        """Whether it is a classifier or regressor rather than a forecaster."""
        return self.task != "forecast"


@dataclass(frozen=True)
class Gates:
    """Which surfaces the recorded evidence qualifies a model for.

    Attributes:
        package: Whether packages may lock it.
        remote: Whether remote serving may run it.
        variants: Surfaces beyond local that run its `@K` and `*N` variants,
            qualified with real weights.
        evidence: What the gates rest on, or what is missing.
    """

    package: bool = False
    remote: bool = False
    variants: frozenset[Surface] = frozenset()
    evidence: str = ""

    def allows(self, surface: Surface, *, variant: bool = False) -> bool:
        """Whether a surface may run the model, or one of its variants."""
        if surface == "local":
            return True
        base = self.package if surface == "package" else self.remote
        return base and (not variant or surface in self.variants)


@dataclass(frozen=True)
class ServingPolicy:
    """What a service allows, kept apart from what models can do.

    Attributes:
        context_rows: Most context rows a request or package binds.
        query_rows: Most query rows one Models API request scores.
        features: Most tabular features.
        horizon: Most forecast steps.
        group_series: Most related series forecast together.
        covariates: Most forecast covariates.
        cpu: CPUs a package revision requests.
        memory: Memory a package revision requests.
        max_codes: Most `@K` error-correcting codes.
        max_estimators: Most `*N` ensemble estimators.
        max_members: Most native estimator runs, codes times estimators.
    """

    context_rows: int
    query_rows: int
    features: int
    horizon: int
    group_series: int
    covariates: int
    cpu: int
    memory: str
    max_codes: int
    max_estimators: int
    max_members: int

    def admits(self, ref: "ModelRef") -> str | None:
        """Why the policy refuses a variant's cost, or None if it runs."""
        codes, estimators = ref.codes or 1, ref.estimators or 1
        if codes > self.max_codes:
            return f"at most @{self.max_codes} error-correcting codes"
        if estimators > self.max_estimators:
            return f"at most *{self.max_estimators} estimators"
        if codes * estimators > self.max_members:
            return f"at most {self.max_members} codes times estimators"
        return None


# Bounds v1 pins and locks recorded, frozen so their identities never move.
V1_POLICY = ServingPolicy(
    context_rows=1024,
    query_rows=100,
    features=32,
    horizon=32,
    group_series=8,
    covariates=8,
    cpu=2,
    memory="8Gi",
    max_codes=8,
    max_estimators=8,
    max_members=16,
)


@dataclass(frozen=True)
class ModelEntry:
    """One built-in model: its identity, capability and qualification.

    Attributes:
        name: The canonical name, such as `tabicl-v2-classifier`.
        family: The loader family, such as `tabicl`.
        artifact: The content it loads.
        capability: What its weights can do.
        gates: Where it is qualified beyond trusted local use.
        checkpoint: The directory name under `GF_CHECKPOINTS` and the cache,
            shared by entries that load the same files differently.
        aliases: Other names that resolve to it.
        options: Loader options, such as Timestral's symmetric averaging.
    """

    name: str
    family: str
    artifact: Artifact
    capability: Capability
    gates: Gates = Gates()
    checkpoint: str = ""
    aliases: tuple[str, ...] = ()
    options: tuple[tuple[str, object], ...] = ()

    @property
    def hostable(self) -> bool:
        """Whether its license permits serving it on General Forecast infrastructure."""
        return self.artifact.license in HOSTED_LICENSES

    def allows(self, surface: Surface, *, variant: bool = False) -> bool:
        """Whether a surface may run the model, or one of its variants.

        A self-hosted server runs on its operator's infrastructure, under the
        operator's own license: it takes the remote gates for hostable models,
        and any pinned base model whose license forbids hosting.
        """
        if surface != "self-hosted":
            return self.gates.allows(surface, variant=variant)
        if self.hostable:
            return self.gates.allows("remote", variant=variant)
        return not variant and bool(self.artifact.files)

    @property
    def repository_id(self) -> str:
        """The id the original registry named it by."""
        return f"{ORGANIZATION}/{self.name}"

    @property
    def directory(self) -> str:
        """Its checkpoint directory's name."""
        return self.checkpoint or self.name

    def capacity(self, ref: "ModelRef") -> int | None:
        """Most classes a variant of this classifier serves, or None for a regressor."""
        slots = self.capability.class_slots
        if slots is None:
            return None
        return MAX_CLASSES if ref.codes else slots


# SHA-256 of each pinned file, read from Hugging Face at each pinned revision.
_FILES: dict[str, dict[str, str]] = {
    "tabicl-v2-classifier": {
        "LICENSE": "4f90c5fa25a2b7883d6235c7f0b11f942a37859b9f89ac36157cc485329913b4",
        "config.json": "7ec2941cc537535e1bd8f37d33fc97cea97953343d69d765a261ee6a1605d937",
        "model.safetensors": "58f5fc1f276744f721974079c19d2f67efacc39083c54483fe3feba52297d253",
    },
    "tabicl-v2-regressor": {
        "LICENSE": "4f90c5fa25a2b7883d6235c7f0b11f942a37859b9f89ac36157cc485329913b4",
        "config.json": "eacc28b62075709af08bc5e7678720e160cbc27881eeeda8b7549245d3695b79",
        "model.safetensors": "b2d384f7ee3e79f971224dddb0fefbe15ba05c49ecae5030028d50a184b9b518",
    },
    "kumo-tabular-small-classifier": {
        "LICENSE": "44b43ed7b8e0f987dc5e90fb6cb4b901eccf0b14dc9f17a9033951603851d37d",
        "config.json": "fc08e3f796698d0c2b6854a92972e706521c7dd2f645a70e8b76428e943bb4a0",
        "model.safetensors": "fe55113ae955f991942fe92a595d1bea3dc58176081f8c2bcdd981733c417990",
    },
    "kumo-tabular-small-regressor": {
        "LICENSE": "44b43ed7b8e0f987dc5e90fb6cb4b901eccf0b14dc9f17a9033951603851d37d",
        "config.json": "fc08e3f796698d0c2b6854a92972e706521c7dd2f645a70e8b76428e943bb4a0",
        "model.safetensors": "fd77cabc96866b6708231fa53e31da79a46a8902db219078fed2fcc75d5493aa",
    },
    "kumo-tabular-medium-classifier": {
        "LICENSE": "44b43ed7b8e0f987dc5e90fb6cb4b901eccf0b14dc9f17a9033951603851d37d",
        "config.json": "c782566b989023b3a9c24a7ebb88d911f8e2eb7cee3b268e26fb99fe715b8323",
        "model.safetensors": "77359664b29c4f55876f3ecd73a7bc55a4d32a4fe2b2d4f61b873f81dd65301b",
    },
    "kumo-tabular-medium-regressor": {
        "LICENSE": "44b43ed7b8e0f987dc5e90fb6cb4b901eccf0b14dc9f17a9033951603851d37d",
        "config.json": "c782566b989023b3a9c24a7ebb88d911f8e2eb7cee3b268e26fb99fe715b8323",
        "model.safetensors": "ace4bdfa861bb308f708050d3017806d4f8ce2b900012bd37776253239ad6b09",
    },
    "kumo-tabular-large-classifier": {
        "LICENSE": "44b43ed7b8e0f987dc5e90fb6cb4b901eccf0b14dc9f17a9033951603851d37d",
        "config.json": "3e38b76af1c915e226acaf51f29511f2eb89f600ac5fe95b673ee1228b5a8466",
        "model.safetensors": "2e398863274d3b8088af5dd6dd5464dfc91a459c515f02872abf03709050386a",
    },
    "kumo-tabular-large-regressor": {
        "LICENSE": "44b43ed7b8e0f987dc5e90fb6cb4b901eccf0b14dc9f17a9033951603851d37d",
        "config.json": "3e38b76af1c915e226acaf51f29511f2eb89f600ac5fe95b673ee1228b5a8466",
        "model.safetensors": "ed84296ce00d436847baff6411835a436b61d0490124cf22c9b0943c86b7085e",
    },
    # Upstream files; their licenses forbid hosting a converted copy.
    "tabpfn-3-classifier": {
        "LICENSE": "dca491280b68f471312a15d54add7b8e724adf19fb0e113544b1ef91e060f5d5",
        "tabpfn-v3-classifier-v3_default.ckpt": "d0d865d54dfbc524f5703104be90620182dca7e5fb2c16de72e9959ea18f3988",
    },
    "tabpfn-3-regressor": {
        "LICENSE": "dca491280b68f471312a15d54add7b8e724adf19fb0e113544b1ef91e060f5d5",
        "tabpfn-v3-regressor-v3_default.ckpt": "311ce18d97e9533d8585eaadafe040fbdd8070533209ed8696641dadc97a7301",
    },
    "tabpfn-3.5-classifier": {
        "LICENSE": "9570fdd687bb76ed45499412ad57d6ba949848e6d731f1242572bb57f2f73ae1",
        "tabpfn-v3.5-20260909.safetensors": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
    },
    "tabpfn-3.5-regressor": {
        "LICENSE": "9570fdd687bb76ed45499412ad57d6ba949848e6d731f1242572bb57f2f73ae1",
        "tabpfn-v3.5-20260909.safetensors": "ece4d67eadfea42eb0e610df5189bea60cb7f31073d81e9c7a019b76eacf0be3",
    },
    "mitra-2-classifier": {
        "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        "NOTICE": "e10e58970f1c1e7ef200d45f274fb5330533a8d2553dda5a6e2d88c0fdf00fab",
        "config.json": "6fae96bf13a03210f575d519a763dfa3c902e05b391bd7ddabd67002e8b370d2",
        "model.safetensors": "5ffab0e2cf52f61c5b7c7eb1e8542996736d1023a0212190cc09abc2119a1e09",
    },
    "mitra-2-regressor": {
        "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        "NOTICE": "7d51fc8c96c5716e2295ae72e540171f0999001c6b0167aa9684630ff45ba79a",
        "config.json": "a4ceb3618fcde95e1b5cc8800fb090d8f8c8549db64e04c72e94802017b2691b",
        "model.safetensors": "8c90b8f281a793f9feb19f8fc3c5529c58ac844dd7ffc690a0fdbb96fae0967c",
    },
    "chronos-2": {
        "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
        "NOTICE": "f244df7a5307a500b53a6675eb4058359509fbf332f37aee9bb490f9829cf3f5",
        "config.json": "4c685ee692bbe94cdf775b8f6d9fa46498078ea1303bec8c2689f39436e60986",
        "model.safetensors": "6d353b01a3a939d324ea3abfb8f11cfba6e817b57db9b30a23b0def70528a087",
    },
    "timestral-209m": {
        "LICENSE": "648194727930df730890e9f9affdbcc5c930d406d6829559b25ca3d5a973fc8c",
        "NOTICE": "78d9a52a5bfea60302f46146870690591c41926f82804147148a39cdbbdec150",
        "config.json": "d6f74c7e003b9f857b13a752e122797a8b96e19f80ed32d19996196de4d2aa96",
        "model.safetensors": "a3a18ea4f2ea68791f0865542ea8fe4ce2ca5fbdf1cc03ac7fd11dc64d647983",
    },
    "timestral-28m": {
        "LICENSE": "648194727930df730890e9f9affdbcc5c930d406d6829559b25ca3d5a973fc8c",
        "NOTICE": "ebd4a1dd9913b158c5f6f0187d3c128053b7c2fcdeb13c83921542b76093cf03",
        "config.json": "79ca626795b5093dc1b05b579eb15a422e3ef8229bf5986b7b5481ce6361ba24",
        "model.safetensors": "b82d17044388365e5f47951431501598baa0904fcb0c6e07602ea08ef42ee986",
    },
    "timesfm-3": {
        "LICENSE": "3e36db7240d23adb6ac6d7d931892dcc5706a14a7d31581f08c35d2f25736dfd",
        "config.json": "ff17bbc07b792c5a904cca265b8468579d736a4fe84981da25eb871b0a125bc6",
        "model.safetensors": "a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da",
    },
}


class ModelRefError(ValueError):
    """A model reference that names no runnable model, with a stable code."""

    def __init__(self, code: str, message: str):
        """Keeps a machine-readable code apart from the message."""
        self.code = code
        super().__init__(message)


# An optional `catalog/`, a name, then optionally an `@sha256:` content pin,
# `@K` error-correcting codes and `*N` estimators, each once, in that order.
_REF = re.compile(
    r"(?:catalog/)?(?P<name>[^@*]+)"
    r"(?:@(?P<digest>sha256:[0-9a-f]{64}))?"
    r"(?:@(?P<codes>\d+))?"
    r"(?:\*(?P<estimators>\d+))?"
)
_TRUSTED = re.compile(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*")
_BASE = re.compile(r"(?:catalog/)?(?P<name>[^@*]+)(?:[@*].*)?", re.DOTALL)


@dataclass(frozen=True)
class ModelRef:
    """A parsed model reference: a model, optionally pinned, with its variant.

    Attributes:
        name: A built-in model's canonical name, or `module:function` for a
            trusted in-process model.
        codes: `@K` error-correcting codes, if any.
        estimators: `*N` ensemble estimators, if any.
        digest: The `@sha256:` artifact digest it is pinned to, if any.
    """

    name: str
    codes: int | None = None
    estimators: int | None = None
    digest: str | None = None

    @property
    def trusted(self) -> bool:
        """Whether it names code to import, which only trusted processes run."""
        return ":" in self.name

    @property
    def variant(self) -> bool:
        """Whether it wraps the model in codes or an ensemble."""
        return self.codes is not None or self.estimators is not None

    @property
    def entry(self) -> ModelEntry:
        """The built-in model it names."""
        if self.trusted:
            raise ModelRefError("UntrustedModel", f"{self.name!r} is not built in")
        return ENTRIES[self.name]

    @property
    def base(self) -> "ModelRef":
        """The same model without its variant."""
        return ModelRef(self.name, digest=self.digest)

    def __str__(self) -> str:
        """The canonical spelling, which parses back to the same reference."""
        return (
            self.name
            + (f"@{self.digest}" if self.digest else "")
            + (f"@{self.codes}" if self.codes is not None else "")
            + (f"*{self.estimators}" if self.estimators is not None else "")
        )


def admit(ref: ModelRef, surface: Surface) -> None:
    """Refuses a reference `surface` may not run, before any of its code loads.

    Raises:
        ModelRefError: If it names code to import beyond local use, a model
            not qualified for the surface, or a variant past the serving
            policy's bounds.
    """
    if ref.trusted:
        if surface != "local":
            raise ModelRefError(
                "UntrustedModel", "module:function models run only locally"
            )
        return
    entry = ref.entry
    if not entry.allows(surface, variant=ref.variant):
        raise ModelRefError(
            "UnsupportedModel",
            f"{ref} is not qualified for {surface} use: {entry.gates.evidence}",
        )
    if surface != "local" and (refused := V1_POLICY.admits(ref)):
        raise ModelRefError("CapacityExceeded", f"serving policy allows {refused}")


def parse(spec: str, *, trusted: bool = False) -> ModelRef:
    """Parses a model reference, resolving aliases and repository ids.

    Args:
        spec: A name, alias or repository id, such as `tabicl-classifier`,
            `general-forecast/tabicl-v2-classifier` or `catalog/<name>@sha256:...`,
            or with `trusted`, `module:function`; then optionally `@K` codes
            and `*N` estimators, each at most once, in that order.
        trusted: Whether the caller runs `module:function` code in its own
            process. Packages, locks and servers never pass it, so uploaded
            text can never name code for them to import.

    Returns:
        The reference, with the model's canonical name.

    Raises:
        ModelRefError: If the spec is malformed, names no built-in model,
            names code without `trusted`, pins other content than the catalog
            holds, or asks a variant the model cannot take.
    """
    match = _REF.fullmatch(spec) if type(spec) is str else None
    if match is None:
        raise ModelRefError(
            "InvalidModel",
            f"a model spec is a name, then optionally @K, then optionally *N: {spec!r}",
        )
    name, digest, codes, estimators = match.group(
        "name", "digest", "codes", "estimators"
    )
    counts = [int(count) for count in (codes, estimators) if count is not None]
    if 0 in counts:
        raise ModelRefError(
            "InvalidModel", f"counts in a model spec are positive: {spec!r}"
        )
    ref = ModelRef(
        name,
        int(codes) if codes is not None else None,
        int(estimators) if estimators is not None else None,
        digest,
    )
    if ":" in name:
        if not trusted:
            raise ModelRefError(
                "UntrustedModel",
                "module:function models run only in a trusted local process",
            )
        if _TRUSTED.fullmatch(name) is None or digest or spec.startswith("catalog/"):
            raise ModelRefError(
                "InvalidModel", f"expected module:function, got {name!r}"
            )
        return ref
    canonical = _NAMES.get(name)
    if canonical is None:
        raise ModelRefError(
            "UnknownModel",
            f"unknown model {name!r}; built-in models are {sorted(ENTRIES)}",
        )
    entry = ENTRIES[canonical]
    if digest is not None and digest != entry.artifact.digest:
        raise ModelRefError("ModelMismatch", "requested model pin differs from catalog")
    if ref.variant and not entry.capability.tabular:
        raise ModelRefError(
            "UnsupportedCapability",
            f"a forecaster takes no codes or estimators: {spec!r}",
        )
    if codes is not None and entry.capability.task != "classification":
        raise ModelRefError(
            "UnsupportedCapability",
            f"error-correcting codes need a classifier: {spec!r}",
        )
    return ModelRef(canonical, ref.codes, ref.estimators, digest)


@overload
def resource_name(reference: str) -> str: ...
@overload
def resource_name(reference: object) -> object: ...
def resource_name(reference: object) -> object:
    """The model resource a reference names: its canonical name, without pin or variant.

    Routes and request envelopes name this resource while the request body
    keeps the full reference, so whatever runs it still admits the pin and
    variant; an invalid one is never dropped. Unknown names are returned as
    given, and non-references unchanged, for the caller's own checks to reject.
    """
    if type(reference) is not str or (match := _BASE.fullmatch(reference)) is None:
        return reference
    name = match.group("name")
    return _NAMES.get(name, name)


def entry(spec: str) -> ModelEntry:
    """The built-in model a reference names, ignoring its variant."""
    return parse(spec).entry


def entries() -> tuple[ModelEntry, ...]:
    """Every built-in model, in catalog order."""
    return tuple(ENTRIES.values())


def manifest() -> dict:
    """The catalog as the native request validator reads it, as portable JSON.

    `gf_core::catalog` compiles in the copy checked in at
    `crates/gf-core/catalog.json`, which `python -m generalforecast.models.catalog`
    prints and a test keeps equal to this. It carries what a request is
    checked against before any weights load: every spelling that resolves to
    a model, each model's task, content digest, license and pin state, the
    classes and quantiles it serves, its gates, and the v1 serving policy.
    """
    return {
        "max_classes": MAX_CLASSES,
        "policy": {
            "context_rows": V1_POLICY.context_rows,
            "query_rows": V1_POLICY.query_rows,
            "features": V1_POLICY.features,
            "horizon": V1_POLICY.horizon,
            "group_series": V1_POLICY.group_series,
            "covariates": V1_POLICY.covariates,
            "max_codes": V1_POLICY.max_codes,
            "max_estimators": V1_POLICY.max_estimators,
            "max_members": V1_POLICY.max_members,
        },
        "names": dict(sorted(_NAMES.items())),
        "entries": [
            {
                "name": item.name,
                "task": item.capability.task,
                "digest": item.artifact.digest,
                "pinned": bool(item.artifact.files),
                "hostable": item.hostable,
                "bounds": {
                    "classes": item.capability.class_slots,
                    "quantiles": list(item.capability.quantiles),
                    "quantile_step": item.capability.quantile_step,
                },
                "gates": {
                    "package": item.gates.package,
                    "remote": item.gates.remote,
                    "variants": sorted(item.gates.variants),
                },
            }
            for item in _ALL
        ],
    }


def _files(name: str) -> MappingProxyType:
    return MappingProxyType(dict(_FILES.get(name, {})))


def _pinned(
    name: str, revision: str, license: str | None, source: str | None
) -> Artifact:
    return Artifact(f"{ORGANIZATION}/{name}", revision, _files(name), license, source)


_TABICL = "pinned config, weights and LICENSE; real-weight package, Models API and @K/*N parity tests"
_LOCAL = (
    "pinned config, weights and license files are hashed; package and remote "
    "serving await runtime qualification with real weights and license review"
)
_LOCAL_GATES = Gates(evidence=_LOCAL)
_UNHASHED = (
    "pinned revision only: the repository is not publicly readable, so no file "
    "hashes or license evidence are recorded; trusted local use only"
)


def _tabular(
    name: str,
    family: str,
    revision: str,
    *,
    trained: tuple[int | None, int | None, int | None] = (None, None, None),
    class_slots: int = 10,
    license: str | None = None,
    source: str | None = "LICENSE",
    gates: Gates = _LOCAL_GATES,
    aliases: tuple[str, ...] = (),
    artifact: Artifact | None = None,
) -> ModelEntry:
    task: Task = "classification" if name.endswith("classifier") else "regression"
    rows, features, classes = trained
    return ModelEntry(
        name,
        family,
        artifact or _pinned(name, revision, license, source if license else None),
        Capability(
            task,
            class_slots=class_slots if task == "classification" else None,
            context_rows=rows,
            context_features=features,
            context_classes=classes if task == "classification" else None,
            quantile_step=0.001 if task == "regression" else None,
        ),
        gates if gates.evidence else Gates(evidence=_UNHASHED),
        aliases=aliases,
    )


def _upstream(
    name: str,
    repository: str,
    revision: str,
    convert: str,
    license: str | None,
    source: str | None,
) -> Artifact:
    return Artifact(repository, revision, _files(name), license, source, convert)


_HOSTED = (
    "pinned config, weights and license files; real-weight Models API "
    "classification/regression and @K/*N tests"
)
_HOSTED_GATES = Gates(remote=True, variants=frozenset({"remote"}), evidence=_HOSTED)
_TABICL_GATES = Gates(
    package=True,
    remote=True,
    variants=frozenset({"package", "remote"}),
    evidence=_TABICL,
)
_KUMO = {
    "small-classifier": "7d4262453c0c0dea486c890991a418535c4d8f4b",
    "small-regressor": "44aad6a62bcffee36f96f08af64007b057f955c1",
    "medium-classifier": "f7c3ce9bb9a2912e8442f68960a959f068e9287a",
    "medium-regressor": "39442da5a79a028399bca5e6166abea96adcf4c5",
    "large-classifier": "6839febcef6c85c0fda58ec38a0d3cdf5d9886af",
    "large-regressor": "d0a0edc4a135f788a5ba8338bc06c6b882651716",
}
_TABPFN_3 = ("Prior-Labs/tabpfn_3", "24a16a89d245878b846555110985634aa2e656d7")
_TABPFN_3_5 = ("Prior-Labs/tabpfn_3_5", "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e")
# Mirrors converted from the upstream repositories, which stay aliases. Each
# mirror's NOTICE records its upstream revision.
_TIMESTRAL = {
    "timestral-209m": (
        "mistralai/Timestral-209M",
        "37676b196d5e39d76ba7495f5393247e23aa5a4b",
    ),
    "timestral-28m": (
        "mistralai/Timestral-28M",
        "645062a557cc9c032883031788991f3582c74d5a",
    ),
}
_MITRA_2 = {
    "mitra-2-classifier": (
        "autogluon/mitra-classifier-2",
        "bc87248a2ba873bf623705beecd8c7c9cd9bfa18",
    ),
    "mitra-2-regressor": (
        "autogluon/mitra-regressor-2",
        "124a99e9a31879ce4f1ef2b7f333b02f716a67be",
    ),
}


def _forecaster(
    name: str,
    family: str,
    artifact: Artifact,
    capability: Capability,
    gates: Gates = _LOCAL_GATES,
    checkpoint: str = "",
    aliases: tuple[str, ...] = (),
    options: tuple[tuple[str, object], ...] = (),
) -> ModelEntry:
    return ModelEntry(
        name, family, artifact, capability, gates, checkpoint, aliases, options
    )


_ALL: tuple[ModelEntry, ...] = (
    _tabular(
        "tabicl-v2-classifier",
        "tabicl",
        "5fb6a4c6f24bd5fb3623f97e3c62186e12e28e5b",
        trained=(60_000, 100, 10),
        license="BSD-3-Clause",
        gates=_TABICL_GATES,
        aliases=("tabicl-classifier",),
    ),
    _tabular(
        "tabicl-v2-regressor",
        "tabicl",
        "1466a562be5d04a25ab612940d80c2c2f901c797",
        trained=(60_000, 100, 10),
        license="BSD-3-Clause",
        gates=_TABICL_GATES,
        aliases=("tabicl-regressor",),
    ),
    *(
        _tabular(
            f"kumo-tabular-{size}",
            "kumo",
            revision,
            trained=(60_000, 100, 10),
            license="OpenMDW-1.1",
            gates=_HOSTED_GATES,
        )
        for size, revision in _KUMO.items()
    ),
    *(
        _tabular(
            name,
            "tabpfn",
            _TABPFN_3[1],
            class_slots=160,
            trained=(1_000_000, 2_000, None),
            artifact=_upstream(
                name,
                *_TABPFN_3,
                "tabpfn_v3",
                "LicenseRef-tabpfn-3-license-v1.0",
                "LICENSE",
            ),
        )
        for name in ("tabpfn-3-classifier", "tabpfn-3-regressor")
    ),
    *(
        _tabular(
            name,
            "tabpfn",
            _TABPFN_3_5[1],
            class_slots=160,
            trained=(1_000_000, 6_000, 160),
            artifact=_upstream(
                name,
                *_TABPFN_3_5,
                "tabpfn_v3_5",
                "LicenseRef-tabpfn-3-5-license-v1.0",
                "LICENSE",
            ),
        )
        for name in ("tabpfn-3.5-classifier", "tabpfn-3.5-regressor")
    ),
    _tabular(
        "limix-2-classifier",
        "limix",
        "e3ba7be7c8d6f265844b1c1aabc4bd9e365adc6a",
        gates=Gates(),
    ),
    _tabular(
        "limix-2-regressor",
        "limix",
        "ad1d2711a9807bdaf3c315960cb0ae4b1f6514a7",
        gates=Gates(),
    ),
    _tabular(
        "mitra-classifier",
        "mitra",
        "64bc87614af639ccea79dd71a96f3bbf357fee86",
        trained=(512, 16, None),
        gates=Gates(),
    ),
    *(
        _tabular(
            name,
            "mitra",
            revision,
            trained=(5_120, 50, None),
            license="Apache-2.0",
            gates=_HOSTED_GATES,
            aliases=(upstream,),
        )
        for name, (upstream, revision) in _MITRA_2.items()
    ),
    _forecaster(
        "chronos-2",
        "chronos2",
        _pinned(
            "chronos-2",
            "a287b88beaeef49cf436e6735d9f61e13dfd50f8",
            "Apache-2.0",
            "LICENSE",
        ),
        Capability(
            "forecast",
            quantiles=CHRONOS2_QUANTILES,
            past_inputs=True,
            future_inputs=True,
        ),
        Gates(
            package=True,
            remote=True,
            evidence="pinned mirror config, weights, LICENSE and NOTICE, "
            "byte-identical to the verified conversion; real-weight package "
            "and Models API grouped forecast tests",
        ),
        aliases=("amazon/chronos-2",),
    ),
    *(
        _forecaster(
            name + suffix,
            "timestral",
            # The LICENSE files are pinned; the upstream cards say only "other".
            _pinned(name, revision, "LicenseRef-timestral-license", "LICENSE"),
            Capability("forecast"),
            gates=_HOSTED_GATES,
            checkpoint=name,
            options=(("symmetric", not suffix),),
            aliases=() if suffix else (upstream,),
        )
        for name, (upstream, revision) in _TIMESTRAL.items()
        # Without its averaging over the group and its negation, which halves
        # the work and the cache at some accuracy.
        for suffix in ("", "-unmirrored")
    ),
    _forecaster(
        "timesfm-3",
        "timesfm3",
        _upstream(
            "timesfm-3",
            "google/timesfm-3.0-pytorch",
            "43046b85ec22d584a13f8098c2ed39c889e129c2",
            "timesfm3",
            "LicenseRef-timesfm-non-commercial-license-v1.0",
            "LICENSE",
        ),
        Capability("forecast"),
    ),
)
if unlicensed := [i.name for i in _ALL if i.gates.remote and not i.hostable]:
    raise AssertionError(f"remote gates need a hosted license: {unlicensed}")
ENTRIES: MappingProxyType = MappingProxyType({item.name: item for item in _ALL})
_NAMES: dict[str, str] = {
    alias: item.name
    for item in _ALL
    for alias in (item.name, item.repository_id, *item.aliases)
}


if __name__ == "__main__":
    import json

    print(json.dumps(manifest(), indent=2))
