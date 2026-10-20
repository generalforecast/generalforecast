import ast
import json
from pathlib import Path

import pytest

import generalforecast
from generalforecast.deployments.contracts import (
    PUBLIC_ERROR_CODES,
    ServiceError,
    public_code,
)
from generalforecast.transport import SERVICE_ERROR_CODES, response_value

# Raised only by local clients/CLI; never sent by a service.
LOCAL_ERROR_CODES = {
    "CleanupFailed",
    "ClientClosed",
    "ClientError",
    "DeploymentFailed",
    "InvalidResult",
    "JobFailed",
    "ReadinessFailed",
    "Timeout",
    "TransportError",
    "UploadFailed",
}
LOCAL_MODULES = {
    "generalforecast/client.py",
    "generalforecast/async_client.py",
    "generalforecast/transport.py",
    "generalforecast/commands.py",
    "generalforecast/cli.py",
    "generalforecast/package_commands.py",
}
CONSTRUCTORS = {
    "ServiceError",
    "PackageError",
    "failure",
    "conflict",
    "source_error",
}


def _raised_codes():
    found = {}
    for package in (generalforecast, generalforecast):
        root = Path(package.__file__).parent
        for path in root.rglob("*.py"):
            relative = path.relative_to(root.parent).as_posix()
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                called = getattr(func, "id", getattr(func, "attr", ""))
                args = node.args
                if called in {"Capability", "ensure"} and len(args) >= 3:
                    args = args[2:]
                elif called not in CONSTRUCTORS:
                    continue
                if (
                    args
                    and isinstance(args[0], ast.Constant)
                    and type(args[0].value) is str
                    and args[0].value.isidentifier()
                ):
                    found.setdefault(args[0].value, set()).add(relative)
    return found


def test_every_service_raised_code_is_public_vocabulary():
    unregistered = {
        code: sorted(paths - LOCAL_MODULES)
        for code, paths in _raised_codes().items()
        if code not in PUBLIC_ERROR_CODES
        and not (
            code in LOCAL_ERROR_CODES
            and paths <= LOCAL_MODULES | {"generalforecast/contracts.py"}
        )
    }
    assert unregistered == {}
    assert SERVICE_ERROR_CODES is PUBLIC_ERROR_CODES
    assert not LOCAL_ERROR_CODES & PUBLIC_ERROR_CODES - {"Timeout"}


@pytest.mark.parametrize(
    "code,expected",
    [
        ("OperationSuperseded", "OperationSuperseded"),
        ("PredictorTimeout", "PredictorTimeout"),
        ("FutureCodeFromNewerServer", "RemoteError"),
        ("ghp_credentialLookingCode", "RemoteError"),
    ],
)
def test_remote_codes_are_classified_without_echoing_unknowns(code, expected):
    assert public_code(code) == expected
    raw = json.dumps({"error": {"code": code, "message": "secret detail"}}).encode()
    with pytest.raises(ServiceError) as error:
        response_value(409, raw)
    assert error.value.code == expected and error.value.status == 409
    assert "secret detail" not in str(error.value) and (
        expected != "RemoteError" or code not in str(error.value.code)
    )
