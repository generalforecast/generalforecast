# SPDX-FileCopyrightText: General Forecast, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The rendered suite matches the contract's geometry, string for string."""

import re

import pytest

from tests.product_cli.render_suite import CASES, render


@pytest.fixture(scope="module")
def suite(tmp_path_factory):
    return render(tmp_path_factory.mktemp("render"))


def lines(suite, name):
    return suite[name].split("\n")


def test_every_case_rendered(suite):
    assert set(suite) >= {case.name for case in CASES}


def test_status_lines_end_their_verb_in_column_twelve(suite):
    seen = 0
    for name, text in suite.items():
        for line in text.split("\n"):
            match = re.match(r"^( *)([A-Z][a-z]+) ", line)
            if match and not name.startswith("help") and len(match.group(0)) == 13:
                seen += 1
                assert len(match.group(1)) + len(match.group(2)) == 12, (name, line)
    assert seen > 20


def test_init_transcript(suite):
    assert lines(suite, "init-prompt") == [
        "       Found context.csv  2,052 rows, 5 columns",
        "",
        "Target",
        "  > churned        0/1",
        "    customer_id    int",
        "    spend          float",
        "    plan           category",
        "    signups        int",
    ]
    assert lines(suite, "init") == [
        "       Found context.csv  2,052 rows, 5 columns",
        "",
        "Target       churned",
        "Task         classification  churned has 2 distinct 0/1 values",
        "Model        general-forecast/kumo-tabular-medium-classifier",
        "Id           customer_id",
        "Inputs       spend, plan, signups",
        "Queries      212 rows with no target",
        "",
        " Initialized predictor churn in ./gf.yaml",
    ]
    assert lines(suite, "init-yes")[-1] == " Initialized predictor churn in ./gf.yaml"
    assert "Queries      a random 0.2 of the rows" in lines(suite, "init-flags")
    assert lines(suite, "init-exists") == [
        "error: gf.yaml already exists in ./churn",
        "  hint: pass --force to overwrite it, or run `gf check` to validate it",
    ]


def test_check(suite):
    assert lines(suite, "check") == [
        "    Resolved orders  classification, 3 inputs, 10,000 context rows, 0 query rows",
        "   Selection retrieved  core 1,500 stratified, 4,500 nearest per query",
        "  id       order_id                      from gf.yaml",
        "  inputs   amount, items, signup_source  every other column",
        "  target   returned                      from gf.yaml",
        "  context  10,000 rows                   context.csv",
        " + input      signup_source",
        " ~ selection  core 1,000 → 1,500, neighbors 3,000 → 4,500",
        "      Locked gf.lock in 0.21s",
    ]
    assert lines(suite, "check-whole")[0] == (
        "    Resolved churn  classification, 3 inputs, 1,840 context rows, 212 query rows"
    )
    assert lines(suite, "check-locked")[-3:] == [
        "error: gf.lock would change",
        "  because: 2 changes are listed above",
        "  hint: run `gf check` without --locked to write them",
    ]


def test_error_shapes(suite):
    assert lines(suite, "check-typo") == [
        "error: target column `churned` is not in the context",
        "  --> gf.yaml:3:9",
        "   |",
        " 3 | target: churned",
        "   |         ^^^^^^^ context.csv has customer_id, spend, plan, churn",
        "  hint: did you mean `churn`?",
    ]
    assert lines(suite, "check-cadence") == [
        "error: predictor sales has no cadence",
        "  because: general-forecast/chronos-2 forecasts series, and gf.yaml",
        "           declares no `time` column",
        "  hint: add `time: date` and `frequency: 1d`, or pick a tabular model",
        "        with `gf model list --task classification`",
    ]
    assert lines(suite, "predict-capacity") == [
        "error: queries exceed the limit for general-forecast/tabicl-v2-classifier",
        "  because: this model accepts 100 queries per call; the request sent 1,000",
        "  hint: split the queries, or push a predictor and call it in batches",
        "  request r-3abb2401f119428db148927c6a0d358c",
    ]
    assert (
        lines(suite, "check-bare")[0]
        == "error: `tabicl-classifier` is not a catalog model id; "
        "use general-forecast/tabicl-v2-classifier"
    )
    assert not any(line.startswith("  hint:") for line in lines(suite, "check-bare"))
    assert lines(suite, "model-bare") == [
        (
            "error: `tabicl-classifier` is not a catalog model id; "
            "use general-forecast/tabicl-v2-classifier"
        ),
    ]
    assert lines(suite, "error-unknown-command") == [
        "error: unknown command `chekc`",
        "  tip: a similar command exists: 'check'",
    ]
    assert lines(suite, "error-unknown-argument") == [
        "error: unexpected argument '--contxt'",
        "  tip: a similar argument exists: '--context'",
    ]
    assert lines(suite, "error-internal") == [
        "    Resolved churn  classification, 1,840 context rows, 212 query rows",
        "error: internal error",
        "  hint: run again with -v",
    ]
    assert "Traceback (most recent call last):" in suite["error-internal-verbose"]
    assert "Traceback" not in suite["error-internal"]
    assert suite["error-json"].startswith(
        '{"error": {"code": "ValidationError", "message": '
    )
    assert suite["predict-json"].count("\n") == 0


def test_eval_table(suite):
    assert lines(suite, "eval") == [
        "    Resolved churn  classification, 1,840 context rows, 212 query rows",
        "   Predicted 212 rows in 0.19s",
        "",
        "             accuracy   0.934",
        "             log loss   0.187",
        "             auroc      0.961",
    ]
    assert lines(suite, "eval-models")[3:] == [
        "                        tabicl-v2-classifier   mitra-2-classifier",
        "             accuracy                  0.934                0.921",
        "             log loss                  0.187                0.203",
        "             auroc                     0.961                0.948",
        "       Wrote scored.parquet",
    ]


def test_predict_and_hosted(suite):
    assert lines(suite, "predict") == [
        "   Predicted 1,000 rows in 0.21s",
        "       Wrote scored.parquet",
    ]
    push = lines(suite, "push")
    assert push[:4] == [
        "  name         churn",
        "  revision     8",
        "  environment  staging",
        "  model        general-forecast/tabicl-v2-classifier@7a01c2e",
    ]
    assert push[-2].startswith("   Uploading churn revision 8  snapshot, 1,840 rows, ")
    assert push[-1] == "    Deployed churn revision 8 to staging in 41s"
    assert "Push? [y/N] y" in lines(suite, "push-ask")
    assert lines(suite, "status") == [
        "environment  revision  status  runtime",
        "production   6         ready   default",
        "staging      7         ready   default",
    ]
    assert lines(suite, "promote") == [
        "    Promoted churn revision 7 to production in 3.2s"
    ]
    assert lines(suite, "rollback") == [
        "     Pointed production at churn revision 5 in 2.8s"
    ]
    assert lines(suite, "context-append") == ["    Appended 1,000 rows to orders-live"]
    assert lines(suite, "admin-diff") == [
        " ~ model      general-forecast/tabicl-v2-classifier 5fb6a4c → 7a01c2e",
        " + input      signups",
    ]


def test_help_pages_close_with_the_contract_line(suite):
    for name, text in suite.items():
        if name.startswith("help"):
            assert text.rstrip().endswith(
                "See 'gf help <command>' for more information"
            ), name


def test_a_prompt_is_ready_only_when_its_frame_is_whole():
    from generalforecast.term import FRAME_END
    from tests.product_cli.render_suite import COLUMNS, Screen, _prompt_ready

    def ready(output: bytes, label: str) -> bool:
        return _prompt_ready(output, Screen(COLUMNS).feed(output).text(), label)

    header = b"       Found context.csv\r\n\r\nTarget\r\n"
    rows = b"  > churned        0/1\r\n    customer_id    int\r\n"
    end = FRAME_END.encode()
    assert not ready(header, "Target")
    assert not ready(header + rows, "Target")
    assert ready(header + rows + end, "Target")
    assert not ready(header + rows + end, "Model")
    assert not ready(
        header + rows + end + b"\x1b[3A\r\x1b[JTarget       churned\r\nModel\r\n",
        "Model",
    )
    assert ready(b"Until        2025-01-28  3 rows\r\n" + end, "Until")
    assert not ready(b"Until        2025-01-28", "Until")
    assert ready(b"Push? [y/N] ", "Push?")
    assert not ready(b"Push? [y/", "Push?")
