"""
Snapshot tests that pin the exact TOML written to the definitions/ directory.
Reading these parametrize arguments tells you what a materialized plan looks like.
"""

from __future__ import annotations

import pytest

from tidyrun import DAG, ParametrizedJob
from tidyrun.job import Job


def _join(left: str, right: str) -> str:
    return f"{left}/{right}"


# ---------------------------------------------------------------------------
# Build the three cases at import time so they can be used in @parametrize.
# ---------------------------------------------------------------------------

_dag_one_job = DAG()
_dag_one_job["result"] = Job(func=_join, kwargs={"left": "a", "right": "x"})

_pjob = ParametrizedJob(
    func=_join,
    parameter_names=["left"],
    parameter_values=[("a",), ("b",)],
    kwargs={"right": "x"},
)

_dag_two_pjobs = DAG()
_dag_two_pjobs["foo"] = ParametrizedJob(
    func=_join,
    parameter_names=["left"],
    parameter_values=[("a",), ("b",)],
    kwargs={"right": "x"},
)
_dag_two_pjobs["bar"] = ParametrizedJob(
    func=_join,
    parameter_names=["left"],
    parameter_values=[("p",), ("q",)],
    kwargs={"right": "y"},
)


@pytest.mark.parametrize(
    "node, expected_definitions",
    [
        # ------------------------------------------------------------------
        # a) DAG with one Job — one definition file, all args are literals.
        # ------------------------------------------------------------------
        pytest.param(
            _dag_one_job,
            {
                "result.tidyrun": """\
kind = "job_definition"
schema_version = 1
dependencies = []
callable_module = "test_plan_as_text"
callable_qualname = "_join"

[args.left]
kind = "literal"
path = "inputs/result/left"

[args.right]
kind = "literal"
path = "inputs/result/right"
""",
            },
            id="dag_one_job",
        ),
        # ------------------------------------------------------------------
        # b) Standalone ParametrizedJob — one file per parameter value;
        #    the bound parameter lands as a literal (no parameter_names).
        # ------------------------------------------------------------------
        pytest.param(
            _pjob,
            {
                "a.tidyrun": """\
kind = "job_definition"
schema_version = 1
dependencies = []
callable_module = "test_plan_as_text"
callable_qualname = "_join"

[args.right]
kind = "literal"
path = "inputs/a/right"

[args.left]
kind = "literal"
path = "inputs/a/left"
""",
                "b.tidyrun": """\
kind = "job_definition"
schema_version = 1
dependencies = []
callable_module = "test_plan_as_text"
callable_qualname = "_join"

[args.right]
kind = "literal"
path = "inputs/b/right"

[args.left]
kind = "literal"
path = "inputs/b/left"
""",
            },
            id="parametrized_job",
        ),
        # ------------------------------------------------------------------
        # c) DAG with two ParametrizedJobs — one array-group file per job;
        #    parameter_names is set and the parameter arg is kind="parameter".
        # ------------------------------------------------------------------
        pytest.param(
            _dag_two_pjobs,
            {
                "foo.tidyrun": """\
kind = "job_definition"
schema_version = 1
dependencies = []
parameter_names = [ "left",]
callable_module = "test_plan_as_text"
callable_qualname = "_join"

[args.right]
kind = "literal"
path = "inputs/foo/right"

[args.left]
kind = "parameter"
values = [ "a", "b",]
""",
                "bar.tidyrun": """\
kind = "job_definition"
schema_version = 1
dependencies = []
parameter_names = [ "left",]
callable_module = "test_plan_as_text"
callable_qualname = "_join"

[args.right]
kind = "literal"
path = "inputs/bar/right"

[args.left]
kind = "parameter"
values = [ "p", "q",]
""",
            },
            id="dag_two_parametrized_jobs",
        ),
    ],
)
def test_plan_as_text(node, expected_definitions, tmp_path):
    plan_dir = node.materialize(tmp_path)
    actual = {
        p.name: p.read_text()
        for p in sorted((plan_dir / "definitions").glob("*.tidyrun"))
    }
    assert actual == expected_definitions
