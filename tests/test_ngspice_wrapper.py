"""Check path relocation and cleanup without requiring external PDK models."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from eval_engines.ngspice.ngspice_wrapper_parallel import NgSpiceWrapper


def test_model_paths_survive_netlist_relocation(tmp_path):
    model = tmp_path / "models with spaces" / "devices.lib"
    model.parent.mkdir()
    model.touch()
    source_dir = tmp_path / "netlists"
    source_dir.mkdir()
    include = '.include "../models with spaces/devices.lib"\n'
    library = ".LIB '../models with spaces/devices.lib' typical\n"
    resolve = NgSpiceWrapper.resolve_model_path
    assert resolve(include, source_dir) == f'.include "{model}"\n'
    assert resolve(library, source_dir) == f'.LIB "{model}" typical\n'
    assert resolve(".lib typical\n", source_dir) == ".lib typical\n"


def test_simulator_error_is_reported_and_temporary_design_is_removed(tmp_path):
    wrapper = NgSpiceWrapper.__new__(NgSpiceWrapper)
    failed = subprocess.CompletedProcess([], 1, stdout="", stderr="model not found")
    with patch("subprocess.run", return_value=failed) as run:
        with pytest.raises(RuntimeError, match="model not found"):
            wrapper.create_design_and_simulate({}, "test", str(tmp_path), [".end\n"])
        command = run.call_args.args[0]
        assert command[:2] == ["ngspice", "-b"]
        assert not Path(command[2]).exists()
    assert list(tmp_path.iterdir()) == []
