"""Shared benchmark configuration and paths anchored to the repository."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = Path(__file__).resolve().parent / "gen_specs"
YAML_DIR = PROJECT_ROOT / "eval_engines/ngspice/ngspice_inputs/yaml_files"

BENCHMARKS = {
    "TSA": ("two_stage_opamp_gf180.yaml", "ngspice_specs_gen_tsa_gf180_tran_test_200"),
    "CMA": (
        "cascode_miller_opamp_gf180.yaml",
        "ngspice_specs_gen_cma_gf180_tran_test_200",
    ),
    "LDO": ("ldo_sky130.yaml", "ngspice_specs_gen_ldo_sky130_test_100"),
    "COMP": ("comparator_gf180.yaml", "ngspice_specs_gen_comp_gf180_test_100"),
}


def resolve_path(value, base=PROJECT_ROOT):
    """Resolve a repository-relative path, preserving absolute paths."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def resolve_spec_path(value):
    """Bare spec filenames refer to the package's bundled evaluation specs."""
    path = Path(value).expanduser()
    return resolve_path(path, SPEC_DIR if path.parent == Path(".") else PROJECT_ROOT)


def load_config(path):
    # FullLoader supports the existing !!python/tuple benchmark data.
    with resolve_path(path).open() as stream:
        return yaml.full_load(stream)


def benchmark_config(name):
    yaml_name, spec_name = BENCHMARKS[name.upper()]
    return {"CIR_YAML": str(YAML_DIR / yaml_name), "spec_path": spec_name}
