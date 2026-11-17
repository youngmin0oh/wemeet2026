"""Create temporary circuit designs and evaluate their PVT corners with ngspice."""

import os
import re
import shutil
import subprocess
import tempfile
from multiprocessing.dummy import Pool as ThreadPool
from pathlib import Path

import numpy as np
import yaml


class NgSpiceWrapper:
    BASE_TMP_DIR = str(Path(__file__).resolve().parents[2] / "tmp/ckt_da")

    def __init__(
        self, yaml_path, path, num_process=None, root_dir=None, design_netlists=None
    ):
        self.root_dir = os.path.abspath(root_dir or self.BASE_TMP_DIR)
        os.makedirs(self.root_dir, exist_ok=True)
        self.base_design_names = []
        self.gen_dirs = []
        self.tmp_lines = []
        if design_netlists is None:
            with open(yaml_path) as stream:
                design_netlists = yaml.full_load(stream)["dsn_netlist"]
        self.num_process = (
            num_process if num_process is not None else max(1, len(design_netlists))
        )

        for netlist in design_netlists:
            netlist_path = Path(path) / netlist
            name = netlist_path.stem
            self.base_design_names.append(name)
            gen_dir = os.path.join(self.root_dir, "designs_" + name)
            self.gen_dirs.append(gen_dir)
            os.makedirs(gen_dir, exist_ok=True)
            with netlist_path.open() as stream:
                self.tmp_lines.append(
                    [
                        self.resolve_model_path(line, netlist_path.parent)
                        for line in stream
                    ]
                )

    @staticmethod
    def resolve_model_path(line, source_dir):
        """Resolve model paths before copying a netlist into its simulation folder."""
        match = re.match(
            r"""^(\s*\.(?:include|lib)\s+)(?:"([^"]+)"|'([^']+)'|(\S+))(.*)$""",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            return line
        model_path = next(value for value in match.groups()[1:4] if value is not None)
        # A .lib section declaration contains a section name rather than a file.
        if match.group(1).strip().lower() == ".lib" and not (
            match.group(5).strip() or "/" in model_path or "." in model_path
        ):
            return line
        model = Path(model_path)
        if not model.is_absolute():
            model = (Path(source_dir) / model).resolve()
        return f'{match.group(1)}"{model}"{match.group(5)}\n'

    def get_design_name(self, state, base_design_name):
        values = (str(np.round(float(value), 9)) for value in state.values())
        return "_".join([base_design_name, *values])

    def create_design(self, state, new_fname, gen_dir, tmp_lines):
        design_folder = tempfile.mkdtemp(prefix=new_fname + "_", dir=gen_dir)
        fpath = os.path.join(design_folder, new_fname + ".cir")
        lines = []
        for line in tmp_lines:
            if re.match(r"\s*\.param\b", line, re.IGNORECASE):
                for key, value in state.items():
                    line = re.sub(
                        rf"\b{re.escape(key)}\s*=\s*\S+",
                        lambda match, key=key, value=value: f"{key}={value}",
                        line,
                        flags=re.IGNORECASE,
                    )
            match = re.search(r"\b(?:wrdata|write)\s+(\w+\.\w+)", line, re.IGNORECASE)
            if match:
                start, end = match.span(1)
                line = (
                    line[:start]
                    + os.path.join(design_folder, match.group(1))
                    + line[end:]
                )
            lines.append(line)
        try:
            with open(fpath, "w") as stream:
                stream.writelines(lines)
        except Exception:
            shutil.rmtree(design_folder)
            raise
        return design_folder, fpath

    def simulate(self, fpath):
        result = subprocess.run(
            ["ngspice", "-b", fpath], capture_output=True, text=True
        )
        if result.returncode:
            output = (result.stdout + result.stderr)[-4000:]
            raise RuntimeError(f"ngspice failed for {fpath}:\n{output}")
        return 0

    def create_design_and_simulate(
        self, state, base_design_name, gen_dir, tmp_lines, verbose=False
    ):
        name = self.get_design_name(state, base_design_name)
        if verbose:
            print(name)
        design_folder, fpath = self.create_design(state, name, gen_dir, tmp_lines)
        try:
            info = self.simulate(fpath)
            return state, self.translate_result(design_folder), info
        finally:
            shutil.rmtree(design_folder)

    def run(self, state, design_names=None, verbose=False):
        if design_names is None:
            design_names = self.base_design_names
        if not design_names:
            return (), (), ()
        args = [
            (state, name, gen_dir, lines, verbose)
            for name, gen_dir, lines in zip(design_names, self.gen_dirs, self.tmp_lines)
        ]
        with ThreadPool(processes=self.num_process) as pool:
            results = pool.starmap(self.create_design_and_simulate, args)
        states, specs, infos = zip(*results)
        return states, specs, infos

    def translate_result(self, output_path):
        """Parse output files in a circuit-specific subclass."""
        raise NotImplementedError
