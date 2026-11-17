"""Generate reproducible evaluation targets for a circuit benchmark."""

import argparse
import pickle
import random
import sys
from collections import OrderedDict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analog_sizing.config import (
    BENCHMARKS,
    SPEC_DIR,
    benchmark_config,
    load_config,
    resolve_path,
)


def gen_data(cir_yaml, env, num_specs, *, seed=None, output=None):
    """Write targets in the format consumed by the environment."""
    if num_specs < 1:
        raise ValueError("num_specs must be positive")
    rng = random.Random(seed)
    specs = OrderedDict()
    for name, (low, high) in load_config(cir_yaml)["target_specs"].items():
        if isinstance(low, int):
            values = [rng.randint(int(low), int(high)) for _ in range(num_specs)]
        else:
            values = [rng.uniform(float(low), float(high)) for _ in range(num_specs)]
        specs[name] = tuple(values)
    output_path = (
        resolve_path(output)
        if output is not None
        else SPEC_DIR / f"ngspice_specs_gen_{env}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(specs, stream)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", choices=BENCHMARKS, default="TSA")
    parser.add_argument("--num-specs", "--num_specs", type=int, required=True)
    parser.add_argument("--yaml", help="Optional benchmark YAML override")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", help="Output pickle path relative to the repository"
    )
    args = parser.parse_args()
    if args.num_specs < 1:
        parser.error("--num-specs must be positive")
    config = benchmark_config(args.env)
    path = gen_data(
        args.yaml or config["CIR_YAML"],
        f"{args.env.lower()}_{args.num_specs}",
        args.num_specs,
        seed=args.seed,
        output=args.output,
    )
    print(f"Saved {args.num_specs} targets to {path}")


if __name__ == "__main__":
    main()
