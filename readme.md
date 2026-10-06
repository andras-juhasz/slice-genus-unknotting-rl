# Slice genus and unknotting number of links via reinforcement learning

This repository holds the code of the article *Computations of the slice genus and the unknotting number of links via machine learning* by Yutong Dai, Oliver Hayman, András Juhász and Ludovico Morellato.

An agent acts on a link diagram by crossing changes or band moves until the diagram is a diagram of an unlink, and the sequence of moves gives an upper bound on an invariant of the link. The frameworks are the unknotting number (crossing changes), the ribbon genus and the strong ribbon genus (band moves), and the slice genus and the strong slice genus (band moves and births of unknotted components). There are three agents: a policy trained with reinforcement learning (maskable PPO from sb3-contrib), a random walker, and a mixed agent that takes each step either from the policy or from the walker. The bounds pipeline in `src/bounds_pipeline/` computes lower and upper bounds from invariants of the links, so that a value found by an agent can be certified exact.

The datasets, the trained models and the evaluations behind the tables of the article are attached to the release [`v1.0-preprint`](https://github.com/andras-juhasz/slice-genus-unknotting-rl/releases/tag/v1.0-preprint) of this repository.

## Contents

```
src/                    the Nim core (compiled into agent.so and representable.so) and the Python
                        code around it: environments, agents, datasets, experiments
src/bounds_pipeline/    bounds from invariants; needs SageMath, and KnotJob for the s-invariant
datasets/               rolfsen.csv (the 249 knots with at most 10 crossings) and
                        linkinfo-to-9.csv (354 links of LinkInfo with at most 9 crossings)
test/                   unit tests
```

## Getting the code

```
git clone https://github.com/andras-juhasz/slice-genus-unknotting-rl.git
cd slice-genus-unknotting-rl
```

The commands below run from the repository root unless they say otherwise.

## Python environment

The Python dependencies are listed in `environment.yml` and locked for linux-64 in `conda-lock.yml`. Install the locked environment with [conda-lock](https://github.com/conda/conda-lock):

```
conda-lock install --name RLKnotsEnv conda-lock.yml
conda activate RLKnotsEnv
```

`conda env create -f environment.yml` also works, but it resolves the indirect dependencies again instead of using the locked versions. `requirements.txt` lists only the pip packages and does not include SageMath, which the bounds pipeline needs.

**Workaround for sb3_contrib Bug:** Currently, the `sb3_contrib` PyPI package has a bug that causes an error to be raised in PyTorch (`torch/distributions/distribution.py`) at about 500,000 training timesteps. To get around this bug, change line 68 of `sb3_contrib/common/maskable/distributions.py` from
```
super().__init__(logits=logits)
```
to
```
super().__init__(logits=logits, validate_args=False)
```

The error reads `ValueError: Expected parameter probs ... of distribution MaskableCategorical(...) to satisfy the constraint Simplex()`, and with the many actions of the band frameworks it can appear within a few steps of an evaluation. `src/experiment.py` turns off PyTorch's validation of distribution arguments when it is imported, which has the same effect, so the change above is needed only by code that uses `sb3_contrib` without importing `experiment`.

## Nim dependencies

The Nim packages are listed in `requirements.nimble` and locked in `nimble.lock`. From the repository root, install the locked versions and make the compiler use them with

```
nimble install --depsOnly --useSystemNim
nimble setup --useSystemNim
```

`--useSystemNim` keeps the Nim compiler already on your path (the code is tested with Nim 2.2.8) instead of downloading the compiler recorded in the lock.

`nimble setup` writes `nimble.paths` and `config.nims` in the repository root, which tell the compiler where the locked packages are. Do not skip it: when a `nimble.lock` is present, Nim ignores the packages installed in `~/.nimble`, and every compilation stops at `Error: cannot open file: nimpy`.

## Setting up Spherogram-nim

The Nim code depends on [Spherogram-nim](https://github.com/ascchrvalstr/Spherogram-nim), a Nim port of Spherogram, which is not bundled with this repository. Clone it into the repository root, next to `src/`, and check out the commit this code is tested against:

```
git clone https://github.com/ascchrvalstr/Spherogram-nim.git spherogram-nim
cd spherogram-nim
git checkout e2197e6
```

Then make two changes inside `spherogram-nim/`.

1. In `src/config.nim`, set the constant to `false`:

   ```nim
   const reverse_type_ii_check_same_face* = false
   ```

   `join_split_link_diagrams` in `src/state.nim` joins the components of a split diagram with a reverse Reidemeister II move between strands that lie on different faces. With the check left at `true`, every such join raises a `ValueError`.

2. Add the knot Floer homology sources. `src/knot_floer_homology.nim` compiles the C++ program ComputeHFKv2, which Spherogram-nim does not ship. Copy it from [knot_floer_homology](https://github.com/3-manifolds/knot_floer_homology) version 1.2.2 and apply the patch that Spherogram-nim provides:

   ```
   git clone --branch 1.2.2_as_released --depth 1 https://github.com/3-manifolds/knot_floer_homology.git /tmp/knot_floer_homology
   cp -r /tmp/knot_floer_homology/ComputeHFKv2/. src/ComputeHFKv2/
   cd src/ComputeHFKv2
   patch -p1 < Required-Changes.diff
   ```

   The clone in `/tmp` is no longer needed afterwards.

The expected layout is:

```
slice-genus-unknotting-rl/
├── src/
├── test/
└── spherogram-nim/
    └── src/
        ├── config.nim        (reverse_type_ii_check_same_face = false)
        ├── links.nim
        ├── ...
        └── ComputeHFKv2/     (patched)
```

## Setting up KnotJob

The bounds pipeline computes the Rasmussen s-invariant with [KnotJob](https://www.maths.dur.ac.uk/users/dirk.schuetz/knotjob.html) by Dirk Schütz, a Java program distributed under the GPL-3.0. The rest of the code does not need it. Download it and unzip it into `src/`:

```
cd src
curl -LO https://www.maths.dur.ac.uk/users/dirk.schuetz/KnotJob.zip
unzip KnotJob.zip
rm KnotJob.zip
```

This creates `src/KnotJob/KnotJob.jar`, where `src/bounds_pipeline/knotjob_wrapper.py` looks for it. The code is tested with the release of 16 February 2026; earlier releases can compute the s-invariant of a multi-component link with the wrong component orientations. The download is not versioned, so check that you have that release:

```
sha256sum KnotJob/KnotJob.jar
# b63daf5f689660a276ed37bbca3393cb26fe62e097427cdd599647b874304718
```

KnotJob needs Java 23 or later, which the Python environment above provides. To use a different `java`, set `KNOTJOB_JAVA` to its path; the JVM heap is set with `KNOTJOB_JAVA_XMX` (default `8g`). To check the installation, run from the repository root

```
PYTHONPATH=src python src/bounds_pipeline/knotjob_wrapper.py
```

which prints `OK` when KnotJob computes the s-invariants of a few small links correctly.

## Compiling

Set up the Nim dependencies and Spherogram-nim as above first. Then, to compile the Nim code, go to `src/` and run

```
nim -d:release c --app:lib --out:agent.so --threads:on agent.nim
nim -d:release cpp --app:lib --out:representable.so --mm:arc --threads:on representable.nim
```

For code that checks assertions (but it is slower), run

```
nim c --app:lib --out:agent.so --threads:on agent.nim
nim cpp --app:lib --out:representable.so --mm:arc --threads:on representable.nim
```

The setting in `spherogram-nim/src/config.nim` is read at compile time: a library compiled before you changed it keeps the old value, so compile both libraries again after any change in `spherogram-nim/`.

## Running an example

With the environment active and both libraries compiled, run from `src/`

```
python rl-template.py
```

It trains a policy for the slice genus for 8,192 steps on the first three links of `datasets/linkinfo-to-9.csv`, runs the mixed agent on the same links, and prints a table comparing its answers with the slice genera listed in the dataset, followed by the accuracy. The run prints its master seed first; `SEED=<seed> python rl-template.py` repeats it on the same machine.

`bo-template.py` is a longer example: a Bayesian optimisation of the parameters of the random walker for the unknotting number, on the first 25 knots of `datasets/rolfsen.csv`, with 50 evaluations of 50,000 steps each, split between 12 worker processes.

## Running the tests

Run the tests from `test/`, with the environment active and both libraries compiled: they read the datasets through paths relative to that folder. For the Python tests, run

```
PYTHONPATH=../src:. python -m unittest
```

Each Nim test compiles into a binary next to its source. `test_features.nim` and `test_representable.nim` import code that compiles only with the C++ backend, so compile them with `nim cpp`, and the others with `nim c`. The binaries load the PCRE library of the environment at run time, so pass its `lib/` folder when you run them, for example

```
nim c test_state.nim
LD_LIBRARY_PATH=$CONDA_PREFIX/lib ./test_state
nim cpp test_representable.nim
LD_LIBRARY_PATH=$CONDA_PREFIX/lib ./test_representable
```

Set `LD_LIBRARY_PATH` only to run a test, not to compile it: with the environment's libraries on that path, the system compiler can fail.

## License

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 2 of the License, or (at your option) any later version. It is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See [LICENSE](LICENSE) for the full text.

`src/jones_port.nim` is a translation into Nim of code from [SageMath](https://www.sagemath.org/), copyright Miguel Angel Marco Buzunariz and Amit Jamadagni, released under the same terms. `src/theta.nim` translates into Nim [Theta.sage](https://www.rolandvdv.nl/Theta/), Roland van der Veen's Sage implementation of the Theta invariant of Dror Bar-Natan and Roland van der Veen ([arXiv:2509.18456](https://arxiv.org/abs/2509.18456)); it is used with the permission of Roland van der Veen and Dror Bar-Natan, under the terms of [Dror Bar-Natan's copyleft notice](https://www.math.toronto.edu/~drorbn/Copyleft/). The Nim libraries are compiled together with [Spherogram-nim](https://github.com/ascchrvalstr/Spherogram-nim), which is distributed under the GNU General Public License, version 2.