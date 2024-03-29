# SciKeras Docs Guide

Docs are built with [sphinx](https://www.sphinx-doc.org/en/master/) and deployed to the `docs-deploy` branch via the [docs workflow](../.github/workflows/docs.yaml).

The documentation is mostly written in rst, but executable notebooks are written in markdown and processed into HTML using [jupytext](https://github.com/mwouts/jupytext) and [nbsphinx](https://nbsphinx.readthedocs.io/en/0.8.1/).
This ensures that diffs are readable (GitHub is pretty good about highlighting markdown, and the outputs of execution are excluded) and that the execution is consistent and does not generate any errors.
This is all done via the [docs workflow](../.github/workflows/docs.yaml).

## Working with .md notebooks

To make edits to the markdown notebooks, you will need install jupytext with jupyter, which will then allow you to open and execute markdown notebooks as described [here](https://jupytext.readthedocs.io/en/latest/paired-notebooks.html).

```bash
poetry run jupyter notebook
```

## Building docs locally

To build the docs, run the following from the project root directory

```bash
export TF_CPP_MIN_LOG_LEVEL=3
find docs/source/notebooks -maxdepth 1 -type f -name '*.md' -print0 | xargs -0 -n 1 -P 8 poetry run jupytext --set-formats ipynb,md --execute
poetry run sphinx-build -b html -j 8 docs/source docs/_build
```

Note that you can change `-P 8` and `-j 8` to match the number of cores your processor has. Or to run on a single core:

```bash
export TF_CPP_MIN_LOG_LEVEL=3
poetry run jupytext --set-formats ipynb,md --execute docs/source/notebooks/*.md
poetry run sphinx-build -b html docs/source docs/_build
```

You can now load the docs from [docs/_build/index.html](_build/index.html).
