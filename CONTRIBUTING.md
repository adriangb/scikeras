# Developer Setup

You will need Poetry to start contributing on the SciKeras codebase. Refer to the [documentation](https://python-poetry.org/docs/#introduction) to start using Poetry.

You will first need to clone the repository using `git` and place yourself in its directory:

```bash
$ git clone git@github.com:adriangb/scikeras.git
$ cd scikeras
```

Now, you will need to install the required dependency for Poetry and be sure that the current
tests are passing on your machine:

```bash
$ poetry install
$ poetry run pytest tests/
```

SciKeras uses the [black](https://github.com/psf/black) and
[isort](https://github.com/timothycrosley/isort) for coding style and you must ensure that your
code follows it. If not, the CI will fail and your Pull Request will not be merged.

To make sure that you don't accidentally commit code that does not follow the coding style, you can
install a pre-commit hook that will check that everything is in order:

```bash
$ poetry run pre-commit install
```

You can also run it anytime using:

```bash
$ poetry run pre-commit run --all-files
```

Your code must always be accompanied by corresponding tests, if tests are not present your code
will not be merged.

# Deployment

Deployment to PyPi is done automatically by GitHub Actions for tagged commits.
