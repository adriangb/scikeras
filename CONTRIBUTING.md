## Contributing
Contributions are very welcome. Please open an issue to ask for new features or preferable a PR to propose an implementation.

If submitting a PR, please make sure that:

- All existing tests should pass. Please make sure that the test
  suite passes, both locally and on
  [Travis CI](https://travis-ci.org/github/adriangb/scikeras).  Status on
  Travis will be visible on a pull request.

- New functionality should include tests. Please write reasonable
  tests for your code and make sure that they pass on your pull request. Testing is done with [`Pytest`](https://docs.pytest.org/en/latest/) and coverage is checked with [`CodeCov`](https://codecov.io/gh/adriangb/scikeras) and a minimum of 94% coverage is required to pass a build.

- Classes, methods, functions, etc. should have docstrings. The first line of a docstring should be a standalone summary. Parameters and return values should be documented explicitly.

Tools required for development are listed in `requirements_dev.txt`.

### Style
- This project follows [the PEP 8
  standard](http://www.python.org/dev/peps/pep-0008/) and uses
  [Black](https://black.readthedocs.io/en/stable/) and
  [Flake8](http://flake8.pycqa.org/en/latest/) to ensure a consistent
  code format throughout the project.

- Imports should be grouped with standard library imports first,
  3rd-party libraries next, and imports from this module third. Within each
  grouping, imports should be alphabetized. Always use absolute
  imports when possible, and explicit relative imports for local
  imports when necessary in tests.

- You can set up [pre-commit hooks](https://pre-commit.com/) to
  automatically run `black` and `flake8` when you make a git
  commit. This can be done by installing `pre-commit`:

    $ python -m pip install pre-commit black flake8

  From the root of the repository, you should then install
  `pre-commit`:

    $ pre-commit install

  Then `black` and `flake8` will be run automatically each time you
  commit changes. You can skip these checks with `git commit
  --no-verify`.

### Deployment
Deployment to PyPi is done automatically by Travis for tagged commits.
To release a new version, you can use `bump2version`:
```bash
bump2version patch --message "Tag commit message"
git push --tags
```
