Welcome to my AI experimentation repository!

At the time of writing, this repo is a Python port of my [Tic-Tac-Clojure](https://github.com/Jamie-Rodriguez/tic-tac-clojure) repo. I created this port because I want to use Google's [JAX](https://github.com/google/jax) library to experiment with neural networks.

In this repo I want to test different approaches to using AI to play games at a high skill level; mostly using different reinforcement learning techniques.

I may create more game environments to experiment with in the future, we will see...

Requirements
============
This repo should be compatible with *any* version of Python 3.

Shell scripts
=============
The shell scripts provided are mostly to help myself remember how to do certain things in the Python ecosystem.

Although the scripts are just references, I do recommend using `run-tests.sh` however. See [Run unit tests](run-unit-tests).

Instructions
============
I run this project using Python's *virtual environments* to isolate the dependencies for this project from the rest of the system.

## 1. Create a Python virtual environment
I name mine `.venv`, using the following command from the *root directory*:
```bash
python3 -m venv .venv
```
The generated `.venv` folder is already added to `.gitignore`

## 2. Activate virtual environment in current shell
From the *root directory*:
```bash
source .venv/bin/activate
```

## 3. Install project dependencies
```bash
python -m pip install --requirement requirements.txt
```

## 4. Run unit tests
I would recommend using `run-tests.sh` because the script will make sure that you provide a specific directory/project, run tests with a coverage report and clean up the `.coverage` report that would otherwise be left behind by `pytest-cov`.

Example usage:
```bash
./run-tests.sh tic-tac-toe
```

Running games
=============
As of time of writing, some example usage is provided in the `main` function of `tic-tac-toe/engine.py`:
```bash
python tic-tac-toe/engine.py
```

To-Do
=====
- Consider using tail-call optimisation for functions that can potentially be unbounded in size, probably using [this library](https://github.com/0scarB/tail-recursive)
- Add linting
- See if there are better test-coverage reporting libraries
- Get around to *to-do*'s in code