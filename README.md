Welcome to my AI experimentation repository!

In this repo I want to test different approaches to using AI to play games at a high skill level; mostly using different reinforcement learning techniques.

At the time of writing, this repo is a Python port of my [Tic-Tac-Clojure](https://github.com/Jamie-Rodriguez/tic-tac-clojure) repo. I created this port because I want to use Google's [JAX](https://github.com/google/jax) library to experiment with neural networks.

This codebase was written with `Python 3.8` and so may use some slightly older features with the `Typing` module and lacks support for [pattern matching](https://peps.python.org/pep-0634).

I may create more game environments to experiment with in the future, we will see...

Requirements
============
This repo should be compatible with *any* version of Python 3.

Shell scripts
=============
The shell scripts provided are mostly to help myself remember how to do certain things in the Python ecosystem.

Although the scripts are just references, I do recommend using `run-tests.sh` however. See [Run unit tests](#4-run-unit-tests).

Setup
============
I run this project using Python's *virtual environments* to isolate the dependencies for this project from the rest of the system.

## 1. Create a Python virtual environment
I name mine `.venv`, using the following command from the *root directory*:
```shell
python3 -m venv .venv
```
The generated `.venv` folder is already added to `.gitignore`

## 2. Activate virtual environment in current shell
From the *root directory*:
```shell
source .venv/bin/activate
```

## 3. Install project dependencies
```shell
python -m pip install --requirement requirements.txt
```

## 4. Run unit tests
I would recommend using `run-tests.sh` because the script will make sure that you provide a specific directory/project, run tests with a coverage report and clean up the `.coverage` report that would otherwise be left behind by `pytest-cov`.

Example usage:
```shell
./run-tests.sh tictactoe
./run-tests.sh mcts
```

Running games
=============
Some example usage of setting up a Monte Carlo Tree Search agent vs a random move agent is provided in `play-tic-tac-toe.py`:
```shell
python play-tic-tac-toe.py
```

Artificial Intelligence Agents
==============================
At the moment there is an agent that uses Monte Carlo Tree Search (MCTS) to intelligently pick the best move for a given state. I have abstracted implementation details to be as game-agnostic as possible and should be able to be used for other games given that you can provide the following functions that have the following shapes:
- `get_valid_moves_list :: (State) -> list[Move]`
- `is_terminal :: (State) -> bool`
- `apply_move_to_state :: (State, Move) -> State`
- `check_win :: (State) -> int | None`

Performance
===========
Below compare the performance of the Python code in this repo to that of my *naïve* [Clojure version](https://github.com/Jamie-Rodriguez/tic-tac-clojure), where "naïve" means I wrote the code once, with no attempt at optimisation and have not written any concurrent processing.

Initially I translated the Clojure code straight into Python, including preserving the paradigm where functional programming languages use recursion for all looping. However when running large numbers of Monte Carlo simulations, I needed to incorporate the [tail_recursive](https://pypi.org/project/tail-recursive) library to prevent stack overflows. This actually slowed down the performance significantly - 30% slower!

I found this initial naïve Python code to be extremely slow, and so rewrote the tail-call recursive loop in the main Monte Carlo Tree Search loop into a range-based `for` loop and also compared against using a `while` loop. Both increased the speed approximately 2.5×.

To see if I could get additional performance gains, I then rewrote the rest of the tail-call recursive loops in the Monte Carlo Tree Search code into `for`/`while` loops. I would say that the performance improvement was negligible in this case.

The results below are from running 1,000 games of an *agent that picks random moves* vs a *MCTS agent* with `exploration = 1.2` and `number of iterations = 1,000`. This was run on an Apple M1 Max processor:

|                 Optimisation                |  Speed |
|:-------------------------------------------:|:------:|
|                Clojure naïve                |    48s |
|                 Python naïve                | 7m 19s |
|      Python naïve with `tail_recursive`     | 9m 29s |
|     Python MCTS main loop = `while` loop    | 3m 49s |
|      Python MCTS main loop = `for` loop     | 3m 46s |
| Python All MCTS loops = `for`/`while` loops | 3m 22s |

To-Do
=====
- Add linting
- See if there are better test-coverage reporting libraries
- Parallelise MCTS AI
- Get around to *to-do*'s in code
