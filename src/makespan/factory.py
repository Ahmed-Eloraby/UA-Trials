"""Factory and pipeline for orientation algorithms.

An algorithm is a sequence of steps that share a :class:`SolveContext`.
Register a new algorithm with :meth:`AlgorithmFactory.register` and return a
:class:`Pipeline`. Existing steps can be reused without changing them::

    @AlgorithmFactory.register("my-algo")
    def _my_algo() -> Pipeline:
        return Pipeline("my-algo", [GreedyLightestStep(), FlowSearchStep()])

``AlgorithmFactory.create("custom", steps=[...])`` builds a one-off pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

from makespan.context import SolveContext
from makespan.model import Instance, Solution


class AlgorithmLimitError(RuntimeError):
    """Raised when an algorithm refuses an instance that is too large for it."""


class Step(ABC):
    """One transformation of a :class:`SolveContext`.

    Steps may lock edges, write a fractional solution, or propose a complete
    orientation through :meth:`SolveContext.consider`. A step that only
    touches unlocked edges can sit in the middle of a longer pipeline.
    """

    @abstractmethod
    def run(self, ctx: SolveContext) -> None:
        """Read and update ``ctx``. Do not rebuild the instance.

        Args:
            ctx: Shared solve state that this step may lock, round, or bound.
        """


class Algorithm(ABC):
    """Algorithm that turns an instance into a solution.

    Attributes:
        name: Report name written on the returned :class:`Solution`.
    """

    name: str

    @abstractmethod
    def solve(self, instance: Instance) -> Solution:
        """Return a complete orientation and its evaluated makespan.

        Args:
            instance: Graph whose edges this algorithm orients.

        Returns:
            Solution: Complete orientation and its evaluated makespan.
        """


class Pipeline(Algorithm):
    """Run steps in order on a fresh context and return the incumbent.

    Args:
        name: Label written on the returned :class:`Solution`.
        steps: Transformations applied in order on a fresh context.
    """

    def __init__(self, name: str, steps: list[Step]):
        """Store the report name and a private copy of the step list.

        Args:
            name: Label written on the returned :class:`Solution`.
            steps: Transformations applied in order on a fresh context.
        """
        self.name = name
        self.steps = list(steps)

    def solve(self, instance: Instance) -> Solution:
        """Run each step on a fresh context and return the best orientation.

        Args:
            instance: Graph whose edges this pipeline orients.

        Returns:
            Solution: Best orientation found by the steps.
        """
        ctx = SolveContext(instance)
        for step in self.steps:
            step.run(ctx)
        return ctx.to_solution(self.name)


class AlgorithmFactory:
    """Create a registered algorithm by name.

    Builders are callables, so each call can take its own limits and
    thresholds without sharing mutable settings.
    """

    _builders: dict[str, Callable[..., Algorithm]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[..., Algorithm]], Callable[..., Algorithm]]:
        """Decorator that stores a builder under ``name``.

        Registering the same name twice is an error, so a typo cannot silently
        replace an existing algorithm.

        Args:
            name: Command-line name under which the builder is stored.

        Returns:
            Callable: Decorator that stores a builder under ``name``.
        """
        def decorate(builder: Callable[..., Algorithm]) -> Callable[..., Algorithm]:
            """Store ``builder`` and return it unchanged.

            Args:
                builder: Callable that returns a fresh :class:`Algorithm`.

            Returns:
                Callable[..., Algorithm]: The same builder, now registered.
            """
            if name in cls._builders:
                raise ValueError(f"algorithm {name!r} is already registered")
            cls._builders[name] = builder
            return builder

        return decorate

    @classmethod
    def create(cls, algorithm: str, **config) -> Algorithm:
        """Build a fresh algorithm.

        ``algorithm`` is the registered name. ``config`` is passed to that
        builder, so a pipeline can still take its own ``name`` argument.

        Args:
            algorithm: Registered name of the builder to call.
            **config: Keyword arguments forwarded to that builder.

        Returns:
            Algorithm: Fresh algorithm built from the registered factory.
        """
        try:
            builder = cls._builders[algorithm]
        except KeyError as exc:
            known = ", ".join(cls.available()) or "(none)"
            raise KeyError(f"unknown algorithm {algorithm!r}; known: {known}") from exc
        return builder(**config)

    @classmethod
    def available(cls) -> list[str]:
        """Sorted names that :meth:`create` will accept, including ``custom``.

        Returns:
            list[str]: Registered algorithm names, sorted alphabetically.
        """
        return sorted(cls._builders)
