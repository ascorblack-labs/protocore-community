"""The reusable body of a contract conformance suite.

A contract in this core is a Protocol: a shape the core calls and a host
supplies. Nothing checks that the host's object really has that shape until the
core calls it, and by then the call is inside a run — so a method the adapter
spells slightly differently, or wrote synchronously where the contract is
asynchronous, surfaces as a failure of whatever the agent happened to be doing.

A conformance suite closes that gap by asking the questions ahead of time, in
the host's own test run, against the host's own object. The suite is written
once, here, and each contract gets a thin subclass naming its Protocol. A host
binds one to an adapter by subclassing it with a ``subject_factory``:

.. code-block:: python

    from protocore.conformance import SessionStoreConformance

    class TestMyStore(SessionStoreConformance):
        @pytest.fixture
        def subject_factory(self):
            return lambda: MyStore(dsn="...")

The factory, not an instance, is the parameter: a suite may want more than one
subject, and building each inside the test keeps one case's state out of the
next.

What a suite checks is deliberately structural — presence, asynchrony, and the
parameters the core passes by name. It does not check behaviour, because
behaviour is the host's own test suite's job and a core that asserted it would
be dictating storage semantics it has no business knowing.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, ClassVar, NoReturn

import pytest


def declared_members(protocol: type) -> tuple[str, ...]:
    """The member names ``protocol`` declares, in sorted order.

    Read from the Protocol's own class body and annotations rather than from
    ``dir()``, so the object machinery every class carries stays out of it.
    """
    names: set[str] = set()
    for base in protocol.__mro__:
        if base in (object,) or base.__name__ == "Protocol":
            continue
        names.update(
            name for name in vars(base) if not name.startswith("_")
        )
        names.update(
            name
            for name in getattr(base, "__annotations__", {})
            if not name.startswith("_")
        )
    return tuple(sorted(names))


def _protocol_member(protocol: type, name: str) -> Any:
    for base in protocol.__mro__:
        if name in vars(base):
            return vars(base)[name]
    return None


def _accepted_parameters(func: Any) -> tuple[set[str], bool]:
    """Parameter names ``func`` accepts, and whether it accepts arbitrary ones."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - builtins without signatures
        return set(), True
    names = set()
    open_ended = False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            open_ended = True
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            open_ended = True
        else:
            names.add(parameter.name)
    return names, open_ended


def _required_parameters(func: Any) -> set[str]:
    """Parameter names the contract declares, excluding ``self`` and varargs."""
    names, _ = _accepted_parameters(func)
    return names - {"self", "cls"}


class ContractSuite:
    """Base class for a contract's conformance suite.

    Subclasses set :attr:`protocol`. A host binds the suite to its adapter by
    overriding the ``subject_factory`` fixture; without one the suite skips
    rather than fails, so importing a suite it has not bound yet costs a host
    nothing.
    """

    #: The Protocol this suite is about.
    protocol: ClassVar[type]

    @staticmethod
    def unbound() -> NoReturn:
        """Skip: this suite was imported but never given a subject."""
        pytest.skip(
            "no subject bound: override the 'subject_factory' fixture with a "
            "zero-argument callable returning the adapter under test"
        )

    @pytest.fixture
    def subject_factory(self) -> Callable[[], Any]:
        return self.unbound()

    @pytest.fixture
    def subject(self, subject_factory: Callable[[], Any]) -> Any:
        return subject_factory()

    def test_the_adapter_is_recognised_as_the_contract(self, subject: Any) -> None:
        """The structural check the core's own type checking rests on."""
        assert isinstance(subject, self.protocol), (
            f"{type(subject).__name__} is not recognised as "
            f"{self.protocol.__name__}; the members it is missing are "
            f"{sorted(set(declared_members(self.protocol)) - set(dir(subject)))}"
        )

    def test_every_declared_member_is_present(self, subject: Any) -> None:
        missing = [
            name for name in declared_members(self.protocol) if not hasattr(subject, name)
        ]
        assert not missing, (
            f"{type(subject).__name__} does not provide "
            f"{self.protocol.__name__}.{{{', '.join(missing)}}}"
        )

    def test_asynchronous_members_stay_asynchronous(self, subject: Any) -> None:
        """A contract method declared ``async`` that an adapter wrote
        synchronously returns a value where the core awaits one. It type-checks
        against ``Any`` and fails at the call, inside a run."""
        wrong: list[str] = []
        for name in declared_members(self.protocol):
            declared = _protocol_member(self.protocol, name)
            if not inspect.iscoroutinefunction(declared):
                continue
            if not inspect.iscoroutinefunction(getattr(subject, name, None)):
                wrong.append(name)
        assert not wrong, (
            f"{type(subject).__name__} implements these "
            f"{self.protocol.__name__} members synchronously, but the contract "
            f"declares them async: {wrong}"
        )

    def test_asynchronous_iterators_stay_asynchronous_iterators(self, subject: Any) -> None:
        """The same failure one level down: a contract that yields cannot be
        satisfied by something that returns the whole sequence at once."""
        wrong: list[str] = []
        for name in declared_members(self.protocol):
            declared = _protocol_member(self.protocol, name)
            if not inspect.isasyncgenfunction(declared):
                continue
            if not inspect.isasyncgenfunction(getattr(subject, name, None)):
                wrong.append(name)
        assert not wrong, (
            f"{type(subject).__name__} implements these "
            f"{self.protocol.__name__} members without yielding, but the "
            f"contract declares them async generators: {wrong}"
        )

    def test_synchronous_members_stay_synchronous(self, subject: Any) -> None:
        """The same mistake the other way round, and the quieter half of it. A
        contract member the core calls without awaiting, written ``async`` by
        the adapter, returns a coroutine nobody awaits: the call is a silent
        no-op, the value the core reads is a coroutine object, and unlike the
        opposite mistake nothing raises.

        An adapter that writes an async generator where the contract returns
        an async iterator is not this mistake — a call to either gives the
        core the same thing to iterate — so only a coroutine function is
        counted."""
        wrong: list[str] = []
        for name in declared_members(self.protocol):
            declared = _protocol_member(self.protocol, name)
            if not callable(declared) or isinstance(declared, type):
                continue
            if inspect.iscoroutinefunction(declared) or inspect.isasyncgenfunction(declared):
                continue
            if inspect.iscoroutinefunction(getattr(subject, name, None)):
                wrong.append(name)
        assert not wrong, (
            f"{type(subject).__name__} implements these "
            f"{self.protocol.__name__} members asynchronously, but the contract "
            f"declares them synchronous, so the core never awaits them: {wrong}"
        )

    def test_every_member_accepts_the_parameters_the_core_passes(self, subject: Any) -> None:
        """The core calls these by keyword. An adapter that renamed a parameter
        satisfies every structural check above and raises ``TypeError`` on the
        first call — again, inside a run."""
        problems: list[str] = []
        for name in declared_members(self.protocol):
            declared = _protocol_member(self.protocol, name)
            if not callable(declared) or isinstance(declared, type):
                continue
            implementation = getattr(subject, name, None)
            if not callable(implementation):
                continue
            accepted, open_ended = _accepted_parameters(implementation)
            if open_ended:
                continue
            missing = sorted(_required_parameters(declared) - accepted)
            if missing:
                problems.append(f"{name} does not accept {missing}")
        assert not problems, (
            f"{type(subject).__name__} cannot be called the way the core calls "
            f"{self.protocol.__name__}: {problems}"
        )


def binds_a_subject(suite: type[ContractSuite]) -> bool:
    """Whether ``suite`` has been given something to run against.

    A suite with no subject skips, and a directory of skips exits zero — which
    reads in a pipeline as "the adapters conform" when what happened is that
    nobody bound them.
    """
    for base in suite.__mro__:
        if "subject_factory" in vars(base):
            return base is not ContractSuite
    return False


def bound_suites(namespace: Any) -> frozenset[type[ContractSuite]]:
    """Every suite subclass in ``namespace`` that binds a subject.

    ``namespace`` is a module — a host's own conformance package, typically
    passed as ``sys.modules[__name__]``. The result is what a host asserts
    against the contracts it implements, so that forgetting to bind one is a
    failure rather than a skip.
    """
    found: set[type[ContractSuite]] = set()
    for value in vars(namespace).values():
        if not isinstance(value, type) or not issubclass(value, ContractSuite):
            continue
        if getattr(value, "protocol", None) is None:
            continue
        if binds_a_subject(value):
            found.add(value)
    return frozenset(found)


def bound_contracts(namespace: Any) -> frozenset[type]:
    """The contracts ``namespace`` binds a suite to."""
    return frozenset(suite.protocol for suite in bound_suites(namespace))
