from collections.abc import Callable
from dataclasses import field
from typing import Generic, overload

from plum import dispatch

from omniplanner.functor import FunctorTrait, T, dispatchable_parametric, fmap


class Wrapper(FunctorTrait):
    pass


@overload
@dispatch
def fmap(fn: Callable, wrapper: Wrapper):
    return with_new_value(wrapper, fn(extract(wrapper)))


@dispatch
def push(x: Wrapper):
    """Swap nested wrappers, such as SymbolicContext[RobotWrapper[T]]."""
    inner_wrapper = extract(x)
    if not isinstance(inner_wrapper, Wrapper) and not isinstance(inner_wrapper, list):
        raise TypeError("Can only push type Wrapper[Wrapper[T]]")
    return fmap(lambda val: with_new_value(x, val), inner_wrapper)


@dispatchable_parametric
class RobotWrapper(Wrapper, Generic[T]):
    name: str
    value: T


@overload
@dispatch
def fmap(fn: Callable, robot_wrapper: RobotWrapper):
    return RobotWrapper(robot_wrapper.name, fn(robot_wrapper.value))


@overload
@dispatch
def extract(x: RobotWrapper):
    return x.value


@overload
@dispatch
def with_new_value(x: RobotWrapper, value):
    return RobotWrapper(x.name, value)


@dispatchable_parametric
class MultiRobotWrapper(Wrapper, Generic[T]):
    names: list[str]
    value: T
    _outer_name_to_inner_name: dict = field(default_factory=dict)
    _inner_name_to_outer_name: dict = field(default_factory=dict)

    def set_name_remap(self, name, inner_name):
        assert name in self.names
        self._outer_name_to_inner_name[name] = inner_name
        self._inner_name_to_outer_name[inner_name] = name

    def remap_name_to_inner(self, name):
        if name not in self.names:
            return None
        if name in self._outer_name_to_inner_name:
            return self._outer_name_to_inner_name[name]
        return name

    def remap_name_to_outer(self, name):
        if name in self._inner_name_to_outer_name:
            return self._inner_name_to_outer_name[name]
        return None


@overload
@dispatch
def extract(x: MultiRobotWrapper):
    return x.value


@overload
@dispatch
def with_new_value(x: MultiRobotWrapper, value):
    return MultiRobotWrapper(
        x.names,
        value,
        x._outer_name_to_inner_name,
        x._inner_name_to_outer_name,
    )


@overload
@dispatch
def fmap(fn: Callable, multi_robot_wrapper: MultiRobotWrapper):
    return MultiRobotWrapper(
        multi_robot_wrapper.names,
        fn(multi_robot_wrapper.value),
        multi_robot_wrapper._outer_name_to_inner_name,
        multi_robot_wrapper._inner_name_to_outer_name,
    )


@dispatchable_parametric
class SymbolicContext(Wrapper, Generic[T]):
    context: dict
    value: T


@dispatch
def extract(x: SymbolicContext):
    return x.value


@dispatch
def with_new_value(x: SymbolicContext, value):
    return SymbolicContext(x.context, value)


@dispatch
def fmap(fn: Callable, v: SymbolicContext):
    return SymbolicContext(v.context, fn(v.value))
