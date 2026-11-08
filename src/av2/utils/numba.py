"""Signature-preserving typing for the Numba decorators used by AV2."""

from typing import Any, Callable, Optional, TypeVar, Union, cast, overload

import numba

F = TypeVar("F", bound=Callable[..., Any])


@overload
def njit(func: F) -> F: ...


@overload
def njit(*, nogil: bool = False) -> Callable[[F], F]: ...


def njit(
    func: Optional[F] = None, *, nogil: bool = False
) -> Union[F, Callable[[F], F]]:
    """Compile a function while retaining its annotated signature for callers."""
    decorator = numba.njit(nogil=nogil)
    if func is None:
        return cast(Callable[[F], F], decorator)
    return cast(F, decorator(func))
