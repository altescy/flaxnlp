import json
import os
from logging import getLogger
from os import PathLike
from typing import Any, Dict, Literal, Optional, Tuple, Type, TypeVar, Union, overload

from colt.builder import ColtBuilder

logger = getLogger(__name__)


try:
    from _jsonnet import evaluate_file
except ImportError:

    def evaluate_file(filename: str, **_kwargs: Any) -> str:
        logger.warning("jsonnet is unavailable, treating %{filename}s as plain json")
        with open(filename, "r") as evaluation_file:
            return evaluation_file.read()


T = TypeVar("T")
_colt_builder = ColtBuilder(typekey="type")


def load_jsonnet(
    filename: Union[str, PathLike],
    ext_vars: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    def _is_encodable(value: str) -> bool:
        return (value == "") or (value.encode("utf-8", "ignore") != b"")

    def _environment_variables() -> Dict[str, str]:
        return {key: value for key, value in os.environ.items() if _is_encodable(value)}

    ext_vars = {**_environment_variables(), **(ext_vars or {})}
    jsondict = json.loads(evaluate_file(str(filename), ext_vars=ext_vars))  # type: Dict[str, Any]
    return jsondict


@overload
def load_config_from_file(
    filename: Union[str, PathLike],
    cls: Literal[None] = ...,
) -> Tuple[Any, Dict[str, Any]]:
    ...


@overload
def load_config_from_file(
    filename: Union[str, PathLike],
    cls: Type[T],
) -> Tuple[T, Dict[str, Any]]:
    ...


def load_config_from_file(
    filename: Union[str, PathLike],
    cls: Optional[Type[T]] = None,
    *,
    ext_vars: Optional[Dict[str, Any]] = None,
) -> Tuple[Union[T, Any], Dict[str, Any]]:
    config = load_jsonnet(filename, ext_vars=ext_vars)
    return _colt_builder(config, cls), config
