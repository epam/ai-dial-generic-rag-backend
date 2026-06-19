from types import get_original_bases, resolve_bases
from typing import Generic, TypeVar, get_args

from typing_inspect import get_parameters


def resolve_generic_arg(cls_or_instance: type | object, target_type: type, index: int) -> type | TypeVar | None:
    """
    Resolves the concrete type bound to a particular generic type parameter in a class hierarchy.

    This function is useful when working with Python's typing generics, where you may wish
    to introspect and determine the actual type parameter (e.g., the type of `T` in `Generic[T]`)
    provided to a class either at declaration or at runtime via instantiation.

    Args:
        cls_or_instance (type | object): The class or object whose generic argument is to be resolved.
            - If an object is provided, the function attempts to use its type and,
              if available, its `__orig_class__` attribute for more precise resolution.
        target_type (type): The target generic base type (e.g., `List`, `Mapping`) for
            which the generic argument is to be resolved.
        index (int): The index of the parameter in the generic base type to resolve.

    Returns:
        type | TypeVar | None: The concrete type bound to the specified generic parameter,
        the TypeVar if the parameter cannot be fully resolved, or `None` if not found.

    Example:
        >>> from typing import Generic, TypeVar, List
        >>> T = TypeVar('T')
        >>> class MyList(List[int]): pass
        >>> resolve_generic_arg(MyList, List, 0)
        <class 'int'>

    Notes:
        - If a type variable remains unbound or cannot be resolved, it may be returned as a `TypeVar`.
        - This function walks the base classes using both `__orig_bases__` and runtime attributes
          to handle both declared and instantiated generics (including those using PEP 560).
        - Handles both type objects and instances, including those with runtime-specified
          type arguments (e.g., typing generics with __orig_class__).

    Caveats:
        - This function involves introspection and may not work as expected with all forms of
          Python typing or metaclass trickery.

    """
    if not isinstance(cls_or_instance, type):
        return _resolve_generic_arg_for_object(cls_or_instance, target_type, index)

    for base in get_original_bases(cls_or_instance):
        base_type = resolve_bases([base])[0]

        if base_type is target_type:
            args = get_args(base)
            return args[index]

        if not issubclass(base_type, target_type):
            continue

        result = resolve_generic_arg(base_type, target_type, index)

        if not isinstance(result, TypeVar):
            return result

        generic_type = _resolve_generic_type(base_type)

        for generic_arg, base_arg in zip(get_args(generic_type), get_args(base), strict=False):
            if generic_arg is result:
                return base_arg

        if isinstance(result, TypeVar):
            return result

    return None


def _resolve_generic_arg_for_object(obj: object, target_type: type, index: int) -> type | None | TypeVar:
    result = resolve_generic_arg(type(obj), target_type, index)

    if isinstance(result, TypeVar) and hasattr(obj, "__orig_class__"):
        for param, arg in zip(get_parameters(type(obj)), get_args(obj.__orig_class__), strict=False):
            if param is result:
                return arg

    return result


def _resolve_generic_type(type_: type):
    """
    Resolves and returns the direct base class from which a given class inherits `Generic`.

    This function inspects the original bases of the provided class and returns
    the one that is a direct generic base (i.e., whose resolved base is `Generic`).
    It is commonly used in type introspection to identify which base class defines the
    type's type variables (e.g., `Generic[T]`).

    Args:
        type_ (type): The class to inspect for its direct `Generic` base.

    Returns:
        type: The base class of `type_` that directly inherits from `Generic`.
              Typically something like `Generic[T]` or `Generic[T, U]`.

    Raises:
        RuntimeError: If no base class of `type_` is found that directly inherits `Generic`.

    Example:
        >>> from typing import Generic, TypeVar
        >>> T = TypeVar('T')
        >>> class MyClass(Generic[T]): pass
        >>> _resolve_generic_type(MyClass)
        Generic[~T]
    """
    for item in get_original_bases(type_):
        if resolve_bases([item])[0] is Generic:
            return item
    raise RuntimeError("unable to find generic type")
