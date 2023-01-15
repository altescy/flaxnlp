from typing import Any, Callable, List, Optional

import flax
import jax

Array = Any


def sequence_mask(lengths: Array, max_length: int) -> Array:
    """Computes a boolean mask over sequence positions for each given length.
    Example:
    ```
    sequence_mask([1, 2], 3)
    [[True, False, False],
     [True, True, False]]
    ```
    Args:
      lengths: The length of each sequence. <int>[batch_size]
      max_length: The width of the boolean mask. Must be >= max(lengths).
    Returns:
      A mask with shape: <bool>[batch_size, max_length] indicating which
      positions are valid for each sequence.
    """
    return jax.numpy.arange(max_length)[None] < lengths[:, None]


@jax.vmap
def flip_sequences(inputs: Array, lengths: Array) -> Array:
    """Flips a sequence of inputs along the time dimension.
    This function can be used to prepare inputs for the reverse direction of a
    bidirectional LSTM. It solves the issue that, when naively flipping multiple
    padded sequences stored in a matrix, the first elements would be padding
    values for those sequences that were padded. This function keeps the padding
    at the end, while flipping the rest of the elements.
    Example:
    ```python
    inputs = [[1, 0, 0],
              [2, 3, 0]
              [4, 5, 6]]
    lengths = [1, 2, 3]
    flip_sequences(inputs, lengths) = [[1, 0, 0],
                                       [3, 2, 0],
                                       [6, 5, 4]]
    ```
    Args:
      inputs: An array of input IDs <int>[batch_size, seq_length].
      lengths: The length of each sequence <int>[batch_size].
    Returns:
      An ndarray with the flipped inputs.
    """
    # Note: since this function is vmapped, the code below is effectively for
    # a single example.
    max_length = inputs.shape[0]
    return jax.numpy.flip(jax.numpy.roll(inputs, max_length - lengths, axis=0), axis=0)


def concat(operation: str, *arrays: Array, axis: int = -1) -> Array:
    if not operation:
        raise ValueError("operation must be specified")
    results: List[Array] = []
    for suboperation in operation.split(";"):
        if "*" in suboperation:
            lop, rop = suboperation.split("*", 1)
            results.append(concat(lop, *arrays, axis=axis) * concat(rop, *arrays, axis=axis))
        elif "/" in suboperation:
            lop, rop = suboperation.split("/", 1)
            results.append(concat(lop, *arrays, axis=axis) / concat(rop, *arrays, axis=axis))
        elif "+" in suboperation:
            lop, rop = suboperation.split("+", 1)
            results.append(concat(lop, *arrays, axis=axis) + concat(rop, *arrays, axis=axis))
        elif "-" in suboperation:
            lop, rop = suboperation.split("-", 1)
            results.append(concat(lop, *arrays, axis=axis) - concat(rop, *arrays, axis=axis))
        elif suboperation.isdigit():
            results.append(arrays[int(suboperation)])
        else:
            raise ValueError(f"Invalid operation: {suboperation}")
    return jax.numpy.concatenate(results, axis=axis)


def stop_gradient(
    variables: Any,
    filter_fn: Optional[Callable[[str], bool]] = None,
) -> Any:
    filter_fn = filter_fn or (lambda _: True)
    flat_vars = flax.traverse_util.flatten_dict(variables)  # type: ignore[no-untyped-call]
    new_vars = {k: jax.lax.stop_gradient(v) if filter_fn(k) else v for k, v in flat_vars.items()}
    return flax.traverse_util.unflatten_dict(new_vars)  # type: ignore[no-untyped-call]
