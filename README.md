# Einops.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Einops.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Einops.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Einops.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Einops.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Einops.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Einops.jl)

> [!WARNING]  
> This package is still under development, and does not implement all of the features of the original (see [Roadmap](#Roadmap)).

Einops.jl is a Julia implementation of the [einops](https://einops.rocks) Python package, providing an elegant and intuitive notation for tensor operations. `rearrange` offers a unified way to perform Julia's `reshape` and `permutedims`, with plans for also implementing `reduce` and `repeat`.

The Python implementation uses strings to specify the operation, but that would be tricky to compile in Julia, so a string macro `@einops_str` is exported for parity, e.g. `einops"a 1 b c -> (c b) a"`, which expands to the form `(:a, 1, :b, :c,) --> ((:c, :b), :a)` where `-->` is an operator that creates an `Einops.Pattern{(:a, 1, :b, :c), ((:c, :b), :a)}`, allowing for compile-time awareness of dimensionalities and permutations—this is not yet taken advantage of, since the tuple types are sufficient for at least ensuring type stability (see [Roadmap](#Roadmap)).

## Operations

### `rearrange`

The `rearrange` combines reshaping and permutation operations into a single, expressive command:

```julia
julia> images = randn(32, 30, 40, 3); # batch, height, width, channel

# reorder axes to "b c h w" format:
julia> rearrange(images, (:b, :h, :w, :c) --> (:b, :c, :h, :w)) |> size
(32, 3, 30, 40)

# flatten each image into a vector
julia> rearrange(images, (:b, :h, :w, :c) --> (:b, (:h, :w, :c))) |> size
(32, 3600)

# split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
julia> rearrange(images, (:b, (:h1, :h), (:w1, :w), :c) --> ((:b, :h1, :w1), :h, :w, :c), h1=2, w1=2) |> size
(128, 15, 20, 3)
```

### `reduce` (Planned)

The `reduce` function will allow for applying reduction operations (like `sum`, `mean`, `maximum`) along specified axes. This is different from typical `Base.reduce` functionality, which reduces using binary operations, but this could still be implemented on top of `Base.reduce` since our methods can dispatch on `Einops.Pattern`.

```julia
# Example (conceptual):
x = randn(100, 32, 64)
y = reduce(maximum, x, (:t, :b, :c) --> (:b, :c)) # max-reduction on the first axis
```

### `repeat` (Planned)

The `repeat` function will provide a concise way to repeat elements along existing or new axes. This can be implemented on top of `Base.repeat` since our methods can dispatch on `Einops.Pattern`.

```julia
# Example (conceptual):
image = randn(30, 40)
rgb_image = repeat(image, (:h, :w) --> (:repeat, :h, :w), repeat=3)
```

## Roadmap

*   [x] Implement `rearrange`.
*   [x] Support Python implementation's string syntax for patterns with string macro.
*   [ ] Support ellipsis notation (using `..` from [EllipsisNotation.jl](https://github.com/SciML/EllipsisNotation.jl)).
*   [ ] Implement `reduce`.
*   [ ] Implement `repeat`.
*   [ ] Explore integration with `PermutedDimsArray` or `TransmuteDims.jl` for lazy and statically inferrable permutations.
*   [ ] Implement `einsum` (or wrap existing implementation) 
*   [ ] Implement `pack`, and `unpack`.

## Contributing

Contributions are welcome! Please feel free to open an issue to report a bug or start a discussion.
