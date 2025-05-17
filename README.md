<p align="center">
  <img src="./docs/src/assets/logo-dark.png" width="256" />
</p>

<h1 align="center">Einops.jl</h1>

<div align="center">

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Einops.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Einops.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Einops.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Einops.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Einops.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Einops.jl)

</div>

Einops.jl is a Julia implementation of [einops](https://einops.rocks), providing a concise notation for tensor operations, and unifying Julia's `reshape`, `permutedims`, `reduce` and `repeat` functions, with support for automatic differentiation.

The Python implementation uses strings to specify the operation, which is tricky to compile in Julia, so a string macro is exported for parity, e.g. `einops"(a b) 1 c -> (c b) a"` expands to the form `((:a, :b), 1, :c,) --> ((:c, :b), :a)`, where `-->` is a custom operator that puts the left and right operands as type parameters of a special pattern type. This allows for compile-time awareness of dimensionalities, ensuring type stability.

## Operations

### `rearrange`

The `rearrange` combines reshaping and permutation operations into a single, expressive command.

```julia
julia> images = randn(3, 40, 30, 32); # channel, width, height, batch

# reorder axes to "b c h w" format:
julia> rearrange(images, (:c, :w, :h, :b) --> (:w, :h, :c, :b)) |> size
(40, 30, 3, 32)

# flatten each image into a vector
julia> rearrange(images, (c, :w, :h, :b) --> ((:c, :w, :h), :b)) |> size
(32, 3600)

# split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
julia> rearrange(images, (:c, (:w, :w2), (:h, :h2), :b) --> (:c, :w, :h, (:w2, :h2, :b)), h2=2, w2=2) |> size
(3, 20, 15, 128)
```

### `reduce`

The method for `Base.reduce` dispatches on `ArrowPattern`, applying reduction operations (like `sum`, `mean`, `maximum`) along specified axes. This is different from typical `Base.reduce` functionality, which reduces using binary operations.

```julia
julia> x = randn(64, 32, 100);

# perform max-reduction on the first axis
# Axis t does not appear on the right - thus we reduce over t
julia> reduce(maximum, x, (:c, :b, :t) --> (:c, :b)) |> size
(64, 32)

julia> reduce(mean, x, (:c, :b, (:t, :t5)) --> (:b, :c, :t), t5=5) |> size
(32, 64, 20)
```

### `repeat`

The method for `Base.repeat` also dispatches on `ArrowPattern`, and repeats elements along existing or new axes.

```julia
julia> image = randn(40, 30); # a grayscale image (of shape height x width)

# change it to RGB format by repeating in each channel
julia> repeat(image, (:w, :h) --> (:c, :w, :h), c=3) |> size
(3, 40, 30)

# repeat image 2 times along height (vertical axis)
julia> repeat(image, (:w, :h) --> ((:repeat, :h), :w), repeat=2) |> size
(60, 40)

# repeat image 2 time along height and 3 times along width
julia> repeat(image, (:w, :h) --> ((:w, :w3), (:h, :h2)), w3=3, h2=2) |> size
(120, 60)
```

## Roadmap

*   [x] Implement `rearrange`.
*   [x] Support Python implementation's string syntax for patterns with string macro.
*   [x] Implement `pack` and `unpack`.
*   [x] Implement `parse_shape`.
*   [x] Implement `repeat`.
*   [x] Implement `reduce`.
*   [x] Support automatic differentiation (tested with [Zygote.jl](https://github.com/FluxML/Zygote.jl)).
*   [x] Implement `einsum` (or wrap existing implementation) (see https://github.com/MurrellGroup/Einops.jl/issues/3).
*   [x] Support ellipsis notation (using `..` from [EllipsisNotation.jl](https://github.com/SciML/EllipsisNotation.jl)) (see https://github.com/MurrellGroup/Einops.jl/issues/9).
*   [ ] Explore integration with `PermutedDimsArray` or [TransmuteDims.jl](https://github.com/mcabbott/TransmuteDims.jl) for lazy and statically inferrable permutations (see https://github.com/MurrellGroup/Einops.jl/issues/4).

## Contributing

Contributions are welcome! Please feel free to open an issue to report a bug or start a discussion.
