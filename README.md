

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

Einops.jl is a Julia implementation of the [einops](https://einops.rocks) Python package, providing an elegant and intuitive notation for tensor operations, and unifying Julia's `reshape`, `permutedims`, `reduce` and `repeat` functions, with support for automatic differentiation.

The Python implementation uses strings to specify the operation, but that would be tricky to compile in Julia, so a string macro `@einops_str` is exported for parity, e.g. `einops"a 1 b c -> (c b) a"`, which expands to the form `(:a, 1, :b, :c,) --> ((:c, :b), :a)`, allowing for compile-time awareness of dimensionalities, ensuring type stability.

## Operations

### `rearrange`

The `rearrange` combines reshaping and permutation operations into a single, expressive command.

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

### `reduce`

The method for `Base.reduce` dispatches on `Einops.Pattern`, applying reduction operations (like `sum`, `mean`, `maximum`) along specified axes. This is different from typical `Base.reduce` functionality, which reduces using binary operations.

```julia
julia> x = randn(100, 32, 64);

# perform max-reduction on the first axis
# Axis t does not appear on the right - thus we reduce over t
julia> reduce(maximum, x, (:t, :b, :c) --> (:b, :c)) |> size
(32, 64)

julia> reduce(mean, x, ((:t5, :t), :b, :c) --> (:b, (:t, :c)), t5=5) |> size
(32, 1280)
```

### `repeat`

The method for `Base.repeat` also dispatches on `Einops.Pattern`, and repeats elements along existing or new axes.

```julia
julia> image = randn(30, 40); # a grayscale image (of shape height x width)

# change it to RGB format by repeating in each channel
julia> repeat(image, (:h, :w) --> (:h, :w, :c), c=3) |> size
(30, 40, 3)

# repeat image 2 times along height (vertical axis)
julia> repeat(image, (:h, :w) --> ((:repeat, :h), :w), repeat=2) |> size
(60, 40)

# repeat image 2 time along height and 3 times along width
julia> repeat(image, (:h, :w) --> ((:h2, :h), (:w3, :w)), h2=2, w3=3) |> size
(60, 120)
```

## Roadmap

*   [x] Implement `rearrange`.
*   [x] Support Python implementation's string syntax for patterns with string macro.
*   [x] Implement `pack` and `unpack`.
*   [x] Implement `parse_shape`.
*   [x] Implement `repeat`.
*   [x] Implement `reduce`.
*   [x] Support automatic differentiation (tested with [Zygote.jl](https://github.com/FluxML/Zygote.jl)).
*   [ ] Support ellipsis notation (using `..` from [EllipsisNotation.jl](https://github.com/SciML/EllipsisNotation.jl)).
*   [ ] Explore integration with `PermutedDimsArray` or [TransmuteDims.jl](https://github.com/mcabbott/TransmuteDims.jl) for lazy and statically inferrable permutations.
*   [ ] Implement `einsum` (or wrap existing implementation).

## Contributing

Contributions are welcome! Please feel free to open an issue to report a bug or start a discussion.
