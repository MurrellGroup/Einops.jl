<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./docs/src/assets/logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="./docs/src/assets/logo.png">
    <img src="./docs/src/assets/logo.png" width="256" alt="Logo" />
  </picture>
</p>

<h1 align="center">Einops.jl</h1>

<div align="center">

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/Einops.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/Einops.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/Einops.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Einops.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Einops.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Einops.jl)

</div>

> Einops.jl brings the readable and concise tensor operations of [einops](https://einops.rocks) to Julia, reliably expanding to existing primitives like `reshape`, `permutedims`, and `repeat`, with no performance overhead.

## Einops vs Julia primitives

Einops patterns can be constructed with the `einops` string macro, e.g. `einops"a b -> (b a)"` expands to the form `(:a, :b) --> ((:b, :a),)`, where `-->` is a custom operator that puts the left and right operands as type parameters of a special pattern type.

The snippets below show identical transformations expressed first with Einops (one readable line) and then with "hand-rolled" Julia primitives. Notice how Einops collapses multiple e.g. `reshape` / `permutedims` / `dropdims` / `repeat` calls into a single, declarative statement. Note that Einops simply expands to these primitives under the hood, and avoids no-ops, so there is little to no performance overhead.

<table style="table-layout: fixed; width: 100%;">


  <tr>
    <th>Einops</th>
    <th>Julia Base primitives</th>
  </tr>


  <tr>
  <td>

  ```julia
  rearrange(x, einops"a b c -> (a b) c")
  ```
  </td>
  <td>

  ```julia
  reshape(x, :, size(x, 3))
  ```
  </td>
  </tr>


  <tr>
  <td>

  ```julia
  rearrange(x, einops"a b c -> (b a) c")
  ```
  </td>
  <td>

  ```julia
  reshape(permutedims(x, (2, 1, 3)), :, size(a, 3))
  ```
  </td>
  </tr>


  <tr>
  <td>

  ```julia
  rearrange(q, einops"(d h) l b -> d l (h b)"; d=head_dim)
  ```
  </td>
  <td>

  ```julia
  reshape(permutedims(reshape(q, head_dim, size(q, 1) ÷ head_dim, size(q)[2:3]), (2, 1, 3, 4)), head_dim, size(q, 2), :)
  ```
  </td>
  </tr>


  <tr>
  <td>

  ```julia
  repeat(k, einops"(d h) l b -> d l (r h b)"; d=head_dim, r=repeats)
  ```
  </td>
  <td>

  ```julia
  reshape(repeat(permutedims(reshape(k, head_dim, size(k, 1) ÷ head_dim, size(k)[2:3]), (2, 1, 3, 4)), inner=(1, 1, repeats)), head_dim, size(k, 2), :)
  ```
  </td>
  </tr>


</table>

## Contributing

Contributions are welcome! Please feel free to open an issue to report a bug or start a discussion.
