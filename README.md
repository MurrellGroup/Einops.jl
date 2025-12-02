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
[![Dev](https://img.shields.io/badge/einops-rocks-434EB2.svg)](https://einops.rocks)
[![Build Status](https://github.com/MurrellGroup/Einops.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/Einops.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/Einops.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/Einops.jl)

</div>

> Einops.jl brings the readable and concise tensor operations of [einops](https://einops.rocks) to Julia, reliably expanding to existing primitives like `reshape`, `permutedims`, and `repeat`.

## Einops vs Base primitives

Einops uses patterns with explicitly named dimensions, which can be constructed with the `einops` string macro, e.g. `einops"a b -> (b a)"` expands to the form `(:a, :b) --> ((:b, :a),)`, where `-->` is a custom operator that puts the left and right operands as type parameters of a special pattern type, allowing generated functions to compose clean expressions.

The snippets below show identical transformations expressed first with Einops (one readable line) and then with "hand-rolled" Julia primitives. Notice how Einops collapses multiple e.g. `reshape` / `permutedims` / `dropdims` / `repeat` calls into a single, declarative statement, while still expanding to such primitives under the hood and avoiding no-ops.

<table>
<thead>
<tr>
    <th>Description</th>
    <th>Einops</th>
    <th>Base primitives</th>
</tr>
</thead>

<tbody>
<tr><td>

Flatten first two dimensions

</td>
<td>

```julia
rearrange(x, einops"a b c -> (a b) c")
```
</td>
<td>

```julia
reshape(x, :, size(x, 3))
```
</td></tr>

<tr><td>

Permute first two dimensions

</td>
<td style="text-align:center">

```julia
rearrange(x, einops"a b c -> b a c")
```
</td>
<td style="text-align:center">

```julia
permutedims(x, (2, 1, 3))
```
</td></tr>

<tr><td>

Permute and flatten

</td>
<td style="text-align:center">

```julia
rearrange(x, einops"a b -> (b a)")
```
</td>
<td style="text-align:center">

```julia
vec(permutedims(x))
```
</td></tr>

<tr><td>

Remove first dimension singleton

</td>
<td style="text-align:center">

```julia
rearrange(x, einops"1 ... -> ...")
```
</td>
<td style="text-align:center">

```julia
dropdims(x, dims=1)
```
</td></tr>

<tr><td>

Funky repeat

</td>
<td style="text-align:center">

```julia
repeat(x, einops"... -> 2 ... 3")



```
</td>
<td style="text-align:center">

```julia
repeat(
  reshape(x, 1, size(x)...),
  2, ntuple(Returns(1), ndims(x))..., 3)
```
</td></tr>

<tr><td>

Multi-Head Attention

</td>
<td style="text-align:center">

```julia
rearrange(q,
  einops"(d h) l b -> d l (h b)";
  d=head_dim)



```
</td>
<td style="text-align:center">

```julia
reshape(
  permutedims(
    reshape(q, head_dim, :, size(q)[2:3]...),
    (1, 3, 2, 4)),
  head_dim, size(q, 2), :)
```
</td></tr>

<tr><td>

Grouped-Query Attention

</td>
<td style="text-align:center">

```julia
repeat(k,
  einops"(d h) l b -> d l (r h b)";
  d=head_dim, r=repeats)





```
</td>
<td style="text-align:center">

```julia
reshape(
  repeat(
    permutedims(
      reshape(k, head_dim, :, size(k)[2:3]...),
      (1, 3, 2, 4)),
    inner=(1, 1, repeats, 1)),
  head_dim, size(k, 2), :)
```
</td></tr>

</tbody>
</table>

## Contributing

Contributions are welcome! Please feel free to open an issue to report a bug or start a discussion.
