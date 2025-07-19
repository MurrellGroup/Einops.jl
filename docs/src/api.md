# API

## Patterns

```@docs
ArrowPattern
-->
@einops_str
```

### Ellipsis notation

For patterns constructed with `-->`, one can use `..` (from EllipsisNotation.jl) to represent multiple dimensions.

```jldoctest
julia> rearrange(rand(2,3,4), (:a, ..) --> (.., :a)) |> size
(3, 4, 2)
```

## `parse_shape`

```@docs
parse_shape
```

## `rearrange`

```@docs
rearrange
```

## `reduce`

```@docs
reduce
```

## `repeat`

```@docs
repeat
```

## `einsum`

```@docs
einsum
```

## `pack` and `unpack`

```@docs
pack
unpack
```
