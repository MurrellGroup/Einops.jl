@generated function _rearrange(permutedims, x::AbstractArray{<:Any,N}, ::ArrowPattern{L,R}; context...) where {N,L,R}
    left, right = replace_ellipses(L, R, N)
    shape_in = get_shape_in(N, left, pairs_type_to_names(context))
    permutation = get_permutation(extract(Symbol, left), extract(Symbol, right))
    shape_out = get_shape_out(right)
    quote
        $(isnothing(shape_in) || :(x = reshape(x, $shape_in)))
        $(permutation === ntuple(identity, length(permutation)) || :(x = permutedims(x, $permutation)))
        $(isnothing(shape_out) || :(x = reshape(x, $shape_out)))
        return x
    end
end

_rearrange(x::AbstractArray{<:AbstractArray}, pattern::ArrowPattern; context...) = _rearrange(stack(x), pattern; context...)
_rearrange(x, pattern::ArrowPattern; context...) = _rearrange(stack(x), pattern; context...)

"""
    rearrange(array::AbstractArray, left --> right; context...)
    rearrange(arrays, left --> right; context...)

Rearrange the axes of `x` according to the pattern specified by `left --> right`.

Can always be expressed as a `reshape` + `permutedims` + `reshape`.

# Examples

```jldoctest
julia> x = rand(2,3,5);

julia> y = rearrange(x, einops"a b c -> c b a");

julia> size(y)
(5, 3, 2)

julia> y == permutedims(x, (3,2,1))
true

julia> z = rearrange(x, (:a, :b, :c) --> (:a, (:c, :b)));

julia> size(z)
(2, 15)

julia> z == reshape(permutedims(x, (1,3,2)), 2,5*3)
true
```
"""
rearrange(x, pattern::ArrowPattern; context...) = _rearrange(permutedims, x, pattern; context...)

function _permutedims!(x_src, p)
    x_dest = reshape(x_src, ntuple(i -> size(x_src, p[i]), length(p)))
    permutedims!(x_dest, x_src, p)
    return x_dest
end

"""
    rearrange!(array::AbstractArray, left --> right; context...)

In-place version of [`rearrange`](@ref).
"""
rearrange!(x, pattern::ArrowPattern; context...) = _rearrange(_permutedims!, x, pattern; context...)
