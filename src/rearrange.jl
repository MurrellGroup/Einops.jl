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
@generated function rearrange(x::AbstractArray{<:Any,N}, ::ArrowPattern{L,R}; context...) where {N,L,R}
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

rearrange(x::AbstractArray{<:AbstractArray}, pattern::ArrowPattern; context...) = rearrange(stack(x), pattern; context...)
rearrange(x, pattern::ArrowPattern; context...) = rearrange(stack(x), pattern; context...)

@generated function expand(x::AbstractArray{<:Any,N}, ::Val{L}; context...) where {N,L}
    left = replace_ellipses_left(L, N)
    shape_in = get_shape_in(N, left, pairs_type_to_names(context))
    quote
        $(isnothing(shape_in) || :(x = reshape(x, $shape_in)))
        return x
    end
end

@generated function collapse(x::AbstractArray{<:Any,N}, ::Val{R}; context...) where {N,R}
    # Replace ellipsis on the right side using available rank information N.
    # Number of consumed dimensions on the right equals the number of symbols.
    right = begin
        # Count symbol dimensions (they consume axes); integers don't consume axes
        symbol_count = length(extract(Symbol, R))
        if (..) ∈ flatten(R)
            # Allow only a single ellipsis
            count(==(..), flatten(R)) <= 1 || throw("Only one ellipsis is allowed: $R")
            # Ellipsis must stand for the remaining axes
            m = N - symbol_count
            m >= 0 || throw("Ellipsis represents a negative number of dimensions: $R with N=$N")
            replacement = anonymous_symbols(:__ellipsis, m)
            if (..) in R
                insertat(R, only(findfirst(==(..), R)), replacement)
            else
                # ellipsis is in a nested tuple
                i = findfirst(t -> t isa Tuple && (..) in t, R)
                t = R[i]
                j = findfirst(==(..), t)
                insertat(R, i, (insertat(t, j, replacement),))
            end
        else
            R
        end
    end
    shape_out = get_shape_out(right)
    quote
        $(isnothing(shape_out) || :(x = reshape(x, $shape_out)))
        return x
    end
end