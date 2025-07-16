"""
    reduce(f::Function, x::AbstractArray, left --> right; context...)

Reduce an array over the dimensions specified by the pattern,
using e.g. `sum`, `prod`, `minimum`, `maximum`, `any`, `all`, or `Statistics.mean`.

`f` must accept a `dims::Tuple{Vararg{Int}}` keyword argument, allowing
for reduction over specific dimensions. This should reduce the specified dimensions to singletons,
but not drop them.

!!! note
    This method is not meant for binary reduction operations like `+`, `*`, `min`, `max`, `|`, `&`, etc.,
    as would be expected from `Base.reduce`. Also note that Python's
    `min` and `max` equivalents are available in Julia as `minimum` and `maximum` respectively.

# Examples

```jldoctest
julia> x = randn(64, 32, 35);

julia> y = reduce(sum, x, (:c, :b, :t) --> (:c, :b));

julia> size(y)
(64, 32)

julia> y == dropdims(sum(x, dims=3), dims=3)
true

julia> using Statistics: mean

julia> z = reduce(mean, x, (:c, :b, (:t5, :t)) --> ((:t5, :c), :b), t5=5);

julia> size(z)
(320, 32)

julia> z == reshape(permutedims(dropdims(mean(reshape(x, 64,32,5,7), dims=4), dims=4), (3,1,2)), 320,32)
true
```
"""
@generated function Base.reduce(f::Function, x::AbstractArray{<:Any,N}, ::ArrowPattern{L,R}; context...) where {N,L,R}
    left, right = replace_ellipses(L, R, N)
    left, extra_context = remove_anonymous_dims(left)
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    isempty(setdiff(right_names, left_names)) || throw(ArgumentError("All dimension names on right side of pattern must be present on left side: $(setdiff(right_names, left_names))"))
    dims = get_mapping(left_names, setdiff(left_names, right_names))
    shape_in = get_shape_in(N, left, (pairs_type_to_names(context)..., keys(extra_context)...))
    permutation = get_permutation(intersect(left_names, right_names), right_names)
    shape_out = get_shape_out(right)
    quote
        $(isempty(extra_context) || :(context = pairs(merge(NamedTuple(context), $extra_context))))
        $(isnothing(shape_in) || :(x = reshape(x, $shape_in)))
        $(isempty(dims) || :(x = dropdims(f(x; dims=$dims); dims=$dims)))
        $(permutation === ntuple(identity, length(permutation)) || :(x = permutedims(x, $permutation)))
        $(isnothing(shape_out) || :(x = reshape(x, $shape_out)))
        return x
    end
end

Base.reduce(f::Function, x::AbstractArray{<:AbstractArray}, pattern::ArrowPattern; context...) = reduce(f, stack(x), pattern; context...)
Base.reduce(f::Function, x, pattern::ArrowPattern; context...) = reduce(f, stack(x), pattern; context...)
