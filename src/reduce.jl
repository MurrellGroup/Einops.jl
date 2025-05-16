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
    `min` and `max` are available in Julia as `minimum` and `maximum` respectively.

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
function Base.reduce(f::Function, x::AbstractArray, (left, right)::ArrowPattern; context...)
    allunique(extract(Symbol, right)) || throw(ArgumentError("Right names $(right) are not unique"))
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    expanded = reshape_in(x, left; context...)
    reduced_dims, permutation = @ignore_derivatives begin
        isempty(setdiff(right_names, left_names)) || throw(ArgumentError("All dimension names on right side of pattern must be present on left side: $(setdiff(right_names, left_names))"))
        reduced_dim_names = setdiff(left_names, right_names)
        reduced_dims = ntuple(i -> findfirst(isequal(reduced_dim_names[i]), left_names)::Int, length(left_names) - length(right_names))
        reduced_left_names = intersect(left_names, right_names)
        permutation = permutation_mapping(ntuple(i -> reduced_left_names[i], length(right_names)), right_names)
        reduced_dims, permutation
    end
    reduced = f(expanded, dims=reduced_dims)
    dropped = dropdims(reduced, dims=reduced_dims)
    permuted = _permutedims(dropped, permutation)
    collapsed = reshape_out(permuted, right)
    return collapsed
end

Base.reduce(f::Function, x::AbstractArray{<:AbstractArray}, pattern::ArrowPattern; context...) = reduce(f, stack(x), pattern; context...)
Base.reduce(f::Function, x, pattern::ArrowPattern; context...) = reduce(f, stack(x), pattern; context...)
