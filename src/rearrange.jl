"""
    rearrange(array::AbstractArray, left --> right; context...)
    rearrange(arrays, left --> right; context...)

Rearrange the axes of `x` according to the pattern specified by `left --> right`.

Can always be expressed as a `reshape` + `permutedims` + `reshape`.

# Examples

```jldoctest
julia> x = rand(2,3,5);

julia> y = rearrange(x, (:a, :b, :c) --> (:c, :b, :a));

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
function rearrange(x::AbstractArray, (left, right)::ArrowPattern; context...)
    (!isempty(extract(typeof(..), left)) || !isempty(extract(typeof(..), right))) && throw(ArgumentError("Ellipses (..) are currently not supported"))
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    reshaped_in = reshape_in(x, left; context...)
    permuted = permute(reshaped_in, left_names, right_names)
    reshaped_out = reshape_out(permuted, right)
    return reshaped_out
end

rearrange(x::AbstractArray{<:AbstractArray}, pattern::ArrowPattern; context...) = rearrange(stack(x), pattern; context...)
rearrange(x, pattern::ArrowPattern; context...) = rearrange(stack(x), pattern; context...)
