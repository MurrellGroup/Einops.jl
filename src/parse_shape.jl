"""
    parse_shape(x, pattern)

Capture the shape of an array in a pattern by naming dimensions using `Symbol`s,
and `-` to ignore dimensions, and `...` to ignore any number of dimensions.

!!! note
    For proper type inference, the pattern needs to be passed as `Val(pattern)`
    when an ellipsis is present. This is done automatically when using [`@einops_str`](@ref).

# Examples

```jldoctest
julia> parse_shape(rand(2,3,4), (:a, :b, -))
(a = 2, b = 3)

julia> parse_shape(rand(2,3), (-, -))
NamedTuple()

julia> parse_shape(rand(2,3,4,5), einops"first second third fourth")
(first = 2, second = 3, third = 4, fourth = 5)

julia> parse_shape(rand(2,3,4), Val((:a, :b, ..)))
(a = 2, b = 3)
```
"""
@generated function parse_shape(x::AbstractArray{<:Any,N}, ::Val{pattern}) where {N, pattern}
    pattern′ = replace_ellipses_parse_shape(pattern, N)
    inds = findall(x -> x isa Symbol, pattern′)
    names = pattern′[inds]
    allunique(names) || error("Pattern $(pattern) has duplicate elements")
    quote
        NamedTuple{$names}($(Expr(:tuple, (:(size(x, $i)) for i in inds)...)))
    end
end

function parse_shape(x::AbstractArray, ellipsis_pattern)
    Base.depwarn("""
    `parse_shape(x, ellipsis_pattern)` is not type stable, use `parse_shape(x, Val(ellipsis_pattern))`
    or construct the pattern using `@einops_str` instead.
    """, :parse_shape)
    parse_shape(x, Val(ellipsis_pattern))
end

# output type is statically knowable when pattern doesn't contain ellipses
# (needs to be constant-propagated)
function parse_shape(x::AbstractArray{<:Any,N}, pattern::Tuple{Vararg{Union{Symbol,typeof(-)}}}) where N
    names = extract(Symbol, pattern)
    allunique(names) || error("Pattern $(pattern) has duplicate elements")
    inds = findtype(Symbol, pattern)
    shape_info = @ignore_derivatives NamedTuple{names,NTuple{length(inds),Int}}(size(x, i) for i in inds)
    return shape_info
end
