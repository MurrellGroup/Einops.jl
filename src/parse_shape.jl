"""
    parse_shape(x, pattern)

Capture the shape of an array in a pattern by naming dimensions using `Symbol`s,
and `-` to ignore dimensions, and `..` to ignore any number of dimensions.

Duplicate symbols are allowed, but they must refer to dimensions of the same size.

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

julia> parse_shape(rand(2,2,4), Val((:a, :a, :b)))  # duplicate 'a' with same size
(a = 2, b = 4)
```
"""
@generated function parse_shape(x::AbstractArray{<:Any,N}, ::Val{pattern}) where {N, pattern}
    pattern′ = replace_ellipses_parse_shape(pattern, N)
    inds = findall(x -> x isa Symbol, pattern′)
    names = pattern′[inds]    
    unique_names = unique(names)
    duplicate_checks = []
    for name in unique_names
        name_indices = findall(==(name), names)
        if length(name_indices) > 1
            first_idx = inds[name_indices[1]]
            for i in 2:length(name_indices)
                idx = inds[name_indices[i]]
                push!(duplicate_checks, :(size(x, $first_idx) == size(x, $idx) || 
                    error("Dimension $($(QuoteNode(name))) appears multiple times with different sizes: $(size(x, $first_idx)) and $(size(x, $idx))")))
            end
        end
    end
    unique_inds = [inds[findfirst(==(name), names)] for name in unique_names]
    quote
        $(duplicate_checks...)
        NamedTuple{$(Tuple(unique_names))}($(Expr(:tuple, (:(size(x, $i)) for i in unique_inds)...)))
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
    inds = findtype(Symbol, pattern)
    shape_info = @ignore_derivatives NamedTuple{names,NTuple{length(inds),Int}}(size(x, i) for i in inds)
    return shape_info
end
