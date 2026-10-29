"""
    parse_shape(x, pattern)

Capture the shape of an array in a pattern by naming dimensions, with `_` to ignore
a dimension and `...` to ignore any number of them. Without an ellipsis, a tuple of
`Symbol`s and `-` works too.

Duplicate names are allowed, but they must refer to dimensions of the same size.

# Examples

```jldoctest
julia> parse_shape(rand(2,3,4), (:a, :b, -))
(a = 2, b = 3)

julia> parse_shape(rand(2,3), (-, -))
NamedTuple()

julia> parse_shape(rand(2,3,4,5), einops"first second third fourth")
(first = 2, second = 3, third = 4, fourth = 5)

julia> parse_shape(rand(2,3,4), einops"a b ...")
(a = 2, b = 3)

julia> parse_shape(rand(2,2,4), einops"a a b")  # duplicate 'a' with same size
(a = 2, b = 4)
```
"""
@generated function parse_shape(x, ::Val{pattern}) where {pattern}
    N = ndims(x)
    count(==(Ellipsis()), pattern) <= 1 || throw(ArgumentError("`...` appears more than once in the pattern"))
    check_rank(pattern, N)
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
                    throw(DimensionMismatch(string("`", $(QuoteNode(name)), "` has length ", size(x, $first_idx), " at axis ", $first_idx,
                                                   " but ", size(x, $idx), " at axis ", $idx)))))
            end
        end
    end
    unique_inds = [inds[findfirst(==(name), names)] for name in unique_names]
    quote
        $(duplicate_checks...)
        NamedTuple{$(Tuple(unique_names))}($(Expr(:tuple, (:(size(x, $i)) for i in unique_inds)...)))
    end
end

# output type is statically knowable when pattern doesn't contain ellipses
# (needs to be constant-propagated)
function parse_shape(x, pattern::Tuple{Vararg{Union{Symbol,typeof(-)}}})
    check_rank(pattern, ndims(x))
    names = extract(Symbol, pattern)
    inds = findtype(Symbol, pattern)
    shape_info = @ignore_derivatives NamedTuple{names,NTuple{length(inds),Int}}(size(x, i) for i in inds)
    return shape_info
end
