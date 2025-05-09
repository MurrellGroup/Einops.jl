_get(x, i::Int) = x[i]
_get(x, ::Nothing) = 1

function prerepeat_shape(input_shape::Dims, left::Tuple{Vararg{Symbol}}, right::NTuple{N,Symbol}) where N
    output_shape = map(key -> _get(input_shape, findfirst(isequal(key), left)), right)
    return ntuple(i -> output_shape[i], length(right))
end

# TODO: split repeat method into functions, and dispatch on left/right patterns to minimize operations
# function _repeat end

# TODO: support integers > 1 in `right`

"""
    repeat(x::AbstractArray, left --> right; context...)

Repeat elements of `x` along specified axes.

# Examples

```jldoctest
julia> x = rand(2,3);

julia> y = repeat(x, (:a, :b) --> (:a, :b, 1, :r), r=2);

julia> size(y)
(2, 3, 1, 2)

julia> y == reshape(repeat(x, 1,1,2), 2,3,1,2)
true

julia> z = repeat(x, (:a, :b) --> (:a, (:b, :r)), r=2);

julia> size(z)
(2, 6)

julia> z == reshape(repeat(x, 1,1,2), 2,6)
true
```
"""
function Base.repeat(x::AbstractArray, (left, right)::Pattern; context...)
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    local context_info, permutation, repeats
    @ignore_derivatives begin
        repeat_dim_names = setdiff(right_names, left_names)
        context_repeat = NamedTuple(d => context[d] for d in repeat_dim_names)
        info_dim_names = setdiff(keys(context), repeat_dim_names)
        context_info = NamedTuple(d => context[d] for d in info_dim_names)
        isempty(setdiff(right_names, left_names, keys(context))) || throw(ArgumentError("Unknown dimension sizes: $(setdiff(right_names, left_names))"))
        right_names_no_repeat = setdiff(right_names, repeat_dim_names)
        permutation = permutation_mapping(left_names, ntuple(i -> right_names_no_repeat[i], length(left_names)))
        repeats = ntuple(i -> get(context_repeat, right_names[i], 1), length(right_names))
    end
    expanded = reshape_in(x, left; context_info...)
    permuted = _permutedims(expanded, permutation)
    reshaped = reshape(permuted, prerepeat_shape(size(expanded), left_names, right_names))
    repeated = repeat(reshaped, repeats...)
    result = reshape_out(repeated, right)
    return result
end
