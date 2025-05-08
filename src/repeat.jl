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
"""
function Base.repeat(x::AbstractArray, (left, right)::Pattern; context...)
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    left_names isa Tuple{Vararg{Symbol}} || throw(ArgumentError("Invalid left pattern: $left"))
    right_names isa Tuple{Vararg{Symbol}} || throw(ArgumentError("Invalid right pattern: $right"))
    repeat_dimensions = setdiff(right_names, left_names)
    context_repeat = NamedTuple(d => context[d] for d in repeat_dimensions)
    info_dimensions = setdiff(keys(context), repeat_dimensions)
    context_info = NamedTuple(d => context[d] for d in info_dimensions)
    isempty(setdiff(right_names, left_names, keys(context))) || throw(ArgumentError("Unknown dimension sizes: $(setdiff(right_names, left_names))"))
    expanded = reshape_in(x, left; context_info...)
    right_names_no_repeat = setdiff(right_names, repeat_dimensions)
    permuted = permutedims(expanded, permutation_mapping(left_names, ntuple(i -> right_names_no_repeat[i], length(left_names))))
    reshaped = reshape(permuted, prerepeat_shape(size(expanded), left_names, right_names))
    repeated = repeat(reshaped, ntuple(i -> get(context_repeat, right_names[i], 1), length(right_names))...)
    result = reshape_out(repeated, right)
    return result
end
