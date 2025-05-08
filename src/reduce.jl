# TODO: support integers > 1 in `left`

"""
    reduce(f, x::AbstractArray, left --> right; context...)

Reduce an array over the dimensions specified by the pattern,
using the function `f`, e.g. `sum`, `prod`, `minimum`, `maximum`, `any`, `all`, or `Statistics.mean`.
"""
function Base.reduce(f, x::AbstractArray, (left, right)::Pattern; context...)
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    reduced_dim_names = setdiff(left_names, right_names)
    info_dim_names = setdiff(keys(context), reduced_dim_names)
    reshaped = reshape_in(x, left; context...) # TODO: use info dims
    reduced_dims = ntuple(i -> findfirst(isequal(reduced_dim_names[i]), left_names), length(reduced_dim_names))
    reduced = dropdims(f(reshaped, dims=reduced_dims); dims=reduced_dims)
    reduced_left_names = intersect(left_names, right_names)
    permuted = _permutedims(reduced, permutation_mapping(ntuple(i -> reduced_left_names[i], length(right_names)), right_names))
    result = reshape_out(permuted, right)
    return result
end
