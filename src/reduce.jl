"""
    reduce(f::Function, x::AbstractArray, pattern::ArrowPattern; context...)

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

julia> y = reduce(sum, x, einops"c b t -> c b");

julia> size(y)
(64, 32)

julia> y == dropdims(sum(x, dims=3), dims=3)
true

julia> using Statistics: mean

julia> z = reduce(mean, x, einops"c b (t5 t) -> (t5 c) b", t5=5);

julia> size(z)
(320, 32)

julia> z == reshape(permutedims(dropdims(mean(reshape(x, 64,32,5,7), dims=4), dims=4), (3,1,2)), 320,32)
true
```
"""
@generated function reduce(f::Function, x, ::ArrowPattern{L,R}; context...) where {L,R}
    N = ndims(x)
    left, right = replace_ellipses(L, R, N)
    quote
        context = NamedTuple(context)
        $(reduce_body(N, left, right, pairs_type_to_names(context)))
    end
end

reduce(f::Function, x, p::BoundPattern; context...) = reduce(f, x, p.pattern; p.lengths..., context...)

# See `rearrange_body` for the shared conventions; also operates on `f`.
function reduce_body(N, left, right, context_names)
    reject_anonymous_axes(right)
    left, extra_context = remove_anonymous_dims(left)
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    isempty(setdiff(right_names, left_names)) || throw(ArgumentError("All dimension names on right side of pattern must be present on left side: $(setdiff(right_names, left_names))"))
    dims = get_mapping(left_names, setdiff(left_names, right_names))
    known = (context_names..., keys(extra_context)...)
    shape_in = get_shape_in(N, left, known)
    permutation = get_permutation(intersect(left_names, right_names), right_names)
    drop_shape = get_dropdims_shape(length(left_names), dims)
    shape_out = get_shape_out(right)
    body = Expr(:block)
    isempty(extra_context) || push!(body.args, :(context = merge(context, $extra_context)))
    append!(body.args, axis_length_checks(left, known))
    isnothing(shape_in) || push!(body.args, :(x = _reshape(x, $shape_in)))
    isempty(dims) || push!(body.args, :(x = _reshape(f(x; dims=$dims), $drop_shape)))
    permutation === ntuple(identity, length(permutation)) || push!(body.args, :(x = $(Rewrap.Permute(permutation))(x)))
    isnothing(shape_out) || push!(body.args, :(x = _reshape(x, $shape_out)))
    push!(body.args, :x)
    return body
end

# Ellipsis variant; see `rearrange_body_ellipsis`. Ellipsis dims are kept, never reduced.
function reduce_body_ellipsis(L, R, context_names)
    reject_anonymous_axes(R)
    left = replace_ellipsis_placeholder(L)
    right = replace_ellipsis_placeholder(R)
    m = ELLIPSIS_M
    left, extra_context = remove_anonymous_dims(left)
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    isempty(setdiff(right_names, left_names)) || throw(ArgumentError("All dimension names on right side of pattern must be present on left side: $(setdiff(right_names, left_names))"))
    dims = get_mapping(left_names, setdiff(left_names, right_names))
    known = (context_names..., keys(extra_context)...)
    shape_in = get_shape_in(length(left), left, known)
    kept = intersect(left_names, right_names)
    permutation = get_permutation(kept, right_names)
    drop_shape = get_dropdims_shape(length(left_names), dims)
    shape_out = get_shape_out(right)
    pli_left = findfirst(==(ELLIPSIS_PLACEHOLDER), left_names)
    pli_kept = findfirst(==(ELLIPSIS_PLACEHOLDER), kept)
    pli_entry = findfirst(==(ELLIPSIS_PLACEHOLDER), left)
    isnothing(shape_in) || expand_keep!(shape_in, pli_entry, m)
    dims_x = isempty(dims) ? dims : Expr(:tuple, (d < pli_left ? d : :($(d - 1) + $m) for d in dims)...)
    expand_keep!(drop_shape, pli_left, m)
    perm = permutation === ntuple(identity, length(permutation)) ? nothing : permute_run_expr(permutation, pli_kept)
    isnothing(shape_out) || expand_shape_out!(shape_out, right, m)
    body = Expr(:block, ellipsis_m_binding(L))
    isempty(extra_context) || push!(body.args, :(context = merge(context, $extra_context)))
    append!(body.args, axis_length_checks(left, known; axis=ellipsis_axis(pli_entry)))
    isnothing(shape_in) || push!(body.args, :(x = _reshape(x, $shape_in)))
    isempty(dims) || push!(body.args, :(x = _reshape(f(x; dims=$dims_x), $drop_shape)))
    isnothing(perm) || push!(body.args, :(x = $perm))
    isnothing(shape_out) || push!(body.args, :(x = _reshape(x, $shape_out)))
    push!(body.args, :x)
    return body
end
