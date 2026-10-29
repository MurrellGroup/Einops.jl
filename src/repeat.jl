function reshape_pre_repeat(N, positions)
    new_shape = :()
    ops = new_shape.args
    for _ in 1:N
        push!(ops, :($Keep()))
    end
    for i in sort(collect(positions))
        if i > length(ops)
            for _ in 1:(i - length(ops))
                push!(ops, :($Unsqueeze()))
            end
        else
            insert!(ops, i, :($Unsqueeze()))
        end
    end
    return new_shape
end

"""
    repeat(x::AbstractArray, pattern::ArrowPattern; context...)

Repeat elements of `x` along specified axes.

# Examples

```jldoctest
julia> x = rand(2,3);

julia> y = repeat(x, einops"a b -> a b 1 r", r=2);

julia> size(y)
(2, 3, 1, 2)

julia> y == reshape(repeat(x, 1,1,2), 2,3,1,2)
true

julia> z = repeat(x, einops"a b -> a (b r)", r=2);

julia> size(z)
(2, 6)

julia> z == reshape(repeat(x, 1,1,2), 2,6)
true
```
"""
@generated function repeat(x, ::ArrowPattern{L,R}; context...) where {L,R}
    N = ndims(x)
    left, right = replace_ellipses(L, R, N)
    quote
        context = NamedTuple(context)
        $(repeat_body(N, left, right, pairs_type_to_names(context)))
    end
end

# Resolve ambiguity with `Base.repeat(::AbstractArray, counts...)`: the untyped method is
# more specific in the pattern, Base's in the array, so `AbstractArray` needs its own.
repeat(x::AbstractArray, pattern::ArrowPattern; context...) =
    invoke(repeat, Tuple{Any,ArrowPattern}, x, pattern; context...)

repeat(x, p::BoundPattern; context...) = repeat(x, p.pattern; p.lengths..., context...)
repeat(x::AbstractArray, p::BoundPattern; context...) = repeat(x, p.pattern; p.lengths..., context...)

# Each axis `repeat` creates needs its length.
function check_lengths_given(new_names, known)
    missing = filter(∉(known), new_names)
    isempty(missing) || throw(ArgumentError(length(missing) == 1 ?
        "`repeat` needs the length of new axis $(quoted(missing)); pass it as a keyword" :
        "`repeat` needs the lengths of new axes $(quoted(missing)); pass them as keywords"))
    return nothing
end

# See `rearrange_body` for the shared conventions.
function repeat_body(N, left, right, context_names)
    reject_anonymous_axes(left)
    right, extra_context = remove_anonymous_dims(right)
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    repeat_names = setdiff(right_names, left_names)
    check_lengths_given(repeat_names, (context_names..., keys(extra_context)...))
    right_names_no_repeat = setdiff(right_names, repeat_names)
    shape_in = get_shape_in(N, left, context_names)
    permutation = get_permutation(left_names, right_names_no_repeat)
    positions = get_mapping(right_names, repeat_names)
    repeats = [:(getfield(context, $(QuoteNode(name)))) for name in repeat_names]
    repeat_dims = [i in positions ? repeats[findfirst(==(i), positions)] : 1 for i in 1:maximum(positions; init=0)]
    shape_out = get_shape_out(right)
    body = Expr(:block)
    isempty(extra_context) || push!(body.args, :(context = merge(context, $extra_context)))
    append!(body.args, axis_length_checks(left, context_names))
    isnothing(shape_in) || push!(body.args, :(x = _reshape(x, $shape_in)))
    permutation === ntuple(identity, length(permutation)) || push!(body.args, :(x = $(Rewrap.Permute(permutation))(x)))
    if !all(==(1), repeat_dims)
        push!(body.args, :(x = _reshape(x, $(reshape_pre_repeat(length(left_names), positions)))))
        push!(body.args, :(x = Repeat(($(repeat_dims...),))(x)))
    end
    isnothing(shape_out) || push!(body.args, :(x = _reshape(x, $shape_out)))
    push!(body.args, :x)
    return body
end

# Ellipsis variant; see `rearrange_body_ellipsis`. Ellipsis dims are kept (never
# repeated), so their `Repeat` factor slot expands to a run of 1's.
function repeat_body_ellipsis(L, R, context_names)
    reject_anonymous_axes(L)
    left = replace_ellipsis_placeholder(L)
    right = replace_ellipsis_placeholder(R)
    m = ELLIPSIS_M
    right, extra_context = remove_anonymous_dims(right)
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    repeat_names = setdiff(right_names, left_names)
    check_lengths_given(repeat_names, (context_names..., keys(extra_context)...))
    right_names_no_repeat = setdiff(right_names, repeat_names)
    shape_in = get_shape_in(length(left), left, context_names)
    permutation = get_permutation(left_names, right_names_no_repeat)
    positions = get_mapping(right_names, repeat_names)
    repeats = [:(getfield(context, $(QuoteNode(name)))) for name in repeat_names]
    repeat_dims = Any[i in positions ? repeats[findfirst(==(i), positions)] : 1 for i in 1:maximum(positions; init=0)]
    shape_out = get_shape_out(right)

    pli_left = findfirst(==(ELLIPSIS_PLACEHOLDER), left_names)
    pli_norepeat = findfirst(==(ELLIPSIS_PLACEHOLDER), right_names_no_repeat)
    pli_right = findfirst(==(ELLIPSIS_PLACEHOLDER), right_names)

    pli_entry = findfirst(==(ELLIPSIS_PLACEHOLDER), left)
    isnothing(shape_in) || expand_keep!(shape_in, pli_entry, m)
    perm = permutation === ntuple(identity, length(permutation)) ? nothing : permute_run_expr(permutation, pli_left)
    do_repeat = !all(==(1), repeat_dims)
    pre_repeat = repeat_tuple = nothing
    if do_repeat
        pre_repeat = reshape_pre_repeat(length(left_names), positions)
        expand_keep!(pre_repeat, nth_keep_index(pre_repeat, pli_norepeat), m)
        pli_right <= length(repeat_dims) && (repeat_dims[pli_right] = Expr(:..., :(ntuple(Returns(1), Val($m)))))
        repeat_tuple = Expr(:tuple, repeat_dims...)
    end
    isnothing(shape_out) || expand_shape_out!(shape_out, right, m)

    body = Expr(:block, ellipsis_m_binding(L))
    isempty(extra_context) || push!(body.args, :(context = merge(context, $extra_context)))
    append!(body.args, axis_length_checks(left, context_names; axis=ellipsis_axis(pli_entry)))
    isnothing(shape_in) || push!(body.args, :(x = _reshape(x, $shape_in)))
    isnothing(perm) || push!(body.args, :(x = $perm))
    if do_repeat
        push!(body.args, :(x = _reshape(x, $pre_repeat)))
        push!(body.args, :(x = Repeat($repeat_tuple)(x)))
    end
    isnothing(shape_out) || push!(body.args, :(x = _reshape(x, $shape_out)))
    push!(body.args, :x)
    return body
end
