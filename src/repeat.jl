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
    repeat(x::AbstractArray, left --> right; context...)

Repeat elements of `x` along specified axes.

# Examples

```jldoctest
julia> x = rand(2,3);

julia> y = repeat(x, einops"a b -> a b 1 r", r=2);

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

# See `rearrange_body` for the shared conventions.
function repeat_body(N, left, right, context_names)
    right, extra_context = remove_anonymous_dims(right)
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    repeat_names = setdiff(right_names, left_names)
    right_names_no_repeat = setdiff(right_names, repeat_names)
    shape_in = get_shape_in(N, left, context_names)
    permutation = get_permutation(left_names, right_names_no_repeat)
    positions = get_mapping(right_names, repeat_names)
    repeats = [:(getfield(context, $(QuoteNode(name)))) for name in repeat_names]
    repeat_dims = [i in positions ? repeats[findfirst(==(i), positions)] : 1 for i in 1:maximum(positions; init=0)]
    shape_out = get_shape_out(right)
    quote
        $(isempty(extra_context) || :(context = merge(context, $extra_context)))
        $(isnothing(shape_in) || :(x = Rewrap.reshape(x, $shape_in)))
        $(permutation === ntuple(identity, length(permutation)) || :(x = $(Rewrap.Permute(permutation))(x)))
        $(all(==(1), repeat_dims) || :(
            x = Rewrap.reshape(x, $(reshape_pre_repeat(length(left_names), positions)));
            x = Repeat(($(repeat_dims...),))(x)
        ))
        $(isnothing(shape_out) || :(x = Rewrap.reshape(x, $shape_out)))
        x
    end
end

# Ellipsis variant; see `rearrange_body_ellipsis`. Ellipsis dims are kept (never
# repeated), so their `Repeat` factor slot expands to a run of 1's.
function repeat_body_ellipsis(L, R, context_names)
    left = replace_ellipsis_placeholder(L)
    right = replace_ellipsis_placeholder(R)
    m = ELLIPSIS_M
    right, extra_context = remove_anonymous_dims(right)
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    repeat_names = setdiff(right_names, left_names)
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

    isnothing(shape_in) || expand_keep!(shape_in, findfirst(==(ELLIPSIS_PLACEHOLDER), left), m)
    perm = permutation === ntuple(identity, length(permutation)) ? nothing : permute_run_expr(permutation, pli_left, m)
    do_repeat = !all(==(1), repeat_dims)
    pre_repeat = repeat_tuple = nothing
    if do_repeat
        pre_repeat = reshape_pre_repeat(length(left_names), positions)
        expand_keep!(pre_repeat, nth_keep_index(pre_repeat, pli_norepeat), m)
        pli_right <= length(repeat_dims) && (repeat_dims[pli_right] = Expr(:..., :(ntuple(Returns(1), Val($m)))))
        repeat_tuple = Expr(:tuple, repeat_dims...)
    end
    isnothing(shape_out) || expand_shape_out!(shape_out, right, m)

    quote
        $(ellipsis_m_binding(L))
        $(isempty(extra_context) || :(context = merge(context, $extra_context)))
        $(isnothing(shape_in) || :(x = Rewrap.reshape(x, $shape_in)))
        $(isnothing(perm) || :(x = $perm(x)))
        $(!do_repeat || :(
            x = Rewrap.reshape(x, $pre_repeat);
            x = Repeat($repeat_tuple)(x)
        ))
        $(isnothing(shape_out) || :(x = Rewrap.reshape(x, $shape_out)))
        x
    end
end
