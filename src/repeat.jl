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
@generated function repeat(x::AbstractArray{<:Any,N}, ::ArrowPattern{L,R}; context...) where {N,L,R}
    left, right = replace_ellipses(L, R, N)
    right, extra_context = remove_anonymous_dims(right)
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    repeat_names = setdiff(right_names, left_names)
    right_names_no_repeat = setdiff(right_names, repeat_names)
    shape_in = get_shape_in(N, left, pairs_type_to_names(context))
    permutation = get_permutation(left_names, right_names_no_repeat)
    positions = get_mapping(right_names, repeat_names)
    repeats = [:(getfield(context, $(QuoteNode(name)))) for name in repeat_names]
    repeat_dims = [i in positions ? repeats[findfirst(==(i), positions)] : 1 for i in 1:maximum(positions; init=0)]
    shape_out = get_shape_out(right)
    quote
        context = NamedTuple(context)
        $(isempty(extra_context) || :(context = merge(context, $extra_context)))
        $(isnothing(shape_in) || :(x = reshape(x, $shape_in)))
        $(permutation === ntuple(identity, length(permutation)) || :(x = $(Rewrap.Permute(permutation))(x)))
        $(all(==(1), repeat_dims) || :(
            x = reshape(x, $(reshape_pre_repeat(length(left_names), positions)));
            x = Repeat(($(repeat_dims...),))(x)
        ))
        $(isnothing(shape_out) || :(x = reshape(x, $shape_out)))
        return x
    end
end
