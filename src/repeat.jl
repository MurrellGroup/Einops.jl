function reshape_pre_repeat(N, positions)
    new_shape = Expr(:tuple, [:(size(x, $i)) for i in 1:N]...)
    sizes = new_shape.args
    for i in sort(collect(positions))
        if i > length(sizes)
            append!(sizes, ones(Int, i - length(sizes)))
        else
            insert!(sizes, i, 1)
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
@generated function Base.repeat(x::AbstractArray{<:Any,N}, ::ArrowPattern{L,R}; context...) where {N,L,R}
    left, right = replace_ellipses(L, R, N)
    right, extra_context = remove_anonymous_dims(right)
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    repeat_names = setdiff(right_names, left_names)
    right_names_no_repeat = setdiff(right_names, repeat_names)
    shape_in = get_shape_in(N, left, pairs_type_to_names(context))
    permutation = get_permutation(left_names, right_names_no_repeat)
    positions = get_mapping(right_names, repeat_names)
    repeats = [:(context[$(QuoteNode(name))]) for name in repeat_names]
    repeat_dims = [i in positions ? repeats[findfirst(==(i), positions)] : 1 for i in 1:maximum(positions; init=0)]
    shape_out = get_shape_out(right)
    quote
        $(isempty(extra_context) || :(context = pairs(merge(NamedTuple(context), $extra_context))))
        $(isnothing(shape_in) || :(x = reshape(x, $shape_in)))
        $(permutation === ntuple(identity, length(permutation)) || :(x = permutedims(x, $permutation)))
        $(all(==(1), repeat_dims) || :(
            x = reshape(x, $(reshape_pre_repeat(length(left_names), positions)));
            x = repeat(x, $(repeat_dims...))
        ))
        $(isnothing(shape_out) || :(x = reshape(x, $shape_out)))
        return x
    end
end

Base.repeat(x::AbstractArray{<:AbstractArray}, pattern::ArrowPattern; context...) = repeat(stack(x), pattern; context...)
Base.repeat(x, pattern::ArrowPattern; context...) = repeat(stack(x), pattern; context...)
