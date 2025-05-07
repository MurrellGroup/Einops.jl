function reshape_in(x, left; context...)
    length(left) == ndims(x) || throw(ArgumentError("Input length $(length(left)) does not match number of dimensions $(ndims(x))"))
    new_shape = Int[]
    for (i, input_dim) in enumerate(left)
        if input_dim isa Int
            input_dim == 1 || throw(ArgumentError("Singleton dimension size is not 1: $input_dim"))
            continue
        elseif input_dim isa Symbol
            push!(new_shape, size(x, i))
        elseif input_dim isa Tuple{Vararg{Symbol}}
            known_size = filter(name -> haskey(context, name), input_dim)
            length(input_dim) - length(known_size) <= 1 || throw(ArgumentError("Unknown dimension sizes: $(filter(name -> !haskey(context, name), input_dim))"))
            known_size_prod = prod(context[name] for name in known_size)
            new_dim_size = (name in known_size ? context[name] : size(x, i) ÷ known_size_prod for name in input_dim)
            push!(new_shape, new_dim_size...)
        else
            throw(ArgumentError("Invalid input dimension: $input_dim"))
        end
    end
    return reshape(x, ntuple(i -> new_shape[i], length(extract(Symbol, left)))...)
end

reshape_in(x, ::Tuple{Vararg{Symbol}}; context...) = x

# TODO: static `TransmuteDims.transmute` using type parameters of `Pattern`
function permute(x, left, right)
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    isempty(setdiff(left_names, right_names)) || throw(ArgumentError("Set of left names $(left_names) does not match set of right names $(right_names)"))
    allunique(left_names) || throw(ArgumentError("Left names $(left_names) are not unique"))
    allunique(right_names) || throw(ArgumentError("Right names $(right_names) are not unique"))
    perm = permutation_mapping(left_names, right_names)
    return permutedims(x, perm)
end

# TODO: statically remove if no parentheses or singleton dimensions
len(dim::Tuple{Vararg{Symbol}}) = length(dim)
len(::Symbol) = 1
len(x::Int) = x == 1 ? 0 : throw(ArgumentError("Singleton dimension size is not 1: $x"))

function reshape_out(x, right)
    size_iter = Iterators.Stateful(size(x))
    shape = Int[prod(Iterators.take(size_iter, len(dim)); init=1) for dim in right]
    return reshape(x, shape...)
end

reshape_out(x, ::Tuple{Vararg{Symbol}}) = x

"""
    rearrange(x::AbstractArray, left => right)

Rearrange the axes of `x` according to the pattern specified by `left => right`.

Can always be expressed as a `reshape` + `permutedims` + `reshape`.

# Examples

```jldoctest
julia> rearrange(rand(2,3,5), (:a, :b, :c) => (:c, :b, :a)) |> size
(5, 3, 2)

julia> permutedims(rand(2,3,5), (3,2,1)) |> size
(5, 3, 2)

julia> rearrange(rand(2,3,35), (:a, :b, (:c, :d)) => (:a, :d, (:c, :b)), c=5) |> size
(2, 7, 15)

julia> reshape(permutedims(reshape(rand(2,3,35), 2,3,5,7), (1,4,3,2)), 2,7,5*3) |> size
(2, 7, 15)
```
"""
function rearrange(x, @nospecialize pattern::Pattern; context...)
    left, right = pattern
    (!isempty(extract(typeof(..), left)) || !isempty(extract(typeof(..), right))) && throw(ArgumentError("Ellipses (..) are currently not supported"))
    left_names, right_names = extract(Symbol, left), extract(Symbol, right)
    reshaped_in = reshape_in(x, left; context...)
    permuted = permute(reshaped_in, left_names, right_names)
    reshaped_out = reshape_out(permuted, right)
    return reshaped_out
end
