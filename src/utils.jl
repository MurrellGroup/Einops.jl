pairs_type_to_names(::Type{<:Base.Pairs{Symbol,<:Any,<:Any,<:NamedTuple{names}}}) where names = names

function get_shape_in(N, left, context_names)
    length(left) == N || throw(ArgumentError("Input length $(length(left)) does not match array dimensionality $N"))
    allunique(extract(Symbol, left)) || throw(ArgumentError("Left names $(left) are not unique"))
    left isa Tuple{Vararg{Symbol}} && return nothing
    new_shape = :()
    sizes = new_shape.args
    for (i, input_dim) in enumerate(left)
        if input_dim isa Int
            input_dim == 1 || throw(ArgumentError("Singleton dimension size is not 1: $input_dim"))
            continue
        elseif input_dim isa Symbol
            push!(sizes, :(size(x, $i)))
        elseif input_dim isa Tuple{Symbol}
            push!(sizes, :(size(x, $i)))
        elseif input_dim isa Tuple{Vararg{Symbol}}
            known_size = filter(in(context_names), input_dim)
            length(input_dim) - length(known_size) <= 1 || throw(ArgumentError("Unknown dimension sizes: $(filter(∉(context_names), input_dim))"))
            unknown_size = Expr(:call, :*, (:(context[$(QuoteNode(name))]) for name in known_size)...)
            new_dim_size = Tuple(name in known_size ? :(context[$(QuoteNode(name))]) : :(size(x, $i) ÷ $unknown_size) for name in input_dim)
            push!(sizes, new_dim_size...)
        else
            throw(ArgumentError("Invalid input dimension: $input_dim"))
        end
    end
    return new_shape
end

get_mapping(left, right) = Tuple(findfirst.(isequal.(right), (left,)))

function get_permutation(left, right)
    isempty(setdiff(left, right)) || throw(ArgumentError("Set of left names $(left) does not match set of right names $(right)"))
    return get_mapping(left, right)
end

function get_shape_out(right)
    allunique(extract(Symbol, right)) || throw(ArgumentError("Right names $(right) are not unique"))
    right isa Tuple{Vararg{Symbol}} && return nothing
    new_shape = :()
    sizes = new_shape.args
    i = 1
    for dim in right
        if dim isa Int
            dim == 1 || throw(ArgumentError("Singleton dimension size is not 1: $dim"))
            push!(sizes, dim)
        elseif dim isa Symbol
            push!(sizes, :(size(x, $i)))
            i += 1
        elseif dim isa Tuple{Symbol}
            push!(sizes, :(size(x, $i)))
            i += 1
        elseif dim isa Tuple{Vararg{Symbol}}
            push!(sizes, isempty(dim) ? 1 : Expr(:call, :*, (:(size(x, $i)) for i in i:i+length(dim)-1)...))
            i += length(dim)
        end
    end
    return new_shape
end

# fix for 1.10:
_permutedims(x::AbstractArray{T,0}, ::Tuple{}) where T = x
_permutedims(x, perm::NTuple{N,Int}) where N = permutedims(x, perm)
