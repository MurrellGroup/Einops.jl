extract(::Type, ::Tuple{}) = ()
function extract(T::Type, input_tuple::Tuple)
    first_element = first(input_tuple)
    rest_elements = Base.tail(input_tuple)
    instances_from_first = if first_element isa T
        (first_element,)
    elseif first_element isa Tuple
        extract(T, first_element)
    else
        ()
    end
    return (instances_from_first..., extract(T, rest_elements)...)
end

@generated function findtype(::Type{T}, xs::Tuple) where T
    return Expr(:tuple, (i for (i, el_type) in enumerate(xs.parameters) if el_type <: T)...)
end

function anonymous_symbols(prefix, N)
    symbols = []
    for i in 1:N
        push!(symbols, Symbol("$(prefix)_$i"))
    end
    return Tuple(symbols)
end

function remove_anonymous_dims(side)
    integers = filter(!isone, extract(Integer, side))
    symbols = anonymous_symbols(:__anon_dim, length(integers))
    context = NamedTuple{symbols}(integers)
    i = 0
    new_side = map(side) do t
        if t isa Integer && !isone(t)
            i += 1
            symbols[i]
        elseif t isa Tuple && any(i -> i isa Integer && !isone(i), t)
            while any(i -> i isa Integer && !isone(i), t)
                i += 1
                t = insertat(t, findfirst(i -> i isa Integer && !isone(i), t), (symbols[i],))
            end
            t
        else
            t
        end
    end
    return new_side, context
end
