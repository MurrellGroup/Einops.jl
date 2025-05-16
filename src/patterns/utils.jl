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


@generated function anonymous_symbols(::Val{prefix}, ::Val{N}) where {prefix,N}
    ex = :(())
    for i in 1:N
        ex = :((($ex)..., $(:(Symbol(:($prefix), '_', $i)))))
    end
    ex
end

anonymous_symbols(prefix::Symbol, n::Int) = anonymous_symbols(Val(prefix), Val(n))


@generated function remove_anonymous_dims(::Val{side}) where side
    integers = filter(!isone, extract(Integer, side))
    symbols = anonymous_symbols(:__anon_dim, length(integers))
    context = NamedTuple{symbols}(integers)
    i = 0
    new_side = map(side) do t
        if t isa Integer && !isone(t)
            i += 1
            symbols[i]
        else
            t
        end
    end
    :($new_side, $context)
end

remove_anonymous_dims(side) = remove_anonymous_dims(Val(side))
