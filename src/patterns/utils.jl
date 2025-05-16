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


@generated function anonymous_symbols(::Val{N}) where N
    ex = :(())
    for i in 1:N
        ex = :((($ex)..., $(:(Symbol($i)))))
    end
    ex
end

anonymous_symbols(n::Int) = anonymous_symbols(Val(n))


@generated function anonymous_dims_to_context(::Val{side}) where side
    integers = filter(!isone, extract(Integer, side))
    symbols = anonymous_symbols(length(integers))
    context = NamedTuple{symbols}(integers)
    :($context)
end

anonymous_dims_to_context(side) = anonymous_dims_to_context(Val(side))
