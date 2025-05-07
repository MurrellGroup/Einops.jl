function permutation_mapping(left::NTuple{N,T}, right::NTuple{N,T}) where {N,T}
    perm::Vector{Int} = findfirst.(isequal.([right...]), Ref([left...]))
    return ntuple(i -> perm[i], Val(N))
end

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

# fix for 1.10:
_permutedims(x::AbstractArray{T,0}, ::Tuple{}) where T = x
_permutedims(x, perm) = permutedims(x, perm)
