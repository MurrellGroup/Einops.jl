struct Pattern{L,R} end

# TODO: document -->
(-->)(L, R) = Pattern{L, R}()

Base.show(io::IO, ::Pattern{L,R}) where {L,R} = print(io, "$L --> $R")
Base.iterate(::Pattern{L}) where L = (L, Val(:R))
Base.iterate(::Pattern{<:Any,R}, ::Val{:R}) where R = (R, nothing)
Base.iterate(::Pattern, ::Nothing) = nothing


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
    inds = Int[]
    for (i, el_type) in enumerate(xs.parameters)
        el_type <: T && push!(inds, i)
    end
    return Expr(:tuple, inds...)
end

const Ignored = typeof(-)
const ShapePattern{N} = NTuple{N,Union{Symbol,Ignored}}

"""
    parse_shape(x, pattern)

Capture the shape of an array in a pattern by naming dimensions using `Symbol`s,
and `-` to ignore dimensions.

```jldoctest
julia> parse_shape(rand(2,3,4), (:a, :b, -))
(a = 2, b = 3)

julia> parse_shape(rand(2,3), (-, -))
NamedTuple()

julia> parse_shape(rand(2,3,4,5), (:first, :second, :third, :fourth))
(first = 2, second = 3, third = 4, fourth = 5)
```

The output is a `NamedTuple`, whose type contains the `Symbol` elements of the `pattern::NTuple{N,Union{Symbol,typeof(-)}}`,
meaning that, unless the pattern is [constant-propagated](https://discourse.julialang.org/t/how-does-constant-propagation-work/22735/4),
the output type is not known at compile time.

`@code_warntype parse_shape(rand(2,3,4), (:a, :b, -))`

`h() = parse_shape(rand(2,3,4), (:a, :b, -)); @code_warntype h()`
"""
function parse_shape(x::AbstractArray{<:Any,N}, pattern::ShapePattern{N}) where N
    names = extract(Symbol, pattern)
    allunique(names) || error("Pattern $(pattern) has duplicate elements")
    inds = findtype(Symbol, pattern)
    return NamedTuple{names,NTuple{length(inds),Int}}(size(x, i) for i in inds)
end


function permutation_mapping(left::NTuple{N,T}, right::NTuple{N,T}) where {N,T}
    perm::Vector{Int} = findfirst.(isequal.([right...]), Ref([left...]))
    return ntuple(i -> perm[i], Val(N))
end


# fix for 1.10:
_permutedims(x::AbstractArray{T,0}, ::Tuple{}) where T = x
_permutedims(x, perm) = permutedims(x, perm)
