const Wildcard = typeof(*)
const PackPattern{N} = NTuple{N,Union{Symbol,Wildcard}}

function check_packing_pattern(pattern::PackPattern)
    count(x -> x isa Wildcard, pattern) == 1 || error("Only one wildcard (*) is allowed in the pattern")
    allunique(pattern) || error("Pattern $(pattern) has duplicate elements")
    return nothing
end

find_wildcard(pattern::PackPattern) = findfirst(x -> x isa Wildcard, pattern)

size_before_wildcard(dims::Dims, pattern::PackPattern) = dims[1:find_wildcard(pattern)-1]
size_after_wildcard(dims::Dims, pattern::PackPattern) = dims[end-length(pattern)+find_wildcard(pattern)+1:end]
size_wildcard(dims::Dims, pattern::PackPattern) = dims[begin+length(size_before_wildcard(dims, pattern)):end-length(size_after_wildcard(dims, pattern))]
packed_size(dims::Dims, pattern::PackPattern) = (size_before_wildcard(dims, pattern)..., prod(size_wildcard(dims, pattern)), size_after_wildcard(dims, pattern)...)

"""
    pack(unpacked_arrays, pattern)

Pack a vector of arrays into a single array according to the pattern.

!!! note
    This function has not been thoroughly tested, and may not be type stable or differentiable.

# Examples

```jldoctest
julia> inputs = [rand(2,3,5), rand(2,3,7,5), rand(2,3,7,9,5)];

julia> packed_array, packed_shapes = pack(inputs, (:i, :j, *, :k));

julia> size(packed_array)
(2, 3, 71, 5)

julia> packed_shapes
3-element Vector{NTuple{N, Int64} where N}:
 ()
 (7,)
 (7, 9)
```
"""
function pack(unpacked_arrays, pattern::PackPattern{N}) where N
    check_packing_pattern(pattern)
    reshaped_arrays = [reshape(A, packed_size(size(A), pattern)::Dims{N})::AbstractArray{<:Any,N} for A in unpacked_arrays]
    concatenated_array::AbstractArray{<:Any,N} = cat(reshaped_arrays..., dims=find_wildcard(pattern))
    packed_shapes = Dims[size_wildcard(size(unpacked_array), pattern) for unpacked_array in unpacked_arrays]
    return concatenated_array, packed_shapes
end

pack(unpacked_arrays, ::Val{T}) where T = pack(unpacked_arrays, T)

splice(a::Dims, i::Int, r::Dims) = (a[1:i-1]..., r..., a[i+1:end]...)

"""
    unpack(packed_array, packed_shapes, pattern)

!!! note
    This function has not been thoroughly tested, and may not be type stable or differentiable.

Unpack a single array into a vector of arrays according to the pattern.

# Examples

```jldoctest
julia> inputs = [rand(2,3,5), rand(2,3,7,5), rand(2,3,7,9,5)];

julia> inputs == unpack(pack(inputs, (:i, :j, *, :k))..., (:i, :j, *, :k))
true

julia> packed_array = rand(2,3,16);

julia> packed_shapes = [(), (7,), (4, 2)];

julia> unpack(packed_array, packed_shapes, (:i, :j, *)) .|> size
3-element Vector{Tuple{Int64, Int64, Vararg{Int64}}}:
 (2, 3)
 (2, 3, 7)
 (2, 3, 4, 2)
```
"""
function unpack(packed_array::AbstractArray{<:Any,N}, packed_shapes, pattern::PackPattern{N}) where N
    check_packing_pattern(pattern)
    inds = Iterators.accumulate(+, Iterators.map(prod, packed_shapes))
    unpacked_arrays = map(Iterators.flatten((0, inds)), inds, packed_shapes) do i, j, ps
        unpacked_array = selectdim(packed_array, find_wildcard(pattern), i+1:j)
        return collect(reshape(unpacked_array, splice(size(unpacked_array), find_wildcard(pattern), ps)))
    end
    return unpacked_arrays
end

unpack(packed_array, packed_shapes, ::Val{T}) where T = unpack(packed_array, packed_shapes, T)
