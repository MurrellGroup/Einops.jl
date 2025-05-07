const Wildcard = typeof(*)
const PackPattern{N} = NTuple{N,Union{Symbol,Wildcard}}

checkpacking(pattern::PackPattern) = count(x -> x isa Wildcard, pattern) == 1 || error("Only one wildcard (*) is allowed in the pattern")
find_wildcard(pattern::PackPattern) = findfirst(x -> x isa Wildcard, pattern)

size_before_wildcard(dims::Dims, pattern::PackPattern) = dims[1:find_wildcard(pattern)-1]
size_after_wildcard(dims::Dims, pattern::PackPattern) = dims[end-length(pattern)+find_wildcard(pattern)+1:end]
size_wildcard(dims::Dims, pattern::PackPattern) = dims[begin+length(size_before_wildcard(dims, pattern)):end-length(size_after_wildcard(dims, pattern))]
packed_size(dims::Dims, pattern::PackPattern) = (size_before_wildcard(dims, pattern)..., prod(size_wildcard(dims, pattern)), size_after_wildcard(dims, pattern)...)

function pack(unpacked_arrays::Vector, pattern::PackPattern{N}) where N
    checkpacking(pattern)
    reshaped_arrays = [reshape(A, packed_size(size(A), pattern)::Dims{N})::AbstractArray{<:Any,N} for A in unpacked_arrays]
    concatenated_array::AbstractArray{<:Any,N} = cat(reshaped_arrays..., dims=find_wildcard(pattern))
    packed_shapes = Dims[size_wildcard(size(A), pattern) for A in unpacked_arrays]
    return concatenated_array, packed_shapes
end

splice(a::Dims, i::Int, r::Dims) = (a[1:i-1]..., r..., a[i+1:end]...)

# FIXME: results in a reshape of a view ... so should we collect?
function unpack(packed_array::AbstractArray{<:Any,N}, packed_shapes, pattern::PackPattern{N}) where N
    checkpacking(pattern)
    inds = Iterators.accumulate(+, Iterators.map(prod, packed_shapes))
    unpacked_arrays = map(Iterators.flatten((0, inds)), inds, packed_shapes) do i, j, ps
        A = selectdim(packed_array, find_wildcard(pattern), i+1:j)
        reshape(A, splice(size(A), find_wildcard(pattern), ps))
    end
    return unpacked_arrays
end
