module ReactantExt

using Einops: Einops, Rewrap, get_mapping
using Reactant: Reactant, TracedRArray, AnyTracedRArray, Ops

const materialize = Reactant.TracedUtils.materialize_traced_array

ints(dims::Tuple) = Int[dims...]

# `Base.reshape` stays lazy around a traced array. Materializing the wrapper emits a
# `stablehlo.reshape`, so a pattern that ends in a merge or split returns a plain array
# from a compiled function instead of a `ReshapedArray` over a device buffer.
Einops._reshape(x::AnyTracedRArray, shape) = materialize(Rewrap.reshape(materialize(x), shape))

# Axes present in one operand only are summed out first, so `dot_general` only ever sees
# batch, contracting and free axes. `Ops.reduce` drops the reduced axes itself, which keeps
# this a single op and avoids `dropdims`, whose reshape does not stay traced.
sumout(x::TracedRArray{T}, dims) where {T} =
    isempty(dims) ? x : Ops.reduce(x, Ops.constant(zero(T)), ints(dims), +)

function pairwise(ix::Tuple, iy::Tuple, iout::Tuple)
    for labels in (ix, iy)
        allunique(labels) || throw(ArgumentError("Repeated label within one operand is not supported by this backend: $labels"))
    end
    allunique(iout) || throw(ArgumentError("Output labels $iout are not unique"))
    for label in iout
        label in ix || label in iy || throw(ArgumentError("Output label $label does not appear in any input"))
    end
    sumx = Tuple(d for (d, label) in enumerate(ix) if label ∉ iy && label ∉ iout)
    sumy = Tuple(d for (d, label) in enumerate(iy) if label ∉ ix && label ∉ iout)
    # What the primitive sees once the summed-out axes are gone
    kx = Tuple(label for label in ix if label ∈ iy || label ∈ iout)
    ky = Tuple(label for label in iy if label ∈ ix || label ∈ iout)
    shared = Tuple(label for label in kx if label ∈ ky)
    batch_labels = Tuple(label for label in shared if label ∈ iout)
    contract_labels = Tuple(label for label in shared if label ∉ iout)
    batch = (get_mapping(kx, batch_labels), get_mapping(ky, batch_labels))
    contract = (get_mapping(kx, contract_labels), get_mapping(ky, contract_labels))
    free_x = Tuple(label for label in kx if label ∉ ky)
    free_y = Tuple(label for label in ky if label ∉ kx)
    perm = get_mapping((batch_labels..., free_x..., free_y...), iout)
    return (; sumx, sumy, batch, contract, perm)
end

function traced_pair(ixs, iout, x::TracedRArray, y::TracedRArray)
    plan = pairwise(ixs[1], ixs[2], iout)
    z = Ops.dot_general(
        sumout(x, plan.sumx), sumout(y, plan.sumy);
        contracting_dimensions=map(ints, plan.contract),
        batching_dimensions=map(ints, plan.batch),
    )
    return plan.perm == ntuple(identity, length(plan.perm)) ? z : permutedims(z, plan.perm)
end

# `AnyTracedRArray` also covers the wrappers Einops itself produces: grouped axes arrive
# from `expand` as lazy `ReshapedArray`s, and a transposed operand as an `Adjoint`.
# A host array next to a traced one is promoted the way Reactant promotes it for `*`,
# i.e. embedded as a constant.
Einops.contract(ixs::NTuple{2,Tuple}, iout::Tuple, x::AnyTracedRArray, y::AnyTracedRArray; kws...) =
    traced_pair(ixs, iout, materialize(x), materialize(y))
Einops.contract(ixs::NTuple{2,Tuple}, iout::Tuple, x::AnyTracedRArray, y::AbstractArray; kws...) =
    traced_pair(ixs, iout, materialize(x), Reactant.promote_to(TracedRArray, y))
Einops.contract(ixs::NTuple{2,Tuple}, iout::Tuple, x::AbstractArray, y::AnyTracedRArray; kws...) =
    traced_pair(ixs, iout, Reactant.promote_to(TracedRArray, x), materialize(y))

end
