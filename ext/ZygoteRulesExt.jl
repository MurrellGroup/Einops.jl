module ZygoteRulesExt

using Einops: ArrowPattern
using ZygoteRules: @adjoint

# Zygote's catch-all rule for `reshape(xs, dims...)` takes no keywords, so it fails on
# `reshape(x, pattern; a=2)` (#51), and it takes precedence over any `rrule`. This rule is
# more specific. A pattern's `reshape` keeps the elements in order, splitting and merging
# axes without permuting them, so the gradient is the incoming one in the input's shape.
@adjoint function Base.reshape(x::AbstractArray, pattern::ArrowPattern; context...)
    return reshape(x, pattern; context...), Δ -> (reshape(Δ, size(x)), nothing)
end

end
