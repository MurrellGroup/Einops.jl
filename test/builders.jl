using Einops
using Einops: rearrange_body, rearrange_body_ellipsis, reshape_body, reshape_body_ellipsis,
              reduce_body, reduce_body_ellipsis, repeat_body, repeat_body_ellipsis, nth_keep_index
using Test

# The plan builders run only at compile time (inside the `@generated` ops) and at
# macro-expansion time (inside the `@…` macros), so line-coverage never attributes them
# even though every op test exercises them. Call them directly here: building a `quote`
# evaluates its `$(...)` interpolations, so the lines register. Each result is also
# `@eval`'d inside `Einops` (where `Rewrap` et al. resolve) and run, so these double as
# isolated correctness checks against the public ops.

@eval Einops begin
    _cov_rearrange(x)     = (context = (; a=2); $(rearrange_body(2, ((:a,:b),:c), (:c,(:a,:b)), (:a,))))
    _cov_rearrange_ell(x) =                     $(rearrange_body_ellipsis((:a, .., :b), (:a, (:b, ..)), ()))
    _cov_reshape(x)       =                     $(reshape_body(3, (:a,:b,:c), ((:a,:b),:c), ()))
    _cov_reshape_ell(x)   = (context = (; a=2); $(reshape_body_ellipsis((1, (:a,:b), ..), (:a, (:b, ..)), (:a,))))
    _cov_reduce(f, x)     =                     $(reduce_body(3, (:a,:b,:c), (:a,:c), ()))
    _cov_reduce_ell(f, x) =                     $(reduce_body_ellipsis((:c, .., :t), (:c, ..), ()))
    _cov_repeat(x)        = (context = (; r=2); $(repeat_body(2, (:a,:b), (:a,:b,:r), (:r,))))
    _cov_repeat_ell(x)    = (context = (; r=2); $(repeat_body_ellipsis((:a, ..), (:a, .., :r), (:r,))))
end

@testset "plan builders" begin
    @testset "static builders match the ops" begin
        x = rand(6, 5)
        @test Einops._cov_rearrange(x) == rearrange(x, einops"(a b) c -> c (a b)"; a=2)
        x3 = rand(2, 3, 5)
        @test Einops._cov_reshape(x3) == reshape(x3, einops"a b c -> (a b) c")
        @test Einops._cov_reduce(sum, x3) == reduce(sum, x3, einops"a b c -> a c")
        @test Einops._cov_repeat(x) == repeat(x, einops"a b -> a b r"; r=2)
    end

    @testset "ellipsis builders match the ops" begin
        x4 = rand(2, 3, 4, 5)
        @test Einops._cov_rearrange_ell(x4) == rearrange(x4, einops"a ... b -> a (b ...)")
        xr = rand(1, 6, 2, 3)
        @test Einops._cov_reshape_ell(xr) == reshape(xr, einops"1 (a b) ... -> a (b ...)"; a=2)
        @test Einops._cov_reduce_ell(sum, x4) == reduce(sum, x4, einops"c ... t -> c ...")
        x2 = rand(2, 3)
        @test Einops._cov_repeat_ell(x2) == repeat(x2, einops"a ... -> a ... r"; r=2)
    end

    @testset "nth_keep_index defensive branch" begin
        @test_throws ErrorException nth_keep_index(Expr(:tuple), 1)
    end
end
