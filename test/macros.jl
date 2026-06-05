using Einops
using Einops: @rearrange, @reshape, @reduce, @repeat, @einsum
using Test

@testset "Shorthand macros" begin
    @testset "@rearrange" begin
        x = rand(2, 3, 5)
        @test @rearrange(x, "a b c -> c b a") == rearrange(x, einops"a b c -> c b a")
        @test @rearrange(x, "a b c -> a (c b)") == rearrange(x, einops"a b c -> a (c b)")
    end

    @testset "@reshape" begin
        x = rand(2, 3, 5)
        @test @reshape(x, "a b c -> (a b) c") == reshape(x, einops"a b c -> (a b) c")
    end

    @testset "@reduce" begin
        x = rand(2, 3)
        @test @reduce(sum, x, "a b -> a") == reduce(sum, x, einops"a b -> a")
    end

    @testset "@repeat" begin
        x = rand(2, 3)
        @test @repeat(x, "a b -> a b c", c=4) == repeat(x, einops"a b -> a b c", c=4)
        @test @repeat(x, "a b -> a b c"; c=4) == repeat(x, einops"a b -> a b c"; c=4)
    end

    @testset "@einsum" begin
        x, y = rand(2, 3), rand(3, 4)
        @test @einsum(x, y, "i j, j k -> i k") == einsum(x, y, einops"i j, j k -> i k")
    end

    @testset "context keyword" begin
        x = rand(6, 5)
        ref = rearrange(x, einops"(a b) c -> a b c"; a=2)
        # comma form `, a=2`
        @test @rearrange(x, "(a b) c -> a b c", a=2) == ref
        # explicit `; a=2`
        @test @rearrange(x, "(a b) c -> a b c"; a=2) == ref
        # bare-symbol shorthand `; a`
        a = 2
        @test @rearrange(x, "(a b) c -> a b c"; a) == ref
        # splat `; context...`
        context = (; a=2)
        @test @rearrange(x, "(a b) c -> a b c"; context...) == ref
        # property shorthand `; obj.a`  ==>  `a = obj.a`
        @test @rearrange(x, "(a b) c -> a b c"; context.a) == ref
    end

    @testset "error handling" begin
        @test_throws ArgumentError @macroexpand @rearrange(x, y)
        @test_throws ArgumentError @macroexpand @rearrange(x, "a -> a", "b -> b")
    end

    # The macros inline the plan at the call site. Every case below also asserts
    # equality with the generated-function path, so the inlined plan and the
    # generated plan are checked against each other.
    is_inlined(ex) = Meta.isexpr(ex, :let)

    @testset "inlining vs fallback" begin
        x = rand(2, 3, 5)
        # statically sizable -> inlined `let`, no `rearrange` call, no Base.Pairs
        @test is_inlined(@macroexpand @rearrange(x, "a b c -> c b a"))
        @test is_inlined(@macroexpand @reshape(x, "a b c -> (a b) c"))
        @test is_inlined(@macroexpand @reduce(sum, x, "a b c -> a c"))
        @test is_inlined(@macroexpand @repeat(x, "a b -> a b r", r=2))
        # ellipsis -> still inlined (rank rides in via Val(ndims(x)))
        @test is_inlined(@macroexpand @rearrange(x, "a ... -> ... a"))
        # einsum -> always a call (not type stable / OMEinsum path)
        @test !is_inlined(@macroexpand @einsum(x, x, "a b c, a b c -> a"))
        # `; kws...` splat with a multi-symbol left group -> fall back to call
        @test !is_inlined(@macroexpand @rearrange(x, "(a b) c -> a b c"; ctx...))
        # `; kws...` splat without such a group -> still inlined (splat carries values)
        @test is_inlined(@macroexpand @repeat(x, "a b c -> a b c r"; ctx...))
    end

    @testset "splat fallback correctness" begin
        x = rand(6, 5)
        ctx = (; a = 2)
        @test @rearrange(x, "(a b) c -> a b c"; ctx...) == rearrange(x, einops"(a b) c -> a b c"; ctx...)
        r = (; r = 3)
        @test @repeat(x, "a c -> a c r"; r...) == repeat(x, einops"a c -> a c r"; r...)
    end

    @testset "ellipsis inlining vs generated" begin
        @testset "rearrange" begin
            for x in (rand(2, 3), rand(2, 3, 4), rand(2, 3, 4, 5))
                @test @rearrange(x, "a ... b -> b ... a") == rearrange(x, einops"a ... b -> b ... a")
                @test @rearrange(x, "a ... b -> a ... b") == rearrange(x, einops"a ... b -> a ... b")
                @test @rearrange(x, "... a -> a ...") == rearrange(x, einops"... a -> a ...")
            end
            for x in (rand(2, 3, 4), rand(2, 3, 4, 5, 6))
                @test @rearrange(x, "a ... b -> (a b) ...") == rearrange(x, einops"a ... b -> (a b) ...")
                @test @rearrange(x, "a ... b -> a (b ...)") == rearrange(x, einops"a ... b -> a (b ...)")
            end
            for x in (rand(6, 3, 4), rand(6, 3, 4, 5))
                @test @rearrange(x, "(a c) ... b -> b ... a c", a=2) == rearrange(x, einops"(a c) ... b -> b ... a c"; a=2)
            end
        end

        @testset "reshape" begin
            for x in (rand(1, 6, 2, 3), rand(1, 6, 2, 3, 4))
                @test @reshape(x, "1 (a b) ... -> a (b ...)"; a=2) == reshape(x, einops"1 (a b) ... -> a (b ...)"; a=2)
            end
        end

        @testset "reduce" begin
            for x in (rand(4, 3, 5), rand(4, 3, 5, 6))
                @test @reduce(sum, x, "c ... t -> c ...") == reduce(sum, x, einops"c ... t -> c ...")
                @test @reduce(sum, x, "c ... t -> ... c") == reduce(sum, x, einops"c ... t -> ... c")
                @test @reduce(maximum, x, "c d ... -> c ...") == reduce(maximum, x, einops"c d ... -> c ...")
            end
        end

        @testset "repeat" begin
            for x in (rand(2, 3), rand(2, 3, 4))
                @test @repeat(x, "a ... -> a ... r", r=2) == repeat(x, einops"a ... -> a ... r"; r=2)
                @test @repeat(x, "a ... -> r a ...", r=2) == repeat(x, einops"a ... -> r a ..."; r=2)
                @test @repeat(x, "a ... -> a ...") == repeat(x, einops"a ... -> a ...")
            end
        end
    end

    @testset "inlined plans are type stable" begin
        # Mirrors the constant-folding cuTile relies on: rank enters via the arg type.
        r1(x) = @rearrange(x, "a ... b -> b ... a")
        r2(x) = @rearrange(x, "(a c) ... b -> b ... a c", a=2)
        r3(x) = @reshape(x, "1 (a b) ... -> a (b ...)"; a=2)
        r4(x) = @reduce(sum, x, "c ... t -> ... c")
        r5(x) = @repeat(x, "a ... -> a ... r", r=2)
        s1(x) = @rearrange(x, "(k1 m2 m1) k0 m0 -> (k1 k0) (m1 m2 m0)"; k1=4, m2=4)
        @test (@inferred r1(rand(2, 3, 4, 5)); true)
        @test (@inferred r2(rand(6, 3, 4, 5)); true)
        @test (@inferred r3(rand(1, 6, 2, 3, 4)); true)
        @test (@inferred r4(rand(4, 3, 5, 6)); true)
        @test (@inferred r5(rand(2, 3, 4)); true)
        @test (@inferred s1(rand(512, 7, 8)); true)
    end
end
