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
end
