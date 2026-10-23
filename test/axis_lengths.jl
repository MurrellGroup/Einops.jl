using Einops
using Einops: @rearrange, @reshape, @reduce, @repeat
using Test

@testset "Axis lengths" begin
    @testset "a length the pattern fixes is checked" begin
        x = rand(3, 2)
        # an integer or a keyword fixes the whole axis
        @test reduce(sum, x, einops"3 b -> b") == vec(sum(x, dims=1))
        @test_throws "Input axis 1 has length 3, but the pattern expects 4" reduce(sum, x, einops"4 b -> b")
        @test_throws "Input axis 1 has length 3, but the pattern expects 4" @reduce(sum, x, "4 b -> b")
        @test_throws "expects 4 (a = 4)" reduce(sum, x, einops"a b -> b"; a=4)
        @test_throws "expects 4 (a = 4)" rearrange(x, einops"a b -> b a"; a=4)
        @test_throws "expects 4 (a = 4)" reshape(x, einops"a b -> (a b)"; a=4)
        @test_throws "expects 4 (a = 4)" repeat(x, einops"a b -> a b r"; a=4, r=2)
        @test_throws "expects 4 (a = 4)" @rearrange(x, "a b -> b a", a=4)
        @test rearrange(x, einops"a b -> b a"; a=3) == permutedims(x)
        # `1` is the length-one case of the same rule
        @test_throws "Input axis 1 has length 3, but the pattern expects 1" @reduce(sum, x, "1 b -> b")
        @test_throws "Input axis 1 has length 3, but the pattern expects 1" @rearrange(x, "1 b -> b")
        @test @reduce(sum, rand(1, 2), "1 b -> b") |> size == (2,)

        # a group with one name left to infer: the known part must divide the axis
        v = collect(1.0:6)
        @test @reduce(sum, v, "(2 c) -> c") == [3.0, 7.0, 11.0]
        @test_throws "expects a multiple of 4" @reduce(sum, v, "(4 c) -> c")
        @test_throws "expects a multiple of 4 (a = 4)" rearrange(v, einops"(a c) -> a c"; a=4)
        # a group with every length given must match the axis exactly
        @test_throws "expects 8 (a = 2, c = 4)" rearrange(v, einops"(a c) -> a c"; a=2, c=4)
        @test rearrange(v, einops"(a c) -> a c"; a=2, c=3) == reshape(v, 2, 3)

        # past an ellipsis, the message names the input axis, not the pattern entry
        y = rand(2, 3, 5, 4)
        @test reduce(sum, y, einops"a ... 4 -> a ...") ≈ dropdims(sum(y, dims=4), dims=4)
        @test_throws "Input axis 4 has length 4, but the pattern expects 3" reduce(sum, y, einops"a ... 3 -> a ...")
        @test_throws "Input axis 4 has length 4, but the pattern expects 3" @reduce(sum, y, "a ... 3 -> a ...")
        @test_throws "Input axis 4 has length 4" @rearrange(y, "a ... t -> t ... a", t=3)
        @test_throws "Input axis 4 has length 4" @repeat(y, "a ... t -> a ... t r", t=3, r=2)
        @test_throws "Input axis 4 has length 4" @reshape(y, "a ... t -> a ... t", t=3)

        # einsum checks the lengths given for its operands
        A, B = rand(2, 3), rand(3, 4)
        @test_throws "expects 5 (j = 5)" einsum(A, B, einops"i j, j k -> i k"; j=5)
    end

    @testset "a unit inside a group adds nothing (#60)" begin
        x = rand(2, 3)
        @test @rearrange(x, "a b -> (b a 1)") == vec(permutedims(x))
        @test rearrange(x, einops"a b -> (1 b) (a 1)") == permutedims(x)
        @test rearrange(x, einops"a b -> a (1) b ()") == reshape(x, 2, 1, 3, 1)
        @test rearrange(vec(x), einops"(1 b) -> b") == vec(x)
        @test rearrange(rand(1, 2), einops"(1) b -> b") |> size == (2,)
        @test reduce(sum, reshape(x, 6), einops"(a 1 b) -> b"; a=2) == vec(sum(x, dims=1))
        @test_throws "Input axis 1 has length 2, but the pattern expects 1" rearrange(x, einops"(1 1) b -> b")
        for y in (rand(2, 3), rand(2, 3, 4))
            @test @rearrange(y, "a ... -> (... 1 a)") == rearrange(y, einops"a ... -> (... a)")
            @test rearrange(y, einops"a ... -> (... 1 a)") == rearrange(y, einops"a ... -> (... a)")
        end
        @test einsum(x, rand(3, 4), einops"i j, j k -> (i 1 k)") |> size == (8,)
        # other integers can't be carried through, and say so before anything runs
        @test_throws "Integer 3 in pattern" @macroexpand @rearrange(x, "a b -> (b a 3)")
        @test_throws "Integer 3 in pattern" rearrange(rand(6), einops"(a 3) -> a")
    end
end
