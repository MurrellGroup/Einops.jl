using Einops
using Einops: @rearrange, @reshape, @reduce, @repeat, @einsum
using Test

@testset "Axis lengths" begin
    @testset "a length the pattern fixes is checked" begin
        x = rand(3, 2)
        # an integer or a keyword fixes the whole axis
        @test reduce(sum, x, einops"3 b -> b") == vec(sum(x, dims=1))
        @test_throws "Axis 1 of the input has length 3, but the pattern expects 4" reduce(sum, x, einops"4 b -> b")
        @test_throws "Axis 1 of the input has length 3, but the pattern expects 4" @reduce(sum, x, "4 b -> b")
        @test_throws "expects 4 (a = 4)" reduce(sum, x, einops"a b -> b"; a=4)
        @test_throws "expects 4 (a = 4)" rearrange(x, einops"a b -> b a"; a=4)
        @test_throws "expects 4 (a = 4)" reshape(x, einops"a b -> (a b)"; a=4)
        @test_throws "expects 4 (a = 4)" repeat(x, einops"a b -> a b r"; a=4, r=2)
        @test_throws "expects 4 (a = 4)" @rearrange(x, "a b -> b a", a=4)
        @test rearrange(x, einops"a b -> b a"; a=3) == permutedims(x)
        # `1` is the length-one case of the same rule
        @test_throws "Axis 1 of the input has length 3, but the pattern expects 1" @reduce(sum, x, "1 b -> b")
        @test_throws "Axis 1 of the input has length 3, but the pattern expects 1" @rearrange(x, "1 b -> b")
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
        @test_throws "Axis 4 of the input has length 4, but the pattern expects 3" reduce(sum, y, einops"a ... 3 -> a ...")
        @test_throws "Axis 4 of the input has length 4, but the pattern expects 3" @reduce(sum, y, "a ... 3 -> a ...")
        @test_throws "Axis 4 of the input has length 4" @rearrange(y, "a ... t -> t ... a", t=3)
        @test_throws "Axis 4 of the input has length 4" @repeat(y, "a ... t -> a ... t r", t=3, r=2)
        @test_throws "Axis 4 of the input has length 4" @reshape(y, "a ... t -> a ... t", t=3)

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
        @test_throws "Axis 1 of the input has length 2, but the pattern expects 1" rearrange(x, einops"(1 1) b -> b")
        for y in (rand(2, 3), rand(2, 3, 4))
            @test @rearrange(y, "a ... -> (... 1 a)") == rearrange(y, einops"a ... -> (... a)")
            @test rearrange(y, einops"a ... -> (... 1 a)") == rearrange(y, einops"a ... -> (... a)")
        end
        @test einsum(x, rand(3, 4), einops"i j, j k -> (i 1 k)") |> size == (8,)
        # other integers can't be carried through, and say so before anything runs
        @test_throws "Unnamed axis `3`" @macroexpand @rearrange(x, "a b -> (b a 3)")
        @test_throws "Unnamed axis `3`" rearrange(rand(6), einops"(a 3) -> a")
    end

    @testset "interpolation (#35)" begin
        x = reshape(collect(1.0:24), 6, 4)
        k = 2
        p = einops"($k b) c -> b c"
        @test p isa Einops.BoundPattern
        # it stands for the value, as if written there
        @test sprint(show, p) == "einops\"(2 b) c -> b c\""
        @test reduce(sum, x, p) == reduce(sum, x, einops"(2 b) c -> b c")
        @test @reduce(sum, x, "($k b) c -> b c") == reduce(sum, x, p)
        # an expression, evaluated where the pattern is
        @test repeat(x, einops"a b -> a b $(k + 1)") == repeat(x, einops"a b -> a b 3")
        @test @repeat(x, "a b -> a b $(k + 1)") == repeat(x, einops"a b -> a b 3")
        # each occurrence is an axis of its own (#36)
        y = reshape(collect(1.0:16), 4, 4)
        @test @reduce(sum, y, "(h $k) (w $k) -> h w") == reduce(sum, y, einops"(h 2) (w 2) -> h w")
        @test reduce(sum, y, einops"(h $k) (w $k) -> h w") == reduce(sum, y, einops"(h 2) (w 2) -> h w")
        @test @repeat(x, "a b -> ($k a) ($k b)") == repeat(x, einops"a b -> (2 a) (2 b)")
        @test repeat(x, einops"a b -> ($k a) ($k b)") == repeat(x, einops"a b -> (2 a) (2 b)")
        # a length of one is still an axis of length one, reduced or created
        one = 1
        @test @reduce(sum, rand(1, 3), "$one b -> b") |> size == (3,)
        @test @repeat(rand(3), "b -> b $one") |> size == (3, 1)
        # like an integer, it can't be carried through, whatever its value
        @test_throws "Unnamed axis `\$k`" rearrange(x, einops"($k d) n -> d n $k")
        @test_throws "Unnamed axis `\$k`" @macroexpand @rearrange(x, "($k d) n -> d n $k")
        @test_throws "Unnamed axis `\$one`" @macroexpand @rearrange(x, "a b -> a b $one")
        @test_throws "Unnamed axis `\$k`" reshape(x, einops"($k b) c -> $k (b c)")
        @test_throws "Unnamed axis `\$k`" @macroexpand @reduce(sum, x, "a b -> a $k")
        @test_throws "Unnamed axis `\$(k + 1)`" @macroexpand @repeat(x, "($(k + 1) a) b -> a b")
        # einsum could only sum such an axis out, which a name does too
        A, B = rand(6, 3), rand(3, 5)
        @test_throws "isn't supported by `einsum`" einsum(A, B, einops"($k i) j, j l -> i l")
        @test_throws "isn't supported by `einsum`" @macroexpand @einsum(A, B, "($k i) j, j l -> i l")
        # it broadcasts like a pattern
        z = reduce(sum, x, einops"(2 b) c -> b c")
        @test reduce.(sum, (x, x), einops"($k b) c -> b c") == (z, z)
        # interpolated lengths are checked like any other
        @test_throws "expects a multiple of 4" reduce(sum, x, einops"($(k + 2) b) c -> b c")
        @test_throws "Axis 1 of the input has length 6, but the pattern expects 2" @reduce(sum, x, "$k b -> b")
        # only arrow patterns take it, and `$` needs something to interpolate
        @test_throws "needs a pattern with `->`" Einops.parse_interpolated_pattern(Einops.interpolation_parts("a \$k"))
        @test_throws "must be followed by" Einops.interpolation_parts("a \$ b -> a")
    end

    @testset "interpolated lengths stay out of the type (#34)" begin
        x = rand(6, 4)
        f(x, n) = reduce(sum, x, einops"($n b) c -> b c")
        g(x, n) = @reduce(sum, x, "($n b) c -> b c")
        r(x, n) = repeat(x, einops"a b -> a ($n b)")
        @test (@inferred f(x, 3)) == reduce(sum, x, einops"(3 b) c -> b c")
        @test (@inferred g(x, 2)) == reduce(sum, x, einops"(2 b) c -> b c")
        @test (@inferred r(x, 3)) == repeat(x, einops"a b -> a (3 b)")
        # an integer put in a pattern's type is what the above avoids
        h(x, n) = reduce(sum, x, Einops.ArrowPattern{((n, :b), :c), (:b, :c)}())
        @test h(x, 3) == f(x, 3)
        @test_throws ErrorException @inferred h(x, 3)
    end
end
