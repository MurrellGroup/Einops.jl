using Einops
using Test

@testset "Reshape Operations" begin
    @testset "basic reshapes" begin
        x = rand(2, 3, 5)
        @test reshape(x, (:a, :b, :c) --> (:a, :b, :c)) == x
        @test reshape(x, (:a, :b, :c) --> ((:a, :b), :c)) == reshape(x, 6, 5)
        @test reshape(x, (:a, :b, :c) --> (:a, (:b, :c))) == reshape(x, 2, 15)
        @test reshape(x, (:a, :b, :c) --> ((:a, :b, :c),)) == reshape(x, 30)
    end

    @testset "error on permutation" begin
        x = rand(2, 3, 5)
        @test_throws "reshape requires symbols in same order" reshape(x, (:a, :b, :c) --> (:c, :b, :a))
        @test_throws "reshape requires symbols in same order" reshape(x, (:a, :b, :c) --> (:b, :a, :c))
        @test_throws "reshape requires symbols in same order" reshape(x, (:a, :b, :c) --> ((:b, :a), :c))
    end

    @testset "ellipses support" begin
        x = rand(2, 3, 5, 7)
        @test reshape(x, (:a, ..) --> ((:a, ..),)) == reshape(x, :)
        @test reshape(x, (:a, :b, ..) --> ((:a, :b), ..)) == reshape(x, 6, 5, 7)
        @test reshape(x, (.., :c, :d) --> (.., (:c, :d))) == reshape(x, 2, 3, 35)
    end

    @testset "type inference" begin
        x = rand(2, 3, 5)
        @test (@inferred reshape(x, (:a, :b, :c) --> ((:a, :b), :c))) == reshape(x, 6, 5)
        @test (@inferred reshape(x, (:a, ..) --> ((:a, ..),))) == reshape(x, :)
    end

    @testset "dimension decomposition" begin
        x = rand(2, 15)
        @test reshape(x, (:a, (:b, :c)) --> (:a, :b, :c), b=3) == reshape(x, 2, 3, 5)
        @test reshape(x, (:a, (:b, :c)) --> (:a, :b, :c), c=5) == reshape(x, 2, 3, 5)

        x = rand(60)
        @test reshape(x, ((:a, :b, :c),) --> (:a, :b, :c), a=3, b=4) == reshape(x, 3, 4, 5)
    end

    @testset "singleton dimensions" begin
        x = rand(2, 1, 3)
        @test reshape(x, (:a, 1, :b) --> (:a, :b)) == reshape(x, 2, 3)
        @test reshape(x, (:a, 1, :b) --> ((:a, :b),)) == reshape(x, 6)

        x = rand(2, 3)
        @test reshape(x, (:a, :b) --> (:a, 1, :b)) == reshape(x, 2, 1, 3)
        @test reshape(x, (:a, :b) --> (1, :a, 1, :b, 1)) == reshape(x, 1, 2, 1, 3, 1)
    end

    @testset "combined split and merge" begin
        x = rand(1, 6, 2, 3)
        @test reshape(x, einops"1 (a b) ... -> a (b ...)", a=2) |> size == (2, 18)

        x = rand(12, 5)
        @test reshape(x, ((:a, :b), :c) --> (:a, (:b, :c)), a=3) |> size == (3, 20)
    end

    @testset "einops string macro" begin
        x = rand(2, 3, 4)
        @test reshape(x, einops"a b c -> (a b) c") == reshape(x, 6, 4)
        @test reshape(x, einops"a b c -> a (b c)") == reshape(x, 2, 12)
        @test reshape(x, einops"a b c -> (a b c)") == reshape(x, :)
    end

    @testset "empty arrays" begin
        x = rand(0, 3, 5)
        @test reshape(x, (:a, :b, :c) --> ((:a, :b), :c)) |> size == (0, 5)

        x = rand(2, 0, 5)
        @test reshape(x, (:a, :b, :c) --> (:a, (:b, :c))) |> size == (2, 0)
    end

    @testset "scalar and single element" begin
        x = fill(42.0)
        @test reshape(x, () --> ()) == x
        @test reshape(x, () --> (1,)) == reshape(x, 1)

        x = rand(1, 1, 1)
        @test reshape(x, (:a, :b, :c) --> ((:a, :b, :c),)) |> size == (1,)
    end
end

