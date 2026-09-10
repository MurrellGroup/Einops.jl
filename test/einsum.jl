using Einops
using Test, LinearAlgebra

@testset "Einsum Operations" begin
    @testset "basic einsum" begin
        a = rand(2, 3, 5)
        b = rand(3, 4, 5)
        @test einsum(a, b, ((:i, :j, :b), (:j, :k, :b)) --> (:i, :k, :b)) == stack([a * b for (a, b) in zip(eachslice(a, dims=3), eachslice(b, dims=3))])
    end

    @testset "trace operations" begin
        x = rand(4, 4)
        @test einsum(x, einops"i i ->")[] ≈ tr(x)
        @test_broken (@inferred einsum(x, einops"i i ->"))[] == tr(x)
    end

    @testset "Python API reference parity" begin
        # see https://einops.rocks/api/einsum/

        # Filter a set of images:
        batched_images = randn(128, 16, 16)
        filters = randn(16, 16, 30)
        @test einsum(batched_images, filters, einops"batch h w, h w channel -> batch channel") |> size == (128, 30)

        # Matrix multiplication, with an unknown input shape:
        batch_shape = (50, 30)
        data = randn(batch_shape..., 20)
        weights = randn(10, 20)
        @test einsum(weights, data, einops"out_dim in_dim, ... in_dim -> ... out_dim") |> size == (50, 30, 10)

        # Matrix trace on a single tensor:
        matrix = randn(10, 10)
        @test einsum(matrix, einops"i i ->") |> size == ()
    end
end

@testset "Einsum - nested, ellipses, singletons" begin

    @testset "vwn" begin
        x = rand(6, 7)
        y = rand(3, 5)
        @test einsum(x, y, einops"(d n) ..., n m -> (d m) ...") ==
              reshape(einsum(reshape(x, 2, 3, 7), y, einops"d n ..., n m -> d m ..."), 10, 7)
    end

    @testset "nested group on right" begin
        a = rand(2, 3)
        b = rand(3, 4)
        y = einsum(a, b, einops"i j, j (k k2) -> i k k2", k2=2)
        @test size(y) == (2, 2, 2)
        @test y ≈ reshape(a * b, 2, 2, 2)
    end

    @testset "nested group on left (both arrays)" begin
        a = rand(2, 4)
        b = rand(4, 5)
        y = einsum(a, b, einops"i (k k2), (k k2) o -> i o", k2=2)
        @test size(y) == (2, 5)
        @test y ≈ a * b
    end

    @testset "ellipses with singleton on right" begin
        x = rand(2, 3, 5)
        y = rand(3, 4, 1, 5)
        z = einsum(x, y, einops"i j ..., j (k k2) 1 ... -> i k 1 (k2 ...)", k2=2)
        @test size(z) == (2, 2, 1, 10)
    end

    @testset "singleton on left" begin
        x = rand(2, 1, 3)
        y = einsum(x, einops"i 1 j -> i j")
        @test size(y) == (2, 3)
        @test y ≈ reshape(x, 2, 3)
    end

    @testset "contract on host arrays" begin
        x, y, z = rand(2, 3), rand(3, 4), rand(4, 5)
        @test Einops.contract(((:i, :j), (:j, :k)), (:i, :k), x, y) ≈ x * y
        @test Einops.contract(((:i, :j), (:j, :k), (:k, :l)), (:i, :l), x, y, z) ≈ x * y * z
        @test einsum(x, y, einops"i j, j k -> i k") == Einops.contract(((:i, :j), (:j, :k)), (:i, :k), x, y)
        @test einsum(x, y, z, einops"i j, j k, k l -> i l"; optimizer=Einops.OMEinsum.GreedyMethod()) ≈ x * y * z
    end

    @testset "single operand reduces without OMEinsum" begin
        x = rand(3, 4, 5)
        @test einsum(x, einops"i j k -> k i") == reduce(sum, x, einops"i j k -> k i")
        @test einsum(x, einops"i j k -> k i") ≈ Einops.contract(((:i, :j, :k),), (:k, :i), x)
        @test einsum(x, einops"(a b) j k -> b k"; a=3) == reduce(sum, x, einops"(a b) j k -> b k"; a=3)
        @test einsum(x, einops"i ... -> ...") == reduce(sum, x, einops"i ... -> ...")
        x4 = reshape(x, 3, 1, 4, 5)
        @test einsum(x4, einops"i 1 j k -> i j") == reduce(sum, x4, einops"i 1 j k -> i j")
        m = rand(4, 4)
        @test only(einsum(m, einops"i i -> ")) ≈ tr(m)      # repeated labels still go to OMEinsum
    end

    @testset "explicit single operand and scalar operands" begin
        x = rand(2, 3)
        @test einsum(x, (:i, :j) --> (:i,)) == vec(sum(x; dims=2))
        @test einsum(x, (:i, ..) --> (:i,)) == vec(sum(x; dims=2))
        @test einsum(x, ((:a, :b), :j) --> (:b,); a=2) ==
              reduce(sum, x, einops"(a b) j -> b"; a=2)
        scalar = fill(3.0)
        @test einsum(scalar, () --> ()) == scalar
        @test_throws ArgumentError einsum(scalar, ((),) --> ())
        @test einops"a, -> a" === (((:a,), ()) --> (:a,))
        @test einops"a, a, -> a" === (((:a,), (:a,), ()) --> (:a,))
        v = [1.0, 2.0]
        @test einsum(v, scalar, einops"a, -> a") == 3 .* v
        @test einsum(scalar, v, einops", a -> a") == 3 .* v
        @test einsum(v, v, scalar, einops"a, a, -> a") == 3 .* v .* v
        @test einsum(scalar, scalar, einops", ->") == fill(9.0)
        @test_throws ArgumentError einsum(v, einops"a, -> a")
        @test einsum(x, x, ((:i, :j), (:i, :j)) --> (:i,)) ≈ vec(sum(x .* x; dims=2))
        @test_throws ArgumentError einsum(x, x, (:i, :j) --> (:i,))
        @test_throws ArgumentError einsum(v, v, scalar, einops"a, a -> a")
    end

    @testset "arity distinguishes groups from operand lists" begin
        x = collect(1.0:6.0)
        expected = reshape(x, 2, 3)
        @test einsum(x, einops"(a b) -> a b"; a=2) == expected
        @test Einops.@einsum(x, "(a b) -> a b"; a=2) == expected
        @test einsum(expected, einops"a b -> a b") == expected
        @test einsum(x, ((:a, :b),) --> (:a, :b); a=2) == expected
        @test einsum(x, Einops.ArrowPattern{((:a, :b),), (:a, :b)}(); a=2) == expected
        @test_throws ArgumentError einsum(expected, ((:a, :b),) --> (:a, :b))
        @test einsum(reshape(x, 2, 3), einops"(a b) (c d) -> a b c d"; a=2, c=3) == reshape(x, 2, 1, 3, 1)
        @test einsum(x, fill(2.0), einops"(a b), -> a b"; a=2) == 2 .* expected
        # The same left tuple describes two grouped axes or two operands by arity.
        pattern = ((:i,), (:j,)) --> (:i, :j)
        @test einsum(expected, pattern) == expected
        @test einsum([1.0, 2.0], [3.0, 4.0, 5.0], pattern) == [1.0, 2.0] * [3.0, 4.0, 5.0]'
        # The single-input contraction fallback must also retain its grouped axis.
        diagonal = collect(1.0:9.0)
        @test only(einsum(diagonal, einops"(i i) ->"; i=3)) == tr(reshape(diagonal, 3, 3))
    end

end
