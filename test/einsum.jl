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
        @test einsum(x, einops"i i ->")[] == tr(x)
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
end