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