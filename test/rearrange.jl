using Einops
using Test

@testset "Rearrange Operations" begin
    @testset "basic rearrangements" begin
        x = rand(2, 3, 5)
        @test rearrange(x, (:a, :b, :c) --> (:c, :b, :a)) == permutedims(x, (3, 2, 1))
        @test rearrange(x, (:a, :b, :c) --> (:a, (:c, :b))) == reshape(permutedims(x, (1, 3, 2)), 2, 5*3)
        @test rearrange(x, (:first, :second, :third) --> (:third, :second, :first)) == rearrange(x, (:a, :b, :c) --> (:c, :b, :a))
    end

    @testset "error handling" begin
        x = rand(2, 3, 5)
        @test_throws "Input length" rearrange(x, (:a, (:b, :c)) --> (:c, :b, :a))
        @test_throws ["Set of", "does not match"] rearrange(x, (:a, :b, :c) --> (:a, :b, :a))
        @test_throws ["Left names", "not unique"] rearrange(x, (:a, :a, :b) --> (:a, :b))
        @test_throws ["Right names", "not unique"] rearrange(x, (:a, :b, :c) --> (:a, :b, :c, :a))
        @test_throws "Invalid input dimension" rearrange(x, (:a, :b, (:c, 1)) --> (:a, :b, :c))
    end

    @testset "ellipses support" begin
        x = rand(2, 3, 5)
        @test rearrange(x, (:a, :b, ..) --> (:a, .., :b)) == rearrange(x, (:a, :b, :c) --> (:a, :c, :b))
        @test rearrange(x, (:a, :b, :c, ..) --> (:a, .., :b, :c)) == rearrange(x, (:a, :b, :c) --> (:a, :b, :c))
    end

    @testset "type inference" begin
        x = rand(2, 3, 5)
        @test (@inferred rearrange(x, (:a, :b, ..) --> (:a, .., :b))) == rearrange(x, (:a, :b, :c) --> (:a, :c, :b))
        @test (@inferred rearrange(x, ((:a, :a1), :b, ..) --> (:a, .., :b, :a1), a1=1)) == rearrange(x, (:a, :b, :c) --> (:a, :c, :b, 1))
    end

    @testset "scalar arrays" begin
        x = reshape(rand(1))  # size (), length 1
        @test rearrange(x, () --> ()) == x
        @test rearrange(x, () --> (1,)) == reshape(x, 1)
    end

    @testset "dimension decomposition" begin
        x = rand(2, 3, 5*7)
        @test rearrange(x, (:a, :b, (:c, :d)) --> (:a, :d, (:c, :b)), c=5) == reshape(permutedims(reshape(x, 2, 3, 5, 7), (1, 4, 3, 2)), 2, 7, 5*3)
        @test (@inferred rearrange(x, (:a, :b, (:c, :d)) --> (:a, :d, (:c, :b)), c=5)) == reshape(permutedims(reshape(x, 2, 3, 5, 7), (1, 4, 3, 2)), 2, 7, 5*3)

        x = rand(2, 3, 5*7*11)
        @test rearrange(x, (:a, :b, (:c, :d, :e)) --> ((:a, :e), :d, (:c, :b)), c=5, d=7) == reshape(permutedims(reshape(x, 2, 3, 5, 7, 11), (1, 5, 4, 3, 2)), 2*11, 7, 5*3)
        @test_throws "Unknown dimension sizes" rearrange(x, (:a, :b, (:c, :d, :e)) --> (:a, :b, :c, :d, :e), c=5)
    end

    @testset "singleton dimensions" begin
        x = rand(2, 1, 3)
        @test rearrange(x, (:a, 1, :b) --> (:a, :b)) == dropdims(x, dims=2)
        @test_throws "Singleton dimension size is not 1" rearrange(x, (2, :a, :b) --> (:a, :b))
        @test_throws "Singleton dimension size is not 1" rearrange(x, (:a, :b, :c) --> (:a, :b, :c, 2))

        x = rand(2, 3)
        @test rearrange(x, (:a, :b) --> (:b, 1, :a)) == reshape(permutedims(x, (2, 1)), 3, 1, 2)
        @test rearrange(x, (:a, :b) --> (:b, 1, 1, :a, 1)) == reshape(permutedims(x, (2, 1)), 3, 1, 1, 2, 1)
        @test rearrange(x, (:a, :b) --> (:b, (), :a)) == rearrange(x, (:a, :b) --> (:b, (), :a))
    end

    @testset "array collections" begin
        x = rand(2, 3, 5)
        @test rearrange([x, x], (:a, :b, :c, :d) --> (:c, :b, :a, :d)) == permutedims(stack([x, x]), (3, 2, 1, 4))
        @test rearrange(reshape([x, x], 1, 2), (:a, :b, :c, 1, :d) --> (:c, :b, :a, :d)) == permutedims(reshape(cat(x, x, dims=5), 2, 3, 5, 2), (3, 2, 1, 4))
        @test rearrange((x, x), (:a, :b, :c, :d) --> (:c, :b, :a, :d)) == permutedims(cat(x, x, dims=4), (3, 2, 1, 4))
    end

    @testset "Python API reference parity" begin
        # see https://einops.rocks/api/rearrange/

        # suppose we have a set of 32 images in "h w c" format (height-width-channel)
        images = randn(32, 30, 40, 3)

        # stack along first (batch) axis, output is a single array
        @test rearrange(images, einops"b h w c -> b h w c") |> size == (32, 30, 40, 3)

        # stacked and reordered axes to "b c h w" format
        @test rearrange(images, einops"b h w c -> b c h w") |> size == (32, 3, 30, 40)

        # concatenate images along height (vertical axis), 960 = 32 * 30
        @test rearrange(images, einops"b h w c -> (b h) w c") |> size == (960, 40, 3)

        # concatenated images along horizontal axis, 1280 = 32 * 40
        @test rearrange(images, einops"b h w c -> h (b w) c") |> size == (30, 1280, 3)

        # flattened each image into a vector, 3600 = 30 * 40 * 3
        @test rearrange(images, einops"b h w c -> b (c h w)") |> size == (32, 3600)

        # split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
        @test rearrange(images, einops"b (h1 h) (w1 w) c -> (b h1 w1) h w c", h1=2, w1=2) |> size == (128, 15, 20, 3)

        # space-to-depth operation
        @test rearrange(images, einops"b (h h1) (w w1) c -> b h w (c h1 w1)", h1=2, w1=2) |> size == (32, 15, 20, 12)
    end

    @testset "empty arrays" begin
        # Arrays with 0 dimensions
        x = rand(0, 3, 5)
        @test rearrange(x, (:a, :b, :c) --> (:c, :b, :a)) |> size == (5, 3, 0)
        @test rearrange(x, (:a, :b, :c) --> (:b, (:a, :c))) |> size == (3, 0)
        
        x = rand(2, 0, 5)
        @test rearrange(x, (:a, :b, :c) --> (:c, :b, :a)) |> size == (5, 0, 2)
        
        x = rand(2, 3, 0)
        @test rearrange(x, (:a, :b, :c) --> (:c, :b, :a)) |> size == (0, 3, 2)
        
        # Completely empty array
        x = rand(0, 0, 0)
        @test rearrange(x, (:a, :b, :c) --> (:c, :b, :a)) |> size == (0, 0, 0)
        
        # Empty with decomposition
        x = rand(0, 6)
        @test rearrange(x, (:a, (:b, :c)) --> (:c, :b, :a), b=2) |> size == (3, 2, 0)
        @test rearrange(x, (:a, (:b, :c)) --> (:c, :b, :a), b=3) |> size == (2, 3, 0)
    end

    @testset "single element arrays" begin
        x = rand(1)
        @test rearrange(x, (:a,) --> (:a,)) == x
        @test rearrange(x, (1,) --> ()) == fill(x[1])
        
        x = rand(1, 1, 1)
        @test rearrange(x, (:a, :b, :c) --> (:c, :b, :a)) == x
        @test rearrange(x, (1, 1, 1) --> ()) == fill(x[1])
    end

    @testset "all-ones dimensions" begin
        x = rand(1, 1, 1)
        @test rearrange(x, (:a, :b, :c) --> (:c, :b, :a)) |> size == (1, 1, 1)
        @test rearrange(x, (:a, :b, :c) --> ((:a, :b, :c),)) |> size == (1,)
    end

    @testset "invalid inputs" begin
        x = rand(2, 3, 4)
        
        # Mismatched dimensions
        @test_throws Exception rearrange(x, (:a, :b) --> (:b, :a))  # Too few axes
        @test_throws Exception rearrange(x, (:a, :b, :c, :d) --> (:d, :c, :b, :a))  # Too many axes
        
        # Invalid decomposition
        @test_throws Exception rearrange(x, (:a, :b, (:c, :d)) --> (:a, :b, :c, :d), c=3)  # 4 != 3*d
        @test_throws Exception rearrange(x, (:a, :b, (:c, :d)) --> (:a, :b, :c, :d), c=5, d=2)  # 4 != 5*2
        
        # Missing required dimensions
        @test_throws "Unknown dimension sizes" rearrange(x, (:a, :b, (:c, :d)) --> (:a, :b, :c, :d))  # No size for c or d
        
        # Non-numeric arrays - these should work!
        x = ["a" "b"; "c" "d"]
        @test rearrange(x, (:a, :b) --> (:b, :a)) == ["a" "c"; "b" "d"]
    end

    @testset "extreme dimension sizes" begin
        # Very small dimensions mixed with regular ones
        x = rand(1, 100, 1, 50, 1)
        @test rearrange(x, (1, :b, 1, :d, 1) --> (:b, :d)) |> size == (100, 50)
        
        # Large number of dimensions (10+)
        dims = (2, 1, 3, 1, 2, 3, 1, 2, 3, 1, 2)
        x = rand(dims...)
        pattern_left = (:a, :b, :c, :d, :e, :f, :g, :h, :i, :j, :k)
        pattern_right = (:k, :j, :i, :h, :g, :f, :e, :d, :c, :b, :a)
        @test rearrange(x, pattern_left --> pattern_right) |> size == reverse(dims)
    end
end