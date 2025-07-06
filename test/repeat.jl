using Einops
using Test, Statistics

@testset "Repeat Operations" begin
    @testset "basic repetitions" begin
        x = rand(2, 3)
        @test repeat(x, (:a, :b) --> (:a, :b, :r), r=2) == repeat(x, 1, 1, 2)
        @test repeat(x, (:a, :b) --> (:b, :a, :r), r=2) == repeat(permutedims(x, (2, 1)), 1, 1, 2)
        @test repeat(x, (:a, :b) --> (:a, :b, 1, :r), r=2) == reshape(repeat(x, 1, 1, 2), 2, 3, 1, 2)
        @test repeat(x, (:a, :b) --> (:a, (:b, :r)), r=2) == reshape(repeat(x, 1, 1, 2), 2, 6)
        @test repeat(x, (:a, :b) --> (:a, (:b, :r), 1), r=2) == reshape(repeat(x, 1, 1, 2), 2, 6, 1)
        @test repeat(x, (:a, :b) --> (:a, :b, 2)) == repeat(x, 1, 1, 2)
    end

    @testset "type inference" begin
        x = rand(2, 3)
        @test (@inferred repeat(x, (:a, :b) --> (:a, :b, 2))) == repeat(x, 1, 1, 2)
    end

    @testset "array collections" begin
        x = rand(2, 3)
        @test repeat([x, x], einops"a b c -> a b c r", r=3) == repeat(x, 1, 1, 2, 3)
        @test repeat(reshape([x, x], 1, 2), einops"a b 1 c -> a b c r", r=3) == repeat(x, 1, 1, 2, 3)
        @test repeat((x, x), einops"a b c -> a b c r", r=3) == repeat(x, 1, 1, 2, 3)
    end

    @testset "singleton dimensions" begin
        x = rand(2, 1, 3)
        @test repeat(x, (:a, 1, :b) --> (:a, :b, :r), r=2) == repeat(reshape(x, 2, 3), 1, 1, 2)
        @test repeat(x, (:a, 1, :b) --> (:a, :b, 1, :r), r=2) == reshape(repeat(reshape(x, 2, 3), 1, 1, 2), 2, 3, 1, 2)
        @test repeat(x, (:a, 1, :b) --> (:a, (:b, :r)), r=2) == reshape(repeat(reshape(x, 2, 3), 1, 1, 2), 2, 6)
        @test repeat(x, (:a, 1, :b) --> (:a, (:b, :r), 1), r=2) == reshape(repeat(reshape(x, 2, 3), 1, 1, 2), 2, 6, 1)
    end

    @testset "ellipses support" begin
        x = rand(2, 1, 3)
        @test repeat(x, einops"a ... -> a (... r)", r=2) == repeat(x, einops"a b c -> a (b c r)", r=2)
        @test repeat(x, einops"a b ... -> a (b ... r)", r=2) == repeat(x, einops"a b c -> a (b c r)", r=2)
        @test (@inferred repeat(x, einops"a b ... -> a (b ... r)", r=2)) == repeat(x, einops"a b c -> a (b c r)", r=2)
    end

    @testset "complex decompositions" begin
        x = rand(2, 3, 35)
        @test repeat(x, (:a, :b, (:c, :c2)) --> (:a, (:b, :c), :c2, :r), c2=7, r=2) == reshape(repeat(reshape(x, 2, 3, 5, 7), 1, 1, 1, 1, 2), 2, 3*5, 7, 2)
        @test repeat(x, (:a, :b, (:c, :c2)) --> (:r, :c2, :a, (:c, :b)), c2=7, r=2) == reshape(repeat(reshape(permutedims(reshape(x, 2, 3, 5, 7), (4, 1, 3, 2)), 1, 7, 2, 5, 3), 2, 1, 1, 1, 1), 2, 7, 2, 5*3)
    end

    @testset "Python API reference parity" begin
        # see https://einops.rocks/api/repeat/

        # a grayscale image (of shape height x width)
        image = randn(30, 40)

        # change it to RGB format by repeating in each channel
        @test repeat(image, einops"h w -> h w c", c=3) |> size == (30, 40, 3)

        # repeat image 2 times along height (vertical axis)
        @test repeat(image, einops"h w -> (repeat h) w", repeat=2) |> size == (60, 40)

        # repeat image 2 time along height and 3 times along width
        @test repeat(image, einops"h w -> (h2 h) (w3 w)", h2=2, w3=3) |> size == (60, 120)

        # convert each pixel to a small square 2x2, i.e. upsample an image by 2x
        @test repeat(image, einops"h w -> (h h2) (w w2)", h2=2, w2=2) |> size == (60, 80)

        # 'pixelate' an image first by downsampling by 2x, then upsampling
        @test repeat(
            reduce(mean, image, einops"(h h2) (w w2) -> h w", h2=2, w2=2),
            einops"h w -> (h h2) (w w2)", h2=2, w2=2
        ) |> size == (30, 40)
    end

    @testset "empty arrays" begin
        # Completely empty array
        x = rand(0, 0, 0)
        @test repeat(x, (:a, :b, :c) --> (:a, :b, :c, :d), d=4) |> size == (0, 0, 0, 4)
    end

    @testset "single element arrays" begin
        x = rand(1)
        @test repeat(x, (:a,) --> (:a, :b), b=3) |> size == (1, 3)
    end

    @testset "all-ones dimensions" begin
        x = rand(1, 1, 1)
        @test repeat(x, (:a, :b, :c) --> (:a, :b, :c, :d, :e), d=2, e=3) |> size == (1, 1, 1, 2, 3)
    end

    @testset "array types" begin
        # Non-numeric arrays - these should work!
        x = ["a" "b"; "c" "d"]
        @test repeat(x, (:a, :b) --> (:a, :b, :c), c=2) == cat(x, x, dims=3)
    end

    @testset "no-op allocation optimization" begin
        x = rand(2, 3, 4)

        y = repeat(x, (:a, :b, :c) --> (:a, :b, :c, 1))
        @test pointer(x) == pointer(y)

        y = repeat(x, (:a, :b, :c) --> ((:a, :b), :c, 1))
        @test pointer(x) == pointer(y)
    end
end