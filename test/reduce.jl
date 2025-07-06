using Einops
using Test, Statistics

@testset "Reduce Operations" begin
    @testset "basic reductions" begin
        x = rand(2, 3, 35)
        @test reduce(sum, x, einops"a b c -> b c") == dropdims(sum(x, dims=1), dims=1)
        @test reduce(sum, x, einops"a b (c c2) -> a c c2", c2=7) == reshape(sum(reshape(x, 2, 3, 5, 7), dims=2), 2, 5, 7)
        @test reduce(sum, x, einops"a b (c c2) -> (a c) c2", c2=7) == reshape(sum(reshape(x, 2, 3, 5, 7), dims=2), 2*5, 7)
        @test reduce(sum, x, einops"a b (c c2) -> (c a) c2", c2=7) == reshape(permutedims(dropdims(sum(reshape(x, 2, 3, 5, 7), dims=2), dims=2), (2, 1, 3)), 10, 7)
    end

    @testset "ellipses reductions" begin
        x = rand(2, 3, 35)
        @test reduce(sum, x, einops"a b ... -> b ...") == reduce(sum, x, einops"a b c -> b c")
        @test reduce(sum, x, einops"a b ... -> ... b") == reduce(sum, x, einops"a b c -> c b")
        @test reduce(sum, x, einops"a b ... -> b") == reduce(sum, x, einops"a b c -> b")
        @test reduce(sum, x, einops"a b ... -> ...") == reduce(sum, x, einops"a b c -> c")
        @test reduce(sum, x, einops"a b ... -> (a ...)") == reduce(sum, x, einops"a b c -> (a c)")
        @test reduce(sum, x, einops"a b ... -> (... b)") == reduce(sum, x, einops"a b c -> (c b)")
    end

    @testset "type inference" begin
        x = rand(2, 3, 35)
        @test (@inferred reduce(sum, x, einops"(a 2) b ... -> a (... b)")) == reduce(sum, x, einops"2 b c -> 1 (c b)")
    end

    @testset "array collections" begin
        x = rand(2, 3, 35)
        @test reduce(sum, [x, x], einops"a b c r -> a b c") == dropdims(sum(stack([x, x]), dims=4), dims=4)
        @test reduce(sum, reshape([x, x], 1, 2), einops"a b c 1 r -> a b c") == dropdims(sum(stack([x, x]), dims=4), dims=4)
        @test reduce(sum, (x, x), einops"a b c r -> a b c") == dropdims(sum(stack([x, x]), dims=4), dims=4)
        @test reduce(mean, [x, x], einops"a b c r -> a b c") == x
        @test reduce(maximum, reshape([x, x], 1, 2), einops"a b c 1 r -> a b c") == x
        @test reduce(minimum, (x, x), einops"a b c r -> a b c") == x
    end

    @testset "non-reducing operations" begin
        x = rand(2, 3, 35)
        @test reduce(sum, x, einops"a b (c c2) -> a b c c2", c2=7) == reshape(x, 2, 3, 5, 7)
        @test reduce(sum, x, einops"a b (c c2) -> a b c 1 c2", c2=7) == reshape(x, 2, 3, 5, 1, 7)
    end

    @testset "error handling" begin
        x = rand(2, 3, 35)
        @test_throws "right side" reduce(sum, x, einops"a b (c c2) -> a b d c c2", c2=7)
        @test_throws ["Left names", "not unique"] reduce(sum, x, einops"a a (c c2) -> a c c2", c2=7)
        @test_throws ["Right names", "not unique"] reduce(sum, x, einops"a b (c c2) -> a a c c2", c2=7)
    end

    @testset "different operations" begin
        for (T, op) in zip(
                [Float32, Float32, Float32, Float32, Float32, Bool, Bool],
                [    sum,    prod, minimum, maximum,    mean,  any,  all])
            x = rand(T, 2, 3, 5)
            @test reduce(op, x, einops"a b c -> (b a)") == vec(permutedims(dropdims(op(x, dims=3), dims=3), (2, 1)))
        end
    end

    @testset "Python API reference parity" begin
        # see https://einops.rocks/api/reduce/

        # utility function
        reducedrop(args...; dims) = dropdims(reduce(args...; dims); dims)

        x = randn(100, 32, 64)

        # perform max-reduction on the first axis
        # Axis t does not appear on RHS - thus we reduced over t
        @test reduce(maximum, x, einops"t b c -> b c") == reducedrop(max, x, dims=1)

        # same as previous, but using verbose names for axes
        @test reduce(maximum, x, einops"time batch channel -> batch channel") == reducedrop(max, x, dims=1)

        # let's pretend now that x is a batch of images
        # with 4 dims: batch=10, height=20, width=30, channel=40
        x = randn(10, 20, 30, 40)

        # 2d max-pooling with kernel size = 2 * 2 for image processing
        @test reduce(maximum, x, einops"b c (h1 h2) (w1 w2) -> b c h1 w1", h2=2, w2=2) == reducedrop(max, reshape(x, 10, 20, 15, 2, 20, 2), dims=(4, 6))

        # same as previous, using anonymous axes,
        # note: only reduced axes can be anonymous
        @test reduce(maximum, x, einops"b c (h1 2) (w1 2) -> b c h1 w1") == reducedrop(max, reshape(x, 10, 20, 15, 2, 20, 2), dims=(4, 6))
        @test reduce(maximum, x, einops"a b c (2 4 5) -> a b c") == reduce(maximum, x, einops"a b c d -> a b c")

        # adaptive 2d max-pooling to 3 * 4 grid,
        # each element is max of 10x10 tile in the original tensor.
        @test reduce(maximum, x, einops"b c (h1 h2) (w1 w2) -> b c h1 w1", h1=3, w1=4) |> size == (10, 20, 3, 4)

        # Global average pooling
        @test reduce(mean, x, einops"b c h w -> b c") |> size == (10, 20)

        # subtracting mean over batch for each channel;
        # similar to x - np.mean(x, axis=(0, 2, 3), keepdims=True)
        @test x .- reduce(mean, x, einops"b c h w -> 1 c 1 1") == x .- mean(x, dims=(1, 3, 4))

        # Subtracting per-image mean for each channel
        @test x .- reduce(mean, x, einops"b c h w -> b c 1 1") == x .- mean(x, dims=(3, 4))

        # same as previous, but using empty compositions
        @test x .- reduce(mean, x, einops"b c h w -> b c () ()") == x .- mean(x, dims=(3, 4))
    end

    @testset "empty arrays" begin
        x = rand(0, 3, 5)
        @test reduce(sum, x, (:a, :b, :c) --> (:b, :c)) |> size == (3, 5)
        @test reduce(sum, x, (:a, :b, :c) --> (:b, :c)) == zeros(3, 5)
        
        x = rand(2, 0, 5)
        @test reduce(sum, x, (:a, :b, :c) --> (:a, :c)) |> size == (2, 5)
        @test reduce(sum, x, (:a, :b, :c) --> (:a, :c)) == zeros(2, 5)
        
        x = rand(2, 3, 0)
        @test reduce(sum, x, (:a, :b, :c) --> (:a, :b)) |> size == (2, 3)
        @test reduce(sum, x, (:a, :b, :c) --> (:a, :b)) == zeros(2, 3)
        
        # Completely empty array
        x = rand(0, 0, 0)
        @test reduce(sum, x, (:a, :b, :c) --> (:b,)) |> size == (0,)
    end

    @testset "single element arrays" begin
        x = rand(1)
        @test reduce(sum, x, (:a,) --> ()) == fill(x[1])
        
        x = rand(1, 1, 1)
        @test reduce(sum, x, (:a, :b, :c) --> ()) == fill(x[1])
    end

    @testset "all-ones dimensions" begin
        x = rand(1, 1, 1)
        @test reduce(sum, x, (:a, :b, :c) --> (:a,)) |> size == (1,)
    end

    @testset "extreme dimension sizes" begin
        # Very small dimensions mixed with regular ones
        x = rand(1, 100, 1, 50, 1)
        # Can't drop dimensions in rearrange - use reduce instead
        @test reduce(sum, x, (:a, :b, :c, :d, :e) --> (:b, :d)) |> size == (100, 50)
    end

    @testset "no-op allocation optimization" begin
        x = rand(2, 3, 4)
        
        y = reduce(sum, x, (:a, :b, :c) --> (:a, :b, :c))
        @test pointer(x) == pointer(y)

        y = reduce(sum, x, (:a, :b, :c) --> ((:a, :b), :c))
        @test pointer(x) == pointer(y)
    end
end