using Einops
using Test, Statistics

@testset "Einops.jl" begin

    @testset "Pattern" begin
        @test (() --> ()) isa Einops.Pattern
        @test begin
            left, right = (:a, :b, :c) --> (:c, :b, :a)
            left isa Tuple && right isa Tuple
        end
        @test_throws "attempt to access" begin
            left, right, _ = (:a, :b, :c) --> (:c, :b, :a)
        end
        @test repr((:a, :b, :c) --> (:c, :b, :a)) == "(:a, :b, :c) --> (:c, :b, :a)"
    end

    @testset "einops string tokenization" begin
        @test einops"a b c -> a (c b)" == ((:a, :b, :c) --> (:a, (:c, :b)))
        @test einops"a b c -> a(c b)" == ((:a, :b, :c) --> (:a, (:c, :b)))
        @test einops"a b 1 -> a 1 b" == ((:a, :b, 1) --> (:a, 1, :b))
        @test einops"a b () -> a () b" == ((:a, :b, ()) --> (:a, (), :b))
        @test einops"a b()->a()b" == ((:a, :b, ()) --> (:a, (), :b))
        @test einops"b ... -> b ..." == ((:b, ..) --> (:b, ..))
        @test einops"->" == (() --> ())
        @test einops"-> 1" == (() --> (1,))
        @test_throws "'.'" Einops.parse_pattern("-> .")
        @test_throws "'('" Einops.parse_pattern("-> (")
        @test_throws "')'" Einops.parse_pattern("-> )")
        @test_throws "->" Einops.parse_pattern("")
    end

    @testset "rearrange" begin

        x = rand(2,3,5)
        @test rearrange(x, (:a, :b, :c) --> (:c, :b, :a)) == permutedims(x, (3,2,1))
        @test rearrange(x, (:a, :b, :c) --> (:a, (:c, :b))) == reshape(permutedims(x, (1,3,2)), 2,5*3)
        @test rearrange(x, (:first, :second, :third) --> (:third, :second, :first)) == rearrange(x, (:a, :b, :c) --> (:c, :b, :a))
        @test_throws "Input length" rearrange(x, (:a, (:b, :c)) --> (:c, :b, :a))
        @test_throws ["Set of", "does not match"] rearrange(x, (:a, :b, :c) --> (:a, :b, :a))
        @test_throws ["Left names", "not unique"] rearrange(x, (:a, :a, :b) --> (:a, :b))
        @test_throws ["Right names", "not unique"] rearrange(x, (:a, :b, :c) --> (:a, :b, :c, :a))
        @test_throws "Invalid input dimension" rearrange(x, (:a, :b, 'c') --> (:a, :b, :c))
        @test_broken rearrange(x, (:a, :b, ..) --> (:a, .., :b)) == rearrange(x, (:a, :b, :c) --> (:a, :c, :b))

        x = reshape(rand(1)) # size (), length 1
        @test rearrange(x, () --> ()) == x
        @test rearrange(x, () --> (1,)) == reshape(x, 1)

        x = rand(2,3,5*7)
        @test rearrange(x, (:a, :b, (:c, :d)) --> (:a, :d, (:c, :b)), c=5) == reshape(permutedims(reshape(x, 2,3,5,7), (1,4,3,2)), 2,7,5*3)

        x = rand(2,3,5*7*11)
        @test rearrange(x, (:a, :b, (:c, :d, :e)) --> ((:a, :e), :d, (:c, :b)), c=5, d=7) == reshape(permutedims(reshape(x, 2,3,5,7,11), (1,5,4,3,2)), 2*11,7,5*3)
        @test_throws "Unknown dimension sizes" rearrange(x, (:a, :b, (:c, :d, :e)) --> (:a, :b, :c, :d, :e), c=5)

        x = rand(2,1,3)
        @test rearrange(x, (:a, 1, :b) --> (:a, :b)) == dropdims(x, dims=2)
        @test_throws "Singleton dimension size is not 1" rearrange(x, (2, :a, :b) --> (:a, :b))
        @test_throws "Singleton dimension size is not 1" rearrange(x, (:a, :b, :c) --> (:a, :b, :c, 2))

        x = rand(2,3)
        @test rearrange(x, (:a, :b) --> (:b, 1, :a)) == reshape(permutedims(x, (2,1)), 3,1,2)
        @test rearrange(x, (:a, :b) --> (:b, 1, :a)) == rearrange(x, (:a, :b) --> (:b, (), :a))

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

    end

    @testset "reduce" begin

        @testset "Python API reference parity" begin
            # see https://einops.rocks/api/reduce/

            # utility function
            reducedrop(args...; dims) = dropdims(reduce(args...; dims); dims)

            x = randn(100, 32, 64)

            # perform max-reduction on the first axis
            # Axis t does not appear on RHS - thus we reduced over t
            @test_broken reduce(maximum, x, einops"t b c -> b c") == reducedrop(max, x, dims=1)

            # same as previous, but using verbose names for axes
            @test_broken reduce(maximum, x, einops"time batch channel -> batch channel") == reducedrop(max, x, dims=1)

            # let's pretend now that x is a batch of images
            # with 4 dims: batch=10, height=20, width=30, channel=40
            x = randn(10, 20, 30, 40)

            # 2d max-pooling with kernel size = 2 * 2 for image processing
            @test_broken reduce(maximum, x, einops"b c (h1 h2) (w1 w2) -> b c h1 w1", h2=2, w2=2) == reducedrop(max, reshape(x, 10, 20, 15, 2, 20, 2), dims=(4,6))

            # same as previous, using anonymous axes,
            # note: only reduced axes can be anonymous
            @test_broken reduce(maximum, x, einops"b c (h1 2) (w1 2) -> b c h1 w1") == reducedrop(max, reshape(x, 10, 20, 15, 2, 20, 2), dims=(4,6))

            # adaptive 2d max-pooling to 3 * 4 grid,
            # each element is max of 10x10 tile in the original tensor.
            @test_broken reduce(maximum, x, einops"b c (h1 h2) (w1 w2) -> b c h1 w2", h1=3, w1=4) |> size == (10, 20, 3, 4)

            # Global average pooling
            @test_broken reduce(mean, x, einops"b c h w -> b c") |> size == (10, 20)

            # subtracting mean over batch for each channel;
            # similar to x - np.mean(x, axis=(0, 2, 3), keepdims=True)
            @test_broken x .- reduce(mean, x, einops"b c h w -> 1 c 1 1") == x .- mean(x, dims=(1,3,4))

            # Subtracting per-image mean for each channel
            @test_broken x .- reduce(mean, x, einops"b c h w -> b c 1 1") == x .- mean(x, dims=(3,4))

            # same as previous, but using empty compositions
            @test_broken x .- reduce(mean, x, einops"b c h w -> b c () ()") == x .- mean(x, dims=(3,4))
        end

    end

    @testset "repeat" begin

        @testset "Python API reference parity" begin
            # see https://einops.rocks/api/repeat/

            # a grayscale image (of shape height x width)
            image = randn(30, 40)

            # change it to RGB format by repeating in each channel
            @test_broken repeat(image, einops"h w -> repeat h w", repeat=2) |> size == (30, 40, 3)

            # repeat image 2 times along height (vertical axis)
            @test_broken repeat(image, einops"h w -> (repeat h) w", repeat=2) |> size == (60, 40)

            # repeat image 2 time along height and 3 times along width
            @test_broken repeat(image, einops"h w -> (h2 h) (w3 w)", h2=2, w3=3) |> size == (60, 120)

            # convert each pixel to a small square 2x2, i.e. upsample an image by 2x
            @test_broken repeat(image, einops"h w -> (h h2) (w w2)", h2=2, w2=2) |> size == (60, 80)

            # 'pixelate' an image first by downsampling by 2x, then upsampling
            @test_broken begin
                downsampled = reduce(mean, image, einops"(h h2) (w w2) -> h w", h2=2, w2=2)
                repeat(downsampled, einops"h w -> (h h2) (w w2)", h2=2, w2=2) |> size == (60, 80)
            end

        end

    end

end
