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

    @testset "parse_shape" begin
        x = rand(2,3,5)
        @test parse_shape(x, (:a, :b, :c)) == (; a = 2, b = 3, c = 5)
        @test parse_shape(x, (:a, :b, -)) == (; a = 2, b = 3)
        @test parse_shape(x, (:a, -, -)) == (; a = 2)
        @test parse_shape(x, (-, -, -)) == (;)
    end

    @testset "einops string tokenization" begin

        @testset "parse_shape pattern" begin
            @test einops"a _ c" == (:a, -, :c)
            @test einops"_ _ _" == (-, -, -)
        end

        @testset "arrow pattern" begin
            @test einops"a b c -> a (c b)" == ((:a, :b, :c) --> (:a, (:c, :b)))
            @test einops"a b c -> a(c b)" == ((:a, :b, :c) --> (:a, (:c, :b)))
            @test einops"a b 1 -> a 1 b" == ((:a, :b, 1) --> (:a, 1, :b))
            @test einops"a b () -> a () b" == ((:a, :b, ()) --> (:a, (), :b))
            @test einops"a b()->a()b" == ((:a, :b, ()) --> (:a, (), :b))
            @test einops"b ... -> b ..." == ((:b, ..) --> (:b, ..))
            @test einops"b b -> a a" == ((:b, :b) --> (:a, :a))
            @test einops"->" == (() --> ())
            @test einops"-> 1" == (() --> (1,))
            @test_throws "'.'" Einops.parse_pattern("-> .")
            @test_throws "'('" Einops.parse_pattern("-> (")
            @test_throws "')'" Einops.parse_pattern("-> )")
        end

        @testset "pack and unpack pattern" begin
            @test einops"i j * k" == (:i, :j, *, :k)
            @test einops" i  j  *  k " == (:i, :j, *, :k)
            @test einops"* i" == (*, :i)
            @test einops"i *" == (:i, *)
            @test einops"i i" == (:i, :i)
        end

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
        @test rearrange(x, (:a, :b) --> (:b, 1, 1, :a, 1)) == reshape(permutedims(x, (2,1)), 3,1,1,2,1)
        @test rearrange(x, (:a, :b) --> (:b, (), :a)) == rearrange(x, (:a, :b) --> (:b, (), :a))

        x = rand(2,3,5)
        @test rearrange([x, x], (:a, :b, :c, :d) --> (:c, :b, :a, :d)) == permutedims(stack([x, x]), (3,2,1,4))
        @test rearrange(reshape([x, x], 1, 2), (:a, :b, :c, 1, :d) --> (:c, :b, :a, :d)) == permutedims(reshape(cat(x, x, dims=5), 2,3,5,2), (3,2,1,4))
        @test rearrange((x, x), (:a, :b, :c, :d) --> (:c, :b, :a, :d)) == permutedims(cat(x, x, dims=4), (3,2,1,4))

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

        x = rand(2,3,35)
        @test reduce(sum, x, einops"a b c -> b c") == dropdims(sum(x, dims=1), dims=1)
        @test reduce(sum, x, einops"a b (c c2) -> a c c2", c2=7) == reshape(sum(reshape(x, 2,3,5,7), dims=2), 2,5,7)
        @test reduce(sum, x, einops"a b (c c2) -> (a c) c2", c2=7) == reshape(sum(reshape(x, 2,3,5,7), dims=2), 2*5,7)
        @test reduce(sum, x, einops"a b (c c2) -> (c a) c2", c2=7) == reshape(permutedims(dropdims(sum(reshape(x, 2,3,5,7), dims=2), dims=2), (2,1,3)), 10,7)

        @test reduce(sum, [x, x], einops"a b c r -> a b c") == dropdims(sum(stack([x, x]), dims=4), dims=4)
        @test reduce(sum, reshape([x, x], 1, 2), einops"a b c 1 r -> a b c") == dropdims(sum(stack([x, x]), dims=4), dims=4)
        @test reduce(sum, (x, x), einops"a b c r -> a b c") == dropdims(sum(stack([x, x]), dims=4), dims=4)
        @test reduce(mean, [x, x], einops"a b c r -> a b c") == x
        @test reduce(maximum, reshape([x, x], 1, 2), einops"a b c 1 r -> a b c") == x
        @test reduce(minimum, (x, x), einops"a b c r -> a b c") == x

        # non-reducing:
        @test reduce(sum, x, einops"a b (c c2) -> a b c c2", c2=7) == reshape(x, 2,3,5,7)
        @test reduce(sum, x, einops"a b (c c2) -> a b c 1 c2", c2=7) == reshape(x, 2,3,5,1,7)

        @test_throws "right side" reduce(sum, x, einops"a b (c c2) -> a b d c c2", c2=7)
        @test_throws ["Left names", "not unique"] reduce(sum, x, einops"a a (c c2) -> a c c2", c2=7)
        @test_throws ["Right names", "not unique"] reduce(sum, x, einops"a b (c c2) -> a a c c2", c2=7)

        @testset "different operations" begin
            for (T, op) in zip(
                    [Float32, Float32, Float32, Float32, Float32, Bool, Bool],
                    [    sum,    prod, minimum, maximum,    mean,  any,  all])
                x = rand(T, 2,3,5)
                @test reduce(op, x, einops"a b c -> (b a)") == vec(permutedims(dropdims(op(x, dims=3), dims=3), (2,1)))
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
            @test reduce(maximum, x, einops"b c (h1 h2) (w1 w2) -> b c h1 w1", h2=2, w2=2) == reducedrop(max, reshape(x, 10,20,15,2,20,2), dims=(4,6))

            # same as previous, using anonymous axes,
            # note: only reduced axes can be anonymous
            @test_broken reduce(maximum, x, einops"b c (h1 2) (w1 2) -> b c h1 w1") == reducedrop(max, reshape(x, 10,20,15,2,20,2), dims=(4,6))

            # adaptive 2d max-pooling to 3 * 4 grid,
            # each element is max of 10x10 tile in the original tensor.
            @test reduce(maximum, x, einops"b c (h1 h2) (w1 w2) -> b c h1 w1", h1=3, w1=4) |> size == (10, 20, 3, 4)

            # Global average pooling
            @test reduce(mean, x, einops"b c h w -> b c") |> size == (10, 20)

            # subtracting mean over batch for each channel;
            # similar to x - np.mean(x, axis=(0, 2, 3), keepdims=True)
            @test x .- reduce(mean, x, einops"b c h w -> 1 c 1 1") == x .- mean(x, dims=(1,3,4))

            # Subtracting per-image mean for each channel
            @test x .- reduce(mean, x, einops"b c h w -> b c 1 1") == x .- mean(x, dims=(3,4))

            # same as previous, but using empty compositions
            @test x .- reduce(mean, x, einops"b c h w -> b c () ()") == x .- mean(x, dims=(3,4))
        end

    end

    @testset "repeat" begin

        x = rand(2,3)
        @test repeat(x, (:a, :b) --> (:a, :b, :r), r=2) == repeat(x, 1,1,2)
        @test repeat(x, (:a, :b) --> (:b, :a, :r), r=2) == repeat(permutedims(x, (2,1)), 1,1,2)
        @test repeat(x, (:a, :b) --> (:a, :b, 1, :r), r=2) == reshape(repeat(x, 1,1,2), 2,3,1,2)
        @test repeat(x, (:a, :b) --> (:a, (:b, :r)), r=2) == reshape(repeat(x, 1,1,2), 2,6)
        @test repeat(x, (:a, :b) --> (:a, (:b, :r), 1), r=2) == reshape(repeat(x, 1,1,2), 2,6,1)
        @test_broken repeat(x, (:a, :b) --> (:a, :b, 2)) == repeat(x, 1,1,2)

        @test repeat([x, x], einops"a b c -> a b c r", r=3) == repeat(x, 1,1,2,3)
        @test repeat(reshape([x, x], 1, 2), einops"a b 1 c -> a b c r", r=3) == repeat(x, 1,1,2,3)
        @test repeat((x, x), einops"a b c -> a b c r", r=3) == repeat(x, 1,1,2,3)

        x = rand(2,1,3)
        @test repeat(x, (:a, 1, :b) --> (:a, :b, :r), r=2) == repeat(reshape(x, 2,3), 1,1,2)
        @test repeat(x, (:a, 1, :b) --> (:a, :b, 1, :r), r=2) == reshape(repeat(reshape(x, 2,3), 1,1,2), 2,3,1,2)
        @test repeat(x, (:a, 1, :b) --> (:a, (:b, :r)), r=2) == reshape(repeat(reshape(x, 2,3), 1,1,2), 2,6)
        @test repeat(x, (:a, 1, :b) --> (:a, (:b, :r), 1), r=2) == reshape(repeat(reshape(x, 2,3), 1,1,2), 2,6,1)

        x = rand(2,3,35)
        @test repeat(x, (:a, :b, (:c, :c2)) --> (:a, (:b, :c), :c2, :r), c2=7, r=2) == reshape(repeat(reshape(x, 2,3,5,7), 1,1,1,1,2), 2,3*5,7,2)
        @test repeat(x, (:a, :b, (:c, :c2)) --> (:r, :c2, :a, (:c, :b)), c2=7, r=2) == reshape(repeat(reshape(permutedims(reshape(x, 2,3,5,7), (4,1,3,2)), 1,7,2,5,3), 2,1,1,1,1), 2,7,2,5*3)

        #x = rand(2,3,1,4)
        #@test repeat(x, einops"a b 1 (c c2) -> a (b c) c1 r", c1=1, r=2) == reshape(repeat(reshape(x, 2,3,4,1), 1,1,1,1,2), 2,12,1,2)

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

    end

    @testset "pack_unpack" begin

        @testset "Python API reference parity" begin
            # see https://einops.rocks/api/pack_unpack/

            inputs = [rand(2, 3, 5), rand(2, 3, 7, 5), rand(2, 3, 7, 9, 5)]
            @test begin
                packed, ps = pack(inputs, einops"i j * k")
                packed |> size == (2, 3, 71, 5) && ps == [(), (7,), (7, 9)]
            end

            @test begin
                packed, ps = pack(inputs, einops"i j * k")
                inputs_unpacked = unpack(packed, ps, einops"i j * k")
                all(inputs .== inputs_unpacked)
            end

        end

    end

    include("test_zygote.jl")

end
