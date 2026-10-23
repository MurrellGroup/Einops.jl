using Zygote

@testset "Zygote" begin

    @testset "rearrange" begin
        x = rand(Float32, 2, 3, 35)
        function f(x)
            rearranged = rearrange(x, einops"b h (w c) -> b (c h) w", w=5)
            return sum(rearranged)
        end
        @test gradient(f, x) isa Tuple{AbstractArray{Float32,3}}
    end

    @testset "reduce" begin
        x = rand(Float32, 2, 3, 35)
        function f(op, x)
            rearranged = reduce(op, x, einops"b ... (w c) -> b (... w)", w=5)
            return sum(rearranged)
        end
        @testset for op in [sum, prod, maximum, minimum, mean]
            @test gradient(f, op, x) isa Tuple{Nothing, AbstractArray{Float32,3}}
        end
    end

    @testset "repeat" begin
        x = rand(Float32, 2, 15)
        function f(x)
            repeated = repeat(x, einops"b (h w) -> b h (w c)", h=3, c=7)
            return sum(repeated)
        end
        @test gradient(f, x) isa Tuple{AbstractArray{Float32,2}}
    end

    @testset "einsum" begin
        # Test matrix multiplication with einsum
        a = rand(Float32, 3, 4)
        b = rand(Float32, 4, 5)
        function f1(a, b)
            # Matrix multiplication (i,j),(j,k)->(i,k)
            result = einsum(a, b, einops"i j, j k -> i k")
            return sum(result)
        end
        @test gradient(f1, a, b) isa Tuple{AbstractArray{Float32,2}, AbstractArray{Float32,2}}
        
        # Test einsum with batched tensors
        x = rand(Float32, 2, 3, 5)
        y = rand(Float32, 3, 4, 5)
        function f2(x, y)
            # Batched matrix multiplication
            result = einsum(x, y, einops"i j b, j k b -> i k b")
            return sum(result)
        end
        @test gradient(f2, x, y) isa Tuple{AbstractArray{Float32,3}, AbstractArray{Float32,3}}
        
        # Test contraction of multiple indices
        c = rand(Float32, 3, 4, 5)
        d = rand(Float32, 3, 5, 6)
        function f3(c, d)
            # Contract first and second indices
            result = einsum(c, d, einops"i j k, i k l -> j l")
            return sum(result)
        end
        @test gradient(f3, c, d) isa Tuple{AbstractArray{Float32,3}, AbstractArray{Float32,3}}

        # Three operands contract pairwise, and the gradient follows the pairs.
        x3, y3, z3 = rand(2, 3), rand(3, 4), rand(4, 5)
        f4(x, y, z) = sum(einsum(x, y, z, einops"i j, j k, k l -> i l"))
        gx, gy, gz = gradient(f4, x3, y3, z3)
        @test gx ≈ ones(2, 5) * (y3 * z3)'
        @test gy ≈ x3' * ones(2, 5) * z3'
        @test gz ≈ (x3 * y3)' * ones(2, 5)
    end

    @testset "reshape with keywords (#51)" begin
        x = rand(Float32, 6, 4, 3)
        f(x) = sum(abs2, reshape(x, einops"(a b) c ... -> a b (c ...)"; a=2))
        @test only(gradient(f, x)) ≈ 2 .* x
        # the macro's call fallback, which a `; kws...` splat takes
        g(x) = sum(abs2, @reshape(x, "(a b) c ... -> a b (c ...)"; (; a=2)...))
        @test only(gradient(g, x)) ≈ 2 .* x
        h(x) = sum(abs2, reshape(x, einops"a b c -> (a b) c"))
        @test only(gradient(h, x)) ≈ 2 .* x
    end

    @testset "checked and interpolated lengths" begin
        x = rand(Float32, 6, 4)
        k = 2
        f(x) = sum(abs2, reduce(sum, x, einops"($k b) c -> b c"))
        g(x) = sum(abs2, @rearrange(x, "(a b) c -> b c a", a=2))
        @test only(gradient(f, x)) ≈ only(gradient(x -> sum(abs2, reduce(sum, x, einops"(a b) c -> b c"; a=2)), x))
        @test only(gradient(g, x)) ≈ 2 .* x
    end

end
