using Zygote

@testset "Zygote" begin

    @testset "rearrange" begin
        x = rand(Float32, 2, 3, 35)
        function f(x)
            rearranged = rearrange(x, (:b, :h, (:w, :c)) --> (:b, (:c, :h), :w), w=5)
            return sum(rearranged)
        end
        @test gradient(f, x) isa Tuple{AbstractArray{Float32,3}}
    end

    @testset "reduce" begin
        x = rand(Float32, 2, 3, 35)
        function f(op, x)
            rearranged = reduce(op, x, (:b, .., (:w, :c)) --> (:b, (.., :w)), w=5)
            return sum(rearranged)
        end
        @testset for op in [sum, prod, maximum, minimum, mean]
            @test gradient(f, op, x) isa Tuple{Nothing, AbstractArray{Float32,3}}
        end
    end

    @testset "repeat" begin
        x = rand(Float32, 2, 15)
        function f(x)
            repeated = repeat(x, (:b, (:h, :w)) --> (:b, :h, (:w, :c)), h=3, c=7)
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
            result = einsum(a, b, ((:i, :j), (:j, :k)) --> (:i, :k))
            return sum(result)
        end
        @test gradient(f1, a, b) isa Tuple{AbstractArray{Float32,2}, AbstractArray{Float32,2}}
        
        # Test einsum with batched tensors
        x = rand(Float32, 2, 3, 5)
        y = rand(Float32, 3, 4, 5)
        function f2(x, y)
            # Batched matrix multiplication
            result = einsum(x, y, ((:i, :j, :b), (:j, :k, :b)) --> (:i, :k, :b))
            return sum(result)
        end
        @test gradient(f2, x, y) isa Tuple{AbstractArray{Float32,3}, AbstractArray{Float32,3}}
        
        # Test contraction of multiple indices
        c = rand(Float32, 3, 4, 5)
        d = rand(Float32, 3, 5, 6)
        function f3(c, d)
            # Contract first and second indices
            result = einsum(c, d, ((:i, :j, :k), (:i, :k, :l)) --> (:j, :l))
            return sum(result)
        end
        @test gradient(f3, c, d) isa Tuple{AbstractArray{Float32,3}, AbstractArray{Float32,3}}
    end

end
