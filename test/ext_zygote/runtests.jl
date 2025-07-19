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

    @testset "parse_shape" begin
        @test gradient(rand(Float32, 3, 4)) do x
            shape = parse_shape(x, Val((:a, :b)))
            rearrange(x, einops"a (b c) -> b c a"; shape...) |> sum
        end isa Tuple{AbstractArray{Float32,2}}
    end

end
