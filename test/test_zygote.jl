using Zygote

@testset "Zygote" begin

    @testset "rearrange" begin
        x = rand(Float32, 2, 3, 35)
        function f(x)
            rearranged = rearrange(x, (:b, :h, (:w, :c)) --> (:b, (:c, :h), :w), w=5)
            return sum(rearranged)
        end
        @test gradient(f, x) isa Tuple{Array{Float32,3}}
    end

    @testset "reduce" begin
        x = rand(Float32, 2, 3, 35)
        function f(op, x)
            rearranged = reduce(op, x, (:b, :h, (:w, :c)) --> (:b, (:h, :w)), w=5)
            return sum(rearranged)
        end
        @testset for op in [sum, prod, maximum, minimum, mean]
            @test gradient(f, op, x) isa Tuple{Nothing, Array{Float32,3}}
        end
    end

    @testset "repeat" begin
        x = rand(Float32, 2, 15)
        function f(x)
            repeated = repeat(x, (:b, (:h, :w)) --> (:b, :h, (:w, :c)), h=3, c=7)
            return sum(repeated)
        end
        @test gradient(f, x) isa Tuple{Array{Float32,2}}
    end
end
