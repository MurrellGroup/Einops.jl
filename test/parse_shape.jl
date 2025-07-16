using Einops
using Test

@testset "Shape Parsing" begin
    @testset "basic parsing" begin
        x = rand(2, 3, 5)
        @test parse_shape(x, (:a, :b, :c)) == (; a = 2, b = 3, c = 5)
        @test parse_shape(x, (:a, :b, -)) == (; a = 2, b = 3)
        @test parse_shape(x, (:a, -, -)) == (; a = 2)
        @test parse_shape(x, (-, -, -)) == (;)
    end

    @testset "ellipses parsing" begin
        x = rand(2, 3, 5)
        @test_logs (:warn, "not type stable") parse_shape(x, (:a, ..)) == (; a = 2)
        @test parse_shape(x, Val((:a, ..))) == (; a = 2)
        @test parse_shape(x, Val((:a, :b, ..))) == (; a = 2, b = 3)
        @test parse_shape(x, Val((:a, :b, :c, ..))) == (; a = 2, b = 3, c = 5)
    end

    @testset "type inference" begin
        x = rand(2, 3, 5)
        if VERSION >= v"1.10"
            h(x) = parse_shape(x, (:a, :b, :c))
            @test (@inferred h(x)) == (; a = 2, b = 3, c = 5)
        end

        @test (@inferred parse_shape(x, Val((:a, :b, :c)))) == (; a = 2, b = 3, c = 5)
        @test (@inferred parse_shape(x, Val((:a, :b, -)))) == (; a = 2, b = 3)
        @test (@inferred parse_shape(x, Val((:a, :b, ..)))) == (; a = 2, b = 3)
    end
end