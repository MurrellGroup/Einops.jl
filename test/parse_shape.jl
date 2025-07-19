using Einops
using Test

@testset "Shape Parsing" begin
    @testset "basic parsing" begin
        x = rand(2, 3, 5)
        @test parse_shape(x, Val((:a, :b, :c))) == (; a = 2, b = 3, c = 5)
        @test parse_shape(x, Val((:a, :b, -))) == (; a = 2, b = 3)
        @test parse_shape(x, Val((:a, -, -))) == (; a = 2)
        @test parse_shape(x, Val((-, -, -))) == (;)
        @test_throws ErrorException parse_shape(x, Val((:a, :b, :a)))

        x = rand(2, 3, 2)
        @test parse_shape(x, Val((:a, :b, :a))) == (; a = 2, b = 3)
    end

    @testset "ellipses parsing" begin
        x = rand(2, 3, 5)
        @test parse_shape(x, Val((:a, ..))) == (; a = 2)
        @test parse_shape(x, Val((:a, :b, ..))) == (; a = 2, b = 3)
        @test parse_shape(x, Val((:a, .., :c))) == (; a = 2, c = 5)
        @test parse_shape(x, Val((:a, :b, :c, ..))) == (; a = 2, b = 3, c = 5)
    end

    @testset "type inference" begin
        x = rand(2, 3, 5)
        @test (@inferred parse_shape(x, Val((:a, :b, :c)))) == (; a = 2, b = 3, c = 5)
        @test (@inferred parse_shape(x, Val((:a, :b, -)))) == (; a = 2, b = 3)
        @test (@inferred parse_shape(x, Val((:a, :b, ..)))) == (; a = 2, b = 3)
    end
end