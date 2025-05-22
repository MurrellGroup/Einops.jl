using Test
using Einops
using Statistics
using Reactant

@testset "Reactant compilation tests" begin
    # Test data setup
    x2d = Reactant.to_rarray(reshape(1.0:12.0, 3, 4))
    x3d = Reactant.to_rarray(reshape(1.0:24.0, 2, 3, 4))
    x4d = Reactant.to_rarray(reshape(1.0:48.0, 2, 3, 4, 2))
    
    @testset "rearrange compilation" begin
        # Basic transpose
        result1 = @jit rearrange(x2d, (:a, :b) --> (:b, :a))
        @test size(result1) == (4, 3)
        
        # 3D permutation
        result2 = @jit rearrange(x3d, (:a, :b, :c) --> (:c, :a, :b))
        @test size(result2) == (4, 2, 3)
        
        # Reshape with merge
        result3 = @jit rearrange(x3d, (:a, :b, :c) --> ((:a, :b), :c))
        @test size(result3) == (6, 4)
        
        # Reshape with split
        result4 = @jit rearrange(x2d, ((:a, :b), :c) --> (:a, :b, :c), a=3)
        @test size(result4) == (3, 1, 4)
        
        # Multiple operations in one function
        function complex_rearrange(x)
            y = rearrange(x, (:a, :b, :c) --> (:c, :b, :a))
            return rearrange(y, (:c, :b, :a) --> ((:c, :b), :a))
        end
        result5 = @jit complex_rearrange(x3d)
        @test size(result5) == (12, 2)
    end
    
    @testset "reduce compilation" begin
        # Sum reduction
        result1 = @jit reduce(sum, x3d, (:a, :b, :c) --> (:a, :c))
        @test size(result1) == (2, 4)
        
        # Mean reduction
        result2 = @jit reduce(mean, x3d, (:a, :b, :c) --> (:b,))
        @test size(result2) == (3,)
        
        # Max reduction
        result3 = @jit reduce(maximum, x2d, (:a, :b) --> ())
        @test result3[] isa Float64
        
        # Multiple reductions
        function multi_reduce(x)
            s = reduce(sum, x, (:a, :b, :c) --> (:a,))
            return reduce(mean, s, (:a,) --> ())
        end
        result4 = @jit multi_reduce(x3d)
        @test result4[] isa Float64
    end
    
    @testset "repeat compilation" begin
        # Basic repeat
        result1 = @jit repeat(x2d, (:a, :b) --> (:a, :b, :c), c=2)
        @test size(result1) == (3, 4, 2)
        
        # Repeat multiple axes
        result2 = @jit repeat(x2d, (:a, :b) --> (:a, :c, :b, :d), c=2, d=3)
        @test size(result2) == (3, 2, 4, 3)
        
        # Repeat with rearrange
        function repeat_rearrange(x)
            y = repeat(x, (:a, :b) --> (:a, :b, :c), c=2)
            return rearrange(y, (:a, :b, :c) --> (:c, :a, :b))
        end
        result3 = @jit repeat_rearrange(x2d)
        @test size(result3) == (2, 3, 4)
    end
    
    @testset "mixed operations compilation" begin
        # Complex workflow combining multiple operations
        function einops_workflow(x)
            # Rearrange -> reduce -> repeat -> rearrange
            y1 = rearrange(x, (:a, :b, :c) --> (:c, :a, :b))
            y2 = reduce(sum, y1, (:a, :b, :c) --> (:a, :c))
            y3 = repeat(y2, (:a, :b) --> (:a, :b, :c), c=2)
            return rearrange(y3, (:a, :b, :c) --> ((:a, :c), :b))
        end
        result = @jit einops_workflow(x3d)
        @test size(result) == (8, 3)
    end
end
