using Einops
using Test
using SparseArrays
using LinearAlgebra
using Statistics

@testset "Robustness and Edge Cases" begin
    @testset "diverse numeric types" begin
        # Integer types
        for T in [Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64]
            x = rand(T, 2, 3, 4)
            @test rearrange(x, (:a, :b, :c) --> (:c, :b, :a)) |> eltype == T
            # Note: reduce with sum may promote types (e.g., Int8 -> Int64)
            # This is expected Julia behavior
            @test repeat(x, (:a, :b, :c) --> (:a, :b, :c, :d), d=2) |> eltype == T
        end
        
        # Complex numbers
        x = rand(ComplexF32, 2, 3, 4)
        @test rearrange(x, (:a, :b, :c) --> (:c, :b, :a)) |> eltype == ComplexF32
        @test reduce(sum, x, (:a, :b, :c) --> (:b, :c)) |> eltype == ComplexF32
        @test repeat(x, (:a, :b, :c) --> (:a, :b, :c, :d), d=2) |> eltype == ComplexF32
        
        x = rand(ComplexF64, 2, 3, 4)
        @test rearrange(x, (:a, :b, :c) --> (:c, :b, :a)) isa Array{ComplexF64}
        @test reduce(mean, x, (:a, :b, :c) --> (:a,)) isa Array{ComplexF64}
        
        # Rational numbers
        x = [Rational(rand(1:10), rand(1:10)) for _ in 1:2, _ in 1:3, _ in 1:4]
        @test rearrange(x, (:a, :b, :c) --> (:c, :b, :a)) |> eltype == Rational{Int}
        @test reduce(sum, x, (:a, :b, :c) --> (:b, :c)) |> eltype == Rational{Int}
        
        # BitArrays
        x = rand(Bool, 2, 3, 4)
        @test rearrange(x, (:a, :b, :c) --> (:c, :b, :a)) isa Array{Bool}  # rearrange returns Array, not BitArray
        @test reduce(any, x, (:a, :b, :c) --> (:b, :c)) isa Array{Bool}
        @test reduce(all, x, (:a, :b, :c) --> (:a,)) isa Array{Bool}
    end

    @testset "sparse arrays" begin
        x = sparse([1, 2, 3, 1], [1, 2, 3, 3], [1.0, 2.0, 3.0, 4.0], 3, 3)
        x3d = reshape(x, 3, 3, 1)
        
        # Basic operations may not preserve sparsity after reshape
        # rearrange returns a dense array when it involves reshape
        @test rearrange(x3d, (:a, :b, :c) --> (:b, :a, :c)) isa Array
        @test rearrange(x3d, (:a, :b, :c) --> (:c, :b, :a)) |> size == (1, 3, 3)
        
        # Reduce operations
        @test reduce(sum, x3d, (:a, :b, :c) --> (:b, :c)) |> size == (3, 1)
        @test reduce(sum, x3d, (:a, :b, :c) --> (:a,)) |> size == (3,)
        
        # Repeat operations
        @test repeat(x3d, (:a, :b, :c) --> (:a, :b, :c, :d), d=2) |> size == (3, 3, 1, 2)
    end

    @testset "views and subarrays" begin
        x = rand(4, 6, 8)
        
        # View of the array
        v = @view x[2:3, :, 1:2:8]
        @test rearrange(v, (:a, :b, :c) --> (:c, :b, :a)) |> size == (4, 6, 2)
        @test reduce(sum, v, (:a, :b, :c) --> (:b,)) |> size == (6,)
        @test repeat(v, (:a, :b, :c) --> (:a, :b, :c, :d), d=3) |> size == (2, 6, 4, 3)
        
        # ReshapedArray
        r = reshape(x, 8, 3, 8)
        @test rearrange(r, (:a, :b, :c) --> (:c, :b, :a)) |> size == (8, 3, 8)
        
        # PermutedDimsArray
        p = permutedims(x, (3, 2, 1))
        @test rearrange(p, (:a, :b, :c) --> (:c, :b, :a)) |> size == (4, 6, 8)
        
        # Transpose
        x2d = rand(3, 4)
        t = transpose(x2d)
        @test rearrange(t, (:a, :b) --> (:b, :a)) |> size == (3, 4)
        @test reduce(sum, t, (:a, :b) --> (:a,)) |> size == (4,)
    end
end