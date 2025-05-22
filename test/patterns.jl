using Einops
using Test

@testset "Pattern System" begin
    @testset "Core Types" begin
        @testset "ArrowPattern construction" begin
            @test (() --> ()) isa ArrowPattern
            @test (() --> :a) isa ArrowPattern
            @test (:a --> ()) isa ArrowPattern
            @test (:a --> :a) isa ArrowPattern
            @test ((:a, ..) --> (:a,)) isa ArrowPattern
        end

        @testset "ArrowPattern interface" begin
            pattern = (:a, :b, :c) --> (:c, :b, :a)
            @test repr(pattern) == "(:a, :b, :c) --> (:c, :b, :a)"
            
            left, right = pattern
            @test left isa Tuple && right isa Tuple
            
            @test_throws "attempt to access" begin
                left, right, _ = pattern
            end
        end

        @testset "ArrowPattern validation" begin
            @test_throws "Invalid pattern" (:a, 'b') --> ('b', :a)
        end
    end

    @testset "String Pattern Parsing" begin
        @testset "basic patterns" begin
            @test einops"a _ c" == (:a, -, :c)
            @test einops"_ _ _" == (-, -, -)
        end

        @testset "arrow patterns" begin
            @test einops"a b c -> a (c b)" == ((:a, :b, :c) --> (:a, (:c, :b)))
            @test einops"a b c -> a(c b)" == ((:a, :b, :c) --> (:a, (:c, :b)))
            @test einops"a b 1 -> a 1 b" == ((:a, :b, 1) --> (:a, 1, :b))
            @test einops"a b () -> a () b" == ((:a, :b, ()) --> (:a, (), :b))
            @test einops"a b()->a()b" == ((:a, :b, ()) --> (:a, (), :b))
            @test einops"b ... -> b ..." == ((:b, ..) --> (:b, ..))
            @test einops"b b -> a a" == ((:b, :b) --> (:a, :a))
        end

        @testset "einsum patterns" begin
            @test einops"i j, j k -> i k" == (((:i, :j), (:j, :k)) --> (:i, :k))
            @test einops"batch h w, h w channel -> batch channel" == (((:batch, :h, :w), (:h, :w, :channel)) --> (:batch, :channel))
        end

        @testset "empty patterns" begin
            @test einops"->" == (() --> ())
            @test einops"-> 1" == (() --> (1,))
        end

        @testset "pack/unpack patterns" begin
            @test einops"i j * k" == (:i, :j, *, :k)
            @test einops" i  j  *  k " == (:i, :j, *, :k)
            @test einops"* i" == (*, :i)
            @test einops"i *" == (:i, *)
            @test einops"i i" == (:i, :i)
        end

        @testset "parsing errors" begin
            @test_throws "'.'" Einops.parse_pattern("-> .")
            @test_throws "'('" Einops.parse_pattern("-> (")
            @test_throws "')'" Einops.parse_pattern("-> )")
        end
    end

    @testset "Pattern edge cases" begin
        x = rand(2, 3, 4)
        
        # Multiple ellipses in different positions
        @test rearrange(x, (.., :a) --> (:a, ..)) == rearrange(x, (:b, :c, :a) --> (:a, :b, :c))
        @test rearrange(x, (:a, ..) --> (.., :a)) == rearrange(x, (:a, :b, :c) --> (:b, :c, :a))
        
        # Empty patterns with ellipsis
        x1 = rand(2)
        @test rearrange(x1, (..,) --> (..,)) == x1
        @test reduce(sum, x1, (..,) --> ()) == fill(sum(x1))
        
        # Complex decompositions
        x = rand(2, 12)
        @test rearrange(x, (:a, (:b, :c, :d)) --> (:d, :c, :b, :a), b=2, c=2) |> size == (3, 2, 2, 2)
        # Nested decompositions beyond 2 levels are not supported
        # @test rearrange(x, (:a, (:b, (:c, :d))) --> (:d, :c, :b, :a), b=2, c=2) |> size == (3, 2, 2, 2)
    end
end