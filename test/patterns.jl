using Einops
using Test

@testset "Pattern System" begin
    @testset "Core Types" begin
        @testset "ArrowPattern construction" begin
            @test (Einops.ArrowPattern{(), ()}()) isa Einops.ArrowPattern
            @test (Einops.ArrowPattern{(), (:a,)}()) isa Einops.ArrowPattern
            @test (Einops.ArrowPattern{(:a,), ()}()) isa Einops.ArrowPattern
            @test (Einops.ArrowPattern{(:a,), (:a,)}()) isa Einops.ArrowPattern
            @test (Einops.ArrowPattern{(:a, Einops.Ellipsis()), (:a,)}()) isa Einops.ArrowPattern
        end

        @testset "ArrowPattern interface" begin
            pattern = Einops.ArrowPattern{(:a, :b, :c), (:c, :b, :a)}()
            @test repr(pattern) == "einops\"a b c -> c b a\""
            
            left, right = pattern
            @test left isa Tuple && right isa Tuple
            
            @test_throws "attempt to access" begin
                left, right, _ = pattern
            end
        end

        @testset "ArrowPattern broadcast" begin
            pattern = Einops.ArrowPattern{(:a, :b, :c), (:c, :b, :a)}()
            @test identity.(pattern) == pattern
        end

        @testset "ArrowPattern shows as its string" begin
            # every spelling reads back as the same pattern
            for p in (einops"a b c -> c (b a)", einops"a 1 b () -> b a", einops"a ... -> (... 1 a)",
                      einops"a 3 b -> a", einops"->", einops"-> 1", einops"α (β γ) -> γ β α",
                      einops"i j, j k -> i k", einops"(a b), -> a b", einops", ->", einops"(i i) ->")
                @test eval(Meta.parse(repr(p))) === p
            end
            @test repr(einops"a b()->a()b") == "einops\"a b () -> a () b\""
            # one array's groups and a list of operands are spelled as written
            @test repr(einops"(h w) (c b) -> h w c b") == "einops\"(h w) (c b) -> h w c b\""
            @test repr(einops"(i) (j) -> i j") == "einops\"(i) (j) -> i j\""
            @test repr(einops"i j, k l -> i k") == "einops\"i j, k l -> i k\""
            @test repr(einops"i j, j k -> i k") == "einops\"i j, j k -> i k\""
            @test repr(einops"a, b -> a b") == "einops\"a, b -> a b\""
            @test repr(einops"D (d h), D ... -> d h ...") == "einops\"D (d h), D ... -> d h ...\""
            @test repr(einops"(a b), -> a b") == "einops\"(a b), -> a b\""
            # an interpolated length shows as its value
            k = 2
            @test repr(einops"(h $k) (w $k) -> h w") == "einops\"(h 2) (w 2) -> h w\""
        end

        @testset "List of operands" begin
            L, R = einops"i j, j k, -> i k"
            @test L isa Einops.List && R === (:i, :k)
            @test length(L) == 3
            @test L[2] === (:j, :k)
            @test Tuple(L) === ((:i, :j), (:j, :k), ())
            @test collect(L) == [(:i, :j), (:j, :k), ()]
            @test map(length, L) === (2, 2, 0)
            @test [op for op in L] == collect(L)
            # a list holds at least two sides
            @test_throws "two or more pattern sides" Einops.List{((:a,),)}()
            @test_throws "two or more pattern sides" Einops.List{(:a, :b)}()
        end

        @testset "ArrowPattern validation" begin
            @test_throws "Invalid pattern" Einops.ArrowPattern{(:a, 'b'), ('b', :a)}()
        end
    end

    @testset "String Pattern Parsing" begin
        @testset "basic patterns" begin
            @test einops"a _ c" == Val((:a, -, :c))
            @test einops"_ _ _" == Val((-, -, -))
        end

        @testset "arrow patterns" begin
            @test einops"a b c -> a (c b)" == (Einops.ArrowPattern{(:a, :b, :c), (:a, (:c, :b))}())
            @test einops"a b c -> a(c b)" == (Einops.ArrowPattern{(:a, :b, :c), (:a, (:c, :b))}())
            @test einops"a b 1 -> a 1 b" == (Einops.ArrowPattern{(:a, :b, 1), (:a, 1, :b)}())
            @test einops"a b () -> a () b" == (Einops.ArrowPattern{(:a, :b, ()), (:a, (), :b)}())
            @test einops"a b()->a()b" == (Einops.ArrowPattern{(:a, :b, ()), (:a, (), :b)}())
            @test einops"b ... -> b ..." == (Einops.ArrowPattern{(:b, Einops.Ellipsis()), (:b, Einops.Ellipsis())}())
            @test einops"b b -> a a" == (Einops.ArrowPattern{(:b, :b), (:a, :a)}())
        end

        @testset "einsum patterns" begin
            @test einops"i j, j k -> i k" === (Einops.ArrowPattern{Einops.List{((:i, :j), (:j, :k))}(), (:i, :k)}())
            @test einops"batch h w, h w channel -> batch channel" === (Einops.ArrowPattern{Einops.List{((:batch, :h, :w), (:h, :w, :channel))}(), (:batch, :channel)}())
            @test einops"(a b) -> a b" === (Einops.ArrowPattern{((:a, :b),), (:a, :b)}())
            @test einops"a b -> a b" === (Einops.ArrowPattern{(:a, :b), (:a, :b)}())
            # a trailing comma adds an operand, unlike in a Julia tuple
            @test einops", ->" === (Einops.ArrowPattern{Einops.List{((), ())}(), ()}())
            @test einops"a, -> a" === (Einops.ArrowPattern{Einops.List{((:a,), ())}(), (:a,)}())
            # commas and parentheses make different patterns
            @test einops"i, j -> i j" !== einops"(i) (j) -> i j"
            @test_throws "Commas separate" Einops.parse_pattern("a, b")
            @test_throws "Commas separate" Einops.parse_pattern("a _, b")
            @test_throws ArgumentError Einops.parse_pattern("a -> a,")
        end

        @testset "empty patterns" begin
            @test einops"->" == (Einops.ArrowPattern{(), ()}())
            @test einops"-> 1" == (Einops.ArrowPattern{(), (1,)}())
        end

        @testset "pack/unpack patterns" begin
            @test einops"i j * k" == Val((:i, :j, *, :k))
            @test einops" i  j  *  k " == Val((:i, :j, *, :k))
            @test einops"* i" == Val((*, :i))
            @test einops"i *" == Val((:i, *))
            @test einops"i i" == Val((:i, :i))
            @test einops"i 1 ..." == Val((:i, 1, Einops.Ellipsis()))  # test map_special_tokens fallback for Int and ..
        end

        @testset "parsing errors" begin
            @test_throws "'.'" Einops.parse_pattern("-> .")
            @test_throws "'('" Einops.parse_pattern("-> (")
            @test_throws "')'" Einops.parse_pattern("-> )")
        end

        @testset "unicode patterns" begin
            @test Einops.parse_pattern("α β γ -> γ β α") == (Einops.ArrowPattern{(:α, :β, :γ), (:γ, :β, :α)}())
            @test Einops.parse_pattern("日 本 語 -> 語 本 日") == (Einops.ArrowPattern{(:日, :本, :語), (:語, :本, :日)}())
            @test Einops.parse_pattern("🔴 🔵 🟢 -> 🟢 🔵 🔴") == (Einops.ArrowPattern{(:🔴, :🔵, :🟢), (:🟢, :🔵, :🔴)}())
            @test Einops.parse_pattern("(α β) γ -> γ (α β)") == (Einops.ArrowPattern{((:α, :β), :γ), (:γ, (:α, :β))}())
            @test Einops.parse_pattern("... α -> α ...") == (Einops.ArrowPattern{(Einops.Ellipsis(), :α), (:α, Einops.Ellipsis())}())
        end
    end

    @testset "Pattern edge cases" begin
        x = rand(2, 3, 4)
        
        # Multiple ellipses in different positions
        @test rearrange(x, Einops.ArrowPattern{(Einops.Ellipsis(), :a), (:a, Einops.Ellipsis())}()) == rearrange(x, Einops.ArrowPattern{(:b, :c, :a), (:a, :b, :c)}())
        @test rearrange(x, Einops.ArrowPattern{(:a, Einops.Ellipsis()), (Einops.Ellipsis(), :a)}()) == rearrange(x, Einops.ArrowPattern{(:a, :b, :c), (:b, :c, :a)}())
        
        # Empty patterns with ellipsis
        x1 = rand(2)
        @test rearrange(x1, Einops.ArrowPattern{(Einops.Ellipsis(),), (Einops.Ellipsis(),)}()) == x1
        @test reduce(sum, x1, Einops.ArrowPattern{(Einops.Ellipsis(),), ()}()) == fill(sum(x1))
        
        # Complex decompositions
        x = rand(2, 12)
        @test rearrange(x, Einops.ArrowPattern{(:a, (:b, :c, :d)), (:d, :c, :b, :a)}(), b=2, c=2) |> size == (3, 2, 2, 2)
        # Nested decompositions beyond 2 levels are not supported
        # @test rearrange(x, einops"a (b (c d)) -> d c b a", b=2, c=2) |> size == (3, 2, 2, 2)
    end
end
