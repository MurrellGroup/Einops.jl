using Reactant
using LinearAlgebra
Reactant.set_default_backend("cpu")

@testset "Reactant" begin
    x, y = rand(4, 5), rand(5, 6)
    X, Y = Reactant.to_rarray(x), Reactant.to_rarray(y)

    @testset "pairwise" begin
        pairwise = Base.get_extension(Einops, :ReactantExt).pairwise
        @test pairwise((:i, :j), (:j, :k), (:i, :k)) ==
              (; sumx=(), sumy=(), batch=((), ()), contract=((2,), (1,)), perm=(1, 2))
        @test pairwise((:b, :i, :j), (:b, :j, :k), (:b, :i, :k)) ==
              (; sumx=(), sumy=(), batch=((1,), (1,)), contract=((3,), (2,)), perm=(1, 2, 3))
        @test pairwise((:i, :j), (:j, :k), (:k, :i)).perm == (2, 1)
        @test pairwise((:i, :j), (:k, :l), (:i, :k)) ==
              (; sumx=(2,), sumy=(2,), batch=((), ()), contract=((), ()), perm=(1, 2))
        # positions are counted after the summed-out axes are dropped
        @test pairwise((:s, :a, :i, :j), (:t, :j, :a, :k), (:k, :i, :a)) ==
              (; sumx=(1,), sumy=(1,), batch=((1,), (2,)), contract=((3,), (1,)), perm=(3, 2, 1))
        @test_throws ArgumentError pairwise((:i, :i), (:i, :j), (:j,))
        @test_throws ArgumentError pairwise((:i, :j), (:j, :k), (:i, :z))
    end

    @testset "matmul" begin
        @test Array(@jit einsum(X, Y, einops"i j, j k -> i k")) ≈ x * y
        hlo = string(@code_hlo einsum(X, Y, einops"i j, j k -> i k"))
        @test count("stablehlo.dot_general", hlo) == 1
    end

    @testset "batched" begin
        xb, yb = rand(2, 4, 5), rand(2, 5, 6)
        XB, YB = Reactant.to_rarray(xb), Reactant.to_rarray(yb)
        expected = stack([xb[b, :, :] * yb[b, :, :] for b in 1:2]; dims=1)
        @test Array(@jit einsum(XB, YB, einops"b i j, b j k -> b i k")) ≈ expected
    end

    @testset "permuted output" begin
        @test Array(@jit einsum(X, Y, einops"i j, j k -> k i")) ≈ (x * y)'
    end

    @testset "summed-out axes" begin
        @test Array(@jit einsum(X, Y, einops"i j, k l -> i k")) ≈ sum(x; dims=2) * sum(y; dims=2)'
    end

    @testset "all batch" begin
        @test Array(@jit einsum(X, X, einops"i j, i j -> i j")) ≈ x .* x
    end

    @testset "traced with plain array" begin
        @test Array(@jit einsum(X, y, einops"i j, j k -> i k")) ≈ x * y
        @test Array(@jit einsum(x, Y, einops"i j, j k -> i k")) ≈ x * y
    end

    @testset "grouped axes and wrappers" begin
        D, hd, H, Hkv, T, B = 8, 4, 4, 2, 5, 3
        x, wq, wk = rand(Float32, D, T, B), rand(Float32, D, hd * H), rand(Float32, D, hd * Hkv)
        wo3 = rand(Float32, hd, H, D)
        X, WQ, WK, WO3 = Reactant.to_rarray.((x, wq, wk, wo3))

        split(w, x) = einsum(w, x, einops"D (d h), D ... -> d h ..."; h=H)
        q = split(wq, x)
        Q = @jit split(WQ, X)
        @test Q isa Reactant.AbstractConcreteArray   # `expand`'s lazy reshape does not leak
        @test Array(Q) ≈ q
        @test count("stablehlo.dot_general", string(@code_hlo split(WQ, X))) == 1

        k = einsum(wk, x, einops"D (d h), D ... -> d h ..."; h=Hkv)
        K = Reactant.to_rarray(k)
        scores(k, q) = einsum(k, q, einops"d h k ..., d (g h) q ... -> k q g h ..."; h=Hkv)
        @test Array(@jit scores(K, Q)) ≈ scores(k, q)
        @test count("stablehlo.dot_general", string(@code_hlo scores(K, Q))) == 1

        # the group axis merges back on the right, and the result is a plain array
        merge_(v, p) = einsum(v, p, einops"d h k ..., k q g h ... -> d (g h) q ...")
        p = scores(k, q)
        @test Array(@jit merge_(K, Reactant.to_rarray(p))) ≈ merge_(k, p)
        @test (@jit merge_(K, Reactant.to_rarray(p))) isa Reactant.AbstractConcreteArray

        # two contracted axes at once, adjacent in memory
        out3(w, o) = einsum(w, o, einops"d h D, d h ... -> D ...")
        @test Array(@jit out3(WO3, Q)) ≈ out3(wo3, q)
        @test count("stablehlo.dot_general", string(@code_hlo out3(WO3, Q))) == 1

        # a transposed operand is an `Adjoint` wrapper
        y = rand(Float32, D, 6)
        adj(x, y) = einsum(x', y, einops"i j, j k -> i k")     # x' is (hd*H, D), y is (D, 6)
        @test Array(@jit adj(WQ, Reactant.to_rarray(y))) ≈ adj(wq, y)
    end

    @testset "single operand" begin
        x = rand(Float32, 4, 6, 5)
        X = Reactant.to_rarray(x)
        red(x) = einsum(x, einops"i j k -> k i")
        @test Array(@jit red(X)) ≈ red(x)
        explicit_red(x) = einsum(x, (:i, :j, :k) --> (:k, :i))
        @test Array(@jit explicit_red(X)) ≈ red(x)
        @test red(x) ≈ permutedims(dropdims(sum(x; dims=2); dims=2), (2, 1))
        grouped(x) = einsum(x, einops"(a b) j k -> b k"; a=2)
        @test Array(@jit grouped(X)) ≈ grouped(x)
        @test (@jit grouped(X)) isa Reactant.AbstractConcreteArray
        hlo = string(@code_hlo red(X))
        @test count("stablehlo.reduce", hlo) == 1
        @test !occursin("stablehlo.while", hlo)
        split_only(x) = Einops.@einsum(x, "(a b) -> a b"; a=2)
        v = rand(Float32, 6)
        @test Array(@jit split_only(Reactant.to_rarray(v))) == reshape(v, 2, 3)
        scaled_split(x, s) = Einops.@einsum(x, s, "(a b), -> a b"; a=2)
        @test Array(@jit scaled_split(Reactant.to_rarray(v), Reactant.to_rarray(fill(2f0)))) ≈ 2 .* reshape(v, 2, 3)
    end
end
