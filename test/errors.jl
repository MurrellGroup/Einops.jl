using Einops
using Test

# The exception an expression throws, unwrapped from the `LoadError` of a macro's expansion.
macro thrown(ex)
    quote
        try
            $(esc(ex))
            nothing
        catch e
            e isa LoadError ? e.error : e
        end
    end
end

# Each message whole, so they stay in one voice: names and groups as the pattern writes them,
# "left"/"right" for its sides, "the input" or "operand k" for the arrays, an `ArgumentError`
# for a pattern that can't work and a `DimensionMismatch` for an array that doesn't fit it.
@testset "Error messages" begin
    x = rand(2, 3, 5)
    k = 2

    @testset "parsing" begin
        @test @thrown(Einops.parse_pattern("a -> b -> c")) == ArgumentError("Pattern has more than one `->`")
        @test @thrown(Einops.parse_pattern("a b) -> a b")) == ArgumentError("Unmatched `)` in pattern")
        @test @thrown(Einops.parse_pattern("(a b -> a b")) == ArgumentError("Unmatched `(` in pattern")
        @test @thrown(Einops.parse_pattern("a .. -> a")) == ArgumentError("`.` in a pattern must be part of `...`")
        @test @thrown(Einops.parse_pattern("a, b -> a, b")) == ArgumentError("Commas separate einsum operands, which go left of `->`")
        @test @thrown(Einops.parse_pattern("a, b")) == ArgumentError("Commas separate einsum operands, so the pattern needs `->`")
        @test @thrown(rearrange(x, einops"a, b -> a b")) == ArgumentError("The pattern lists 2 operands, but only `einsum` takes more than one")
        @test @thrown(@macroexpand @rearrange(x)) == ArgumentError("`@rearrange` takes its pattern as a string literal")
        @test @thrown(@macroexpand @rearrange(x, "a -> a", "b -> b")) == ArgumentError("`@rearrange` takes only one pattern string")
    end

    @testset "names" begin
        @test @thrown(rearrange(x, einops"a a b -> a b")) == ArgumentError("`a` appears more than once on the left side")
        @test @thrown(rearrange(x, einops"a b c -> a b c a")) == ArgumentError("`a` appears more than once on the right side")
        @test @thrown(rearrange(x, einops"a b c -> a")) == ArgumentError("`b` and `c` are only on the left side; to remove an axis, use `reduce`")
        @test @thrown(rearrange(x, einops"a ... -> a")) == ArgumentError("`...` is only on the left side; to remove an axis, use `reduce`")
        @test @thrown(repeat(x, einops"a b c -> a b")) == ArgumentError("`c` is only on the left side; to remove an axis, use `reduce`")
        for err in (@thrown(rearrange(x, einops"a b c -> a b c d")), @thrown(@macroexpand @rearrange(x, "a b c -> a b c d")),
                    @thrown(reduce(sum, x, einops"a b c -> a d")))
            @test err == ArgumentError("`d` is only on the right side; to create an axis, use `repeat`")
        end
        @test @thrown(reshape(x, einops"a b c -> c b a")) == ArgumentError("`reshape` can't reorder names, as `a b c -> c b a` does; use `rearrange`")
        @test @thrown(reshape(x, einops"a ... -> ... a")) == ArgumentError("`reshape` can't reorder names, as `a ... -> ... a` does; use `rearrange`")
        @test @thrown(reshape(x, einops"a b c -> a b")) == ArgumentError("`c` is only on the left side; to remove an axis, use `reduce`")
    end

    @testset "lengths" begin
        @test @thrown(rearrange(rand(6), einops"(a b) -> a b")) == ArgumentError("Can't infer `a` and `b` in `(a b)`; pass the lengths of all but one as keywords")
        @test @thrown(reduce(sum, rand(12), einops"(h 2 w) -> h")) == ArgumentError("Can't infer `h` and `w` in `(h 2 w)`; pass the lengths of all but one as keywords")
        @test @thrown(repeat(x, einops"a b c -> a b c d")) == ArgumentError("`repeat` needs the length of new axis `d`; pass it as a keyword")
        @test @thrown(@macroexpand @repeat(x, "a b c -> (r a) b c s")) == ArgumentError("`repeat` needs the lengths of new axes `r` and `s`; pass them as keywords")
        @test @thrown(rearrange(x, einops"a b ((c)) -> a b c")) == ArgumentError("Groups can't be nested: `((c))`")
        @test @thrown(rearrange(x, einops"a b 5 -> a b")) == ArgumentError(
            "Unnamed axis `5` can only be reduced (left side of `reduce`) or created (right side of `repeat`); name it and pass its length as a keyword")
        @test @thrown(rearrange(rand(6), einops"($k b) -> b")) == ArgumentError(
            "Unnamed axis `\$k` can only be reduced (left side of `reduce`) or created (right side of `repeat`); name it and pass its length as a keyword")
        @test @thrown(rearrange(x, einops"a b c -> a b c", b=4)) == DimensionMismatch("Axis 2 of the input has length 3, but the pattern expects 4 (b = 4)")
    end

    @testset "ranks and ellipses" begin
        @test @thrown(rearrange(x, einops"a b -> a b")) == DimensionMismatch("The input has 3 dimensions, but the pattern expects 2")
        @test @thrown(rearrange(rand(2), einops"a b ... -> a b ...")) == DimensionMismatch("The input has 1 dimension, but the pattern expects at least 2")
        @test @thrown(parse_shape(x, einops"a b")) == DimensionMismatch("The input has 3 dimensions, but the pattern expects 2")
        @test @thrown(parse_shape(x, (:a, :b))) == DimensionMismatch("The input has 3 dimensions, but the pattern expects 2")
        @test @thrown(parse_shape(x, einops"a a b")) == DimensionMismatch("`a` has length 2 at axis 1 but 3 at axis 2")
        # the function and the inlined macro agree
        for (pattern, message) in (("a ... ... -> a ...", "`...` appears more than once on the left side"),
                                   ("a (b ...) -> a b ...", "`...` can't be in a group on the left side"),
                                   ("a b c -> a b c ...", "`...` is only on the right side"))
            @test @thrown(rearrange(x, Einops.parse_pattern(pattern))) == ArgumentError(message)
            @test @thrown(macroexpand(@__MODULE__, :(@rearrange(x, $pattern)))) == ArgumentError(message)
        end
        @test @thrown(rearrange(x, einops"a ... -> ... a ...")) == ArgumentError("`...` appears more than once on the right side")
        @test @thrown(parse_shape(x, einops"a ... ...")) == ArgumentError("`...` appears more than once in the pattern")
    end

    @testset "einsum" begin
        @test @thrown(einsum(x, x, einops"a b c -> a")) == ArgumentError("The pattern has 1 operand, but 2 arrays were given")
        @test @thrown(einsum(x, rand(2, 3), einops"a b c, a b c -> a")) == DimensionMismatch("Operand 2 has 2 dimensions, but the pattern expects 3")
        @test @thrown(einsum(rand(2, 3), rand(4, 3), einops"a b, a b -> a")) == DimensionMismatch("`a` has length 2 in operand 1 but 4 in operand 2")
        @test @thrown(einsum(rand(2, 3), rand(2, 3), rand(2, 3, 4), einops"a ..., a ..., a ... -> a ...")) ==
              DimensionMismatch("`...` spans 1 dimension in operand 1 but 2 in operand 3")
        @test @thrown(einsum(x, x, einops"a b c, a b c -> d")) == ArgumentError("`d` is on the right side but in no operand")
        @test @thrown(einsum(x, x, einops"a b c, a b c -> a a")) == ArgumentError("`a` appears more than once on the right side")
        @test @thrown(einsum(x, x, einops"a b c, a (b ...) -> a")) == ArgumentError("`...` can't be in a group in operand 2")
        @test @thrown(einsum(x, x, einops"a b c, a b c -> a ...")) == ArgumentError("`...` is only on the right side")
        @test @thrown(einsum(x, x, einops"a b c, a b c -> a 2")) == ArgumentError("Unnamed axis `2` isn't supported by `einsum`; name it and pass its length as a keyword")
        @test @thrown(@macroexpand @einsum(x, x, "($k a) b c, a b c -> a")) ==
              ArgumentError("Unnamed axis `\$k` isn't supported by `einsum`; name it and pass its length as a keyword")
        @test @thrown(einsum(x, x, einops"a b c, a b c -> a"; b=4)) == DimensionMismatch("Axis 2 of operand 1 has length 3, but the pattern expects 4 (b = 4)")
    end

    @testset "pack" begin
        @test @thrown(pack([x], einops"a * *")) == ArgumentError("A pack pattern takes exactly one `*`, but has 2")
        @test @thrown(pack([x], einops"a b")) == ArgumentError("A pack pattern takes exactly one `*`, but has none")
        @test @thrown(pack([x], einops"a a *")) == ArgumentError("`a` appears more than once in the pattern")
        @test @thrown(pack([x, rand(2)], einops"i j *")) == DimensionMismatch("Array 2 has 1 dimension, but the pattern expects at least 2")
        @test @thrown(unpack(rand(2, 3), [(1,)], einops"a b c *")) == DimensionMismatch("The input has 2 dimensions, but the pattern expects 4")
    end
end
