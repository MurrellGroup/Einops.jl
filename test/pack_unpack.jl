using Einops
using Test

@testset "Pack/Unpack Operations" begin
    @testset "Python API reference parity" begin
        # see https://einops.rocks/api/pack_unpack/

        inputs = [rand(2, 3, 5), rand(2, 3, 7, 5), rand(2, 3, 7, 9, 5)]
        @test begin
            packed, ps = pack(inputs, einops"i j * k")
            packed |> size == (2, 3, 71, 5) && ps == [(), (7,), (7, 9)]
        end

        @test begin
            packed, ps = pack(inputs, einops"i j * k")
            inputs_unpacked = unpack(packed, ps, einops"i j * k")
            all(inputs .== inputs_unpacked)
        end
    end
end