const warn_str = "Einops supports `Base.String` patterns for Einops-native functions only for convenience, as using them is type unstable."

function parse_shape(x, pattern::String) 
    Base.depwarn(warn_str, :parse_shape)
    return parse_shape(x, parse_pattern(pattern))
end

function rearrange(x, pattern::String; context...)
    Base.depwarn(warn_str, :rearrange)
    rearrange(x, parse_pattern(pattern); context...)
end

function _einsum(pattern::String, arrays...; kws...)
    Base.depwarn(warn_str, :_einsum)
    _einsum(parse_pattern(pattern), arrays...; kws...)
end

function pack(x, pattern::String)
    Base.depwarn(warn_str, :pack)
    return pack(x, parse_pattern(pattern))
end

function unpack(x, ps, pattern::String)
    Base.depwarn(warn_str, :unpack)
    return unpack(x, ps, parse_pattern(pattern))
end
