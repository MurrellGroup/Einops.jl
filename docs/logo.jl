Pkg.activate(temp=true)
Pkg.add(["GLMakie", "GeometryBasics", "Colors"])

using GLMakie
using GeometryBasics
using Colors

# Saving with transparent background
# See https://docs.makie.org/dev/how-to/save-figure-with-transparency

function calculate_rgba(rgb1, rgb2, rgba_bg)::RGBAf
    rgb1 == rgb2 && return RGBAf(rgb1.r, rgb1.g, rgb1.b, 1)
    c1 = Float64.((rgb1.r, rgb1.g, rgb1.b))
    c2 = Float64.((rgb2.r, rgb2.g, rgb2.b))
    alphas_fg = 1 .+ c1 .- c2
    alpha_fg = clamp(sum(alphas_fg) / 3, 0, 1)
    alpha_fg == 0 && return rgba_bg
    rgb_fg = clamp.((c1 ./ alpha_fg), 0, 1)
    rgb_bg = Float64.((rgba_bg.r, rgba_bg.g, rgba_bg.b))
    alpha_final = alpha_fg + (1 - alpha_fg) * rgba_bg.alpha
    rgb_final = @. 1 / alpha_final * (alpha_fg * rgb_fg + (1 - alpha_fg) * rgba_bg.alpha * rgb_bg)
    return RGBAf(rgb_final..., alpha_final)
end

function alpha_colorbuffer(figure)
    scene = figure.scene
    bg = scene.backgroundcolor[]
    scene.backgroundcolor[] = RGBAf(0, 0, 0, 1)
    b1 = copy(colorbuffer(scene))
    scene.backgroundcolor[] = RGBAf(1, 1, 1, 1)
    b2 = colorbuffer(scene)
    scene.backgroundcolor[] = bg
    return map(b1, b2) do b1, b2
        calculate_rgba(b1, b2, bg)
    end
end

save_alpha(name, fig) = save(name, alpha_colorbuffer(fig))


GLMakie.activate!(; fxaa=true)

CUBE     = 1.0f0
SPACE    = 1.2f0
GAP      = 1.1f0
ALPHA    = 0.4f0

BLUE = RGB(0.251, 0.388, 0.847)
GREEN = RGB(0.22, 0.596, 0.149)
PURPLE = RGB(0.584, 0.345, 0.698)
RED = RGB(0.796, 0.235, 0.2)

COLORS = [RED, GREEN, BLUE, PURPLE]

function cube!(scene, p::Point3f, color, shade)
    rect = Rect(p, Vec3f(CUBE, CUBE, CUBE))
    mesh!(scene, rect;
          color         = RGBA(color.r*shade, color.g*shade, color.b*shade, ALPHA),
          transparency  = true)

    v = Point3f.([p .+ CUBE .* Point3f(x,y,z) for (x,y,z) in Iterators.product(0:1, 0:1, 0:1)])
    edges = [(1,2),(2,4),(4,3),(3,1),
             (5,6),(6,8),(8,7),(7,5),
             (1,5),(2,6),(3,7),(4,8)]
    for (a,b) in edges
        linesegments!(scene, [v[a], v[b]];
                      color     = (:white, 0.3),
                      linewidth = 4, transparency=true)
    end
end

function logo(alpha=false)
    fig = Figure(size=(1024, 1024), backgroundcolor=alpha ? :transparent : :white)

    ax = Axis3(fig[1, 1], aspect=:data, backgroundcolor=alpha ? :transparent : :white)
    hidespines!(ax)
    hidedecorations!(ax)

    for (i, xc) in enumerate((-1f0, 1f0))
        for (j, yc) in enumerate((-1f0, 1f0))
            for (k, zc) in enumerate((-1f0, 1f0))
                for a in 1:2, b in 1:2, c in 1:2
                    shade  = 0.6f0 + 0.2f0k
                    color = (isone(k) ? COLORS : reverse(COLORS))[2i - j + 1]
                    origin = Point3f(xc*SPACE + a*GAP, yc*SPACE + b*GAP, zc*SPACE + c*GAP)
                    cube!(ax, origin, color, shade)
                end
            end
        end
    end
    
    display(fig) # for resetting camera

    return fig
end

save_alpha("logo.png", logo(true));

# logo() # for interactive 3D view
