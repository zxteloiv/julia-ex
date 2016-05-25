function gaussian_weight{T<:Number}(x_i::Vector{T}, x_j::Vector{T}; sigma=1)
    exp(-norm(x_i - x_j)^2 / (2sigma^2))
end

function spectral_clustering(inputs::Vector{Vector{Float64}}; k::UInt=0x5)
end
