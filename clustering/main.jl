x = include("load_data.jl")
include("k-means.jl")

## visualize raw data
#using Gadfly
#plot(x=x[1, :], y=x[2, :])

function main()
    inputs = Vector{Float64}[x[:, i] for i = 1:size(x)[2]]

    centroids = kmeans(convert(UInt, 5), inputs)

    println(centroids)
end

if !isinteractive()
    main()
end



