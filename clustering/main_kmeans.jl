include("logger.jl")
x = include("load_data.jl")
include("k-means.jl")

## visualize raw data
#using Gadfly
#plot(x=x[1, :], y=x[2, :])

function compute_final_error(centroids, nearest_centroids)
    sum = 0.0
    for (i, c) in nearest_centroids
        real_centroid = get_real_centroid(i)
        diff = real_centroid - centroids[c]
        sum += dot(diff, diff)
    end

    sum / length(nearest_centroids)
end

function cluster_stat(k, nearest_centroids)
    counts = zeros(k)
    for (i, c) in nearest_centroids
        counts[c] += 1
    end
    counts
end

function get_real_centroid(i)
    if i <= 200
        [1., -1.]
    elseif i <= 400
        [5.5, -4.5]
    elseif i <= 600
        [1., 4.]
    elseif i <= 800
        [6., 4.5]
    else
        [9., 0.0]
    end
end

function main()
    inputs = Vector{Float64}[x[:, i] for i = 1:size(x)[2]]

    centroids, nearest_centroids = kmeans(5, inputs, lowerbound=0.000001)

    println("=====\nfinal=$(formatnum(centroids))")
    error = compute_final_error(centroids, nearest_centroids)
    dbglog("error=$(formatnum(error))")
    stat = cluster_stat(5, nearest_centroids)
    dbglog("stat=$(formatnum(stat))")

    return centroids, nearest_centroids
end

if !isinteractive()
    main()
end



