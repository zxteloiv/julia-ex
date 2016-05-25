#!/usr/bin/env julia

include("logger.jl")

"""
Do k-means clustering for inputs, each of which is a single data sample.
    @k the number of clusters
    @inputs the input data vector, each element is a vector in feature space.

    return the centroids, each of which is a cluster center vector in feature space.
        the cluster indicator array, each element is the cluster id which
        contains the data example indicated by the array index.
"""
function kmeans{T <: Number}(k::Int, inputs::Vector{Vector{T}}; lowerbound=0.00001, maxiter=1000)
    assert(k > 1)

    # initialize k centroids from inputs
    centroids = init_centroids(k, inputs)
    dbglog("init: centroids=$(formatnum(centroids))")

    iter, sumdiff = 0, 1
    while sumdiff > lowerbound && iter < maxiter
        nearest_centroids = find_nearest_centroids(inputs, centroids)
        new_centroids = find_new_centroids(k, inputs, nearest_centroids)

        sumdiff = sum([dot(v, v) for v in new_centroids - centroids]) 
        iter += 1

        centroids = new_centroids

        dbglog("======\n$iter: diff=$sumdiff centroids=$(formatnum(centroids))")
    end

    # the final cluster assignment
    nearest_centroids = find_nearest_centroids(inputs, centroids)

    return centroids, nearest_centroids
end

"""
Do k-means clustering for inputs
    @k the number of clusters
    @inputs the input data matrix, each column is a vector in feature space.

    return the centroids, each of which is a cluster center vector in feature space.
        the cluster indicator array, each element is the cluster id which
        contains the data example indicated by the array index.
"""
function kmeans{T <: Number}(k::Int, inputs::Matrix{T}; lowerbound=0.00001, maxiter=1000)
    inputs = Vector{T}[inputs[:, j] for j = 1:(size(inputs)[2])]
    kmeans(k, inputs, lowerbound=lowerbound, maxiter=maxiter)
end

"""
Find the new centroids
"""
function find_new_centroids{T <: Number}(k::Int, inputs::Vector{Vector{T}}, nearest_centroids)
    new_centroids = Vector[ begin
        vsum, count = 0, 0
        for (i, min_c) in filter(x -> x[2] == c, nearest_centroids)
            vsum += inputs[i]
            count += 1
        end
        float(vsum) / count 
    end for c = 1:k ]

    return convert(Vector{Vector{Float64}}, new_centroids)
end

"""
Find the nearest centroids for all the data samples.
    @k the number of clusters
    @inputs
    @centroids vector of length k 

    return an array, each element of which is a pair of array index and id of the
    nearest cluster.
"""
function find_nearest_centroids{T <: Number}(inputs::Vector{Vector{T}}, centroids::Vector{Vector{T}})
    # find the nearest centroid for each input
    k = length(centroids)
    belonging = map(enumerate(inputs)) do x
        i, v = x
        min_c = indmin([(d = v - centroids[c]; dot(d, d)) for c in 1:k])
        i, min_c
    end
end

"""
Initialize the centroids.
    @k the number of cluster needed
    @inputs

return an array of centroids, each is a vector in feature space.
"""
function init_centroids{T <: Number}(k::Int, inputs::Vector{Vector{T}})
    assert(length(inputs) > k)
    
    centroids = Vector{Vector{T}}()

    for c = 1:k
        r = rand(inputs)
        while r in centroids
            r = rand(inputs)
        end
        push!(centroids, r)
    end

    return centroids
end
