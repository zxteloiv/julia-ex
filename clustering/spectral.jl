include("k-means.jl")

function gaussian_weight{T<:Number}(x_i::Vector{T}, x_j::Vector{T}; sigma=1)
    exp(-norm(x_i - x_j)^2 / (2sigma^2))
end

function gaussian_weight(normval::Float64; sigma=1)
    exp(-normval^2 / (2sigma^2))
end

"""
Do spectral clustering based on Ng's algorithm (Ng, Jordan, Weiss, & others, 2002).

    @inputs the input data matrix, each column is a vector in feature space.
    @cluster_num the desired number of clusters

    returns an array, whose elements are pairs of array index and the cluster id
"""
function spectral_clustering(inputs::Matrix{Float64}, cluster_num::Int;
    neighbor_num::Int=5, sigma=1)

    W, D = build_matrix_normal_knn(inputs, k=neighbor_num, sigma=sigma)
    L_sym = Symmetric(D^(-1/2) * W * D^(-1/2))

    eigenvals, eigenvecs = eig(L_sym, 1:cluster_num)

    # normalize the eigen vectors
    U = eigenvecs
    for i = 1:size(U)[1], j = 1:size(U)[2]
        U[i, j] = U[i, j] / norm(U[i, :])
    end

    centroids, nearest_centroids = kmeans(cluster_num, inputs)

    return nearest_centroids
end

using NearestNeighbors
"""
Build a Adjacency Matrix using Normal Function as similarity function
    and only k-nearest-neighborhood have edges connected.

    @inputs input data as a matrix, each column is an input sample
    @k only the nearest k samples have an edge connected with the given point
    @sigma the parameter of the gaussian function

    return the pair of the adjacency matrix and the diagnoal matrix
"""
function build_matrix_normal_knn(inputs::Matrix{Float64}; k::Int=10, sigma=1)
    kdtree = KDTree(inputs)
    N = size(inputs)[2]
    adj = zeros(N, N)

    degree = zeros(N) # diagnoal vector at first

    for i = 1:N
        idxs, dists = knn(kdtree, inputs[:, i], k, true)
        for j in 1:length(idxs)
            weight = idxs[j] == i ? 0 : gaussian_weight(dists[j], sigma=sigma)
            adj[i, idxs[j]] = weight
            adj[idxs[j], i] = weight
            degree[i] += weight
        end
    end

    degree = diagm(degree)

    return adj, degree
end
