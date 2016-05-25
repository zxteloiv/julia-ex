include("spectral.jl")

function compute_accuracy(nearest, k)
    count = zeros(k, k)
    for (i, c_algo) in nearest
        c_truth = i <= 100 ? 1 : 2
        count[c_truth, c_algo] += 1
    end
    println(count)
end

function main()
    x = include("spiral_data.jl");
    nearest = spectral_clustering(x, 2; neighbor_num=6, sigma=1)

    println(nearest)

    compute_accuracy(nearest, 2)
end

main()
