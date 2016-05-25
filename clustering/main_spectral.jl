include("spectral.jl")

function compute_accuracy(nearest, k)
    count = zeros(k, k)
    for (i, c_algo) in nearest
        c_truth = i <= 100 ? 1 : 2
        count[c_truth, c_algo] += 1
    end
    count
end

function trial(x, cluster_num; neighbor_num=10, sigma=1)
    nearest = spectral_clustering(x, cluster_num; neighbor_num=neighbor_num, sigma=sigma)
    count = compute_accuracy(nearest, cluster_num)

    println("k=$neighbor_num\tsigma=$sigma\t$(collect(count))")
end

function main()
    x = include("spiral_data.jl");
    cluster_num  = 2;

    # fix sigma = 1, neighbor_num varies
    sigma = 1
    neighbor_nums = [5, 10, 20, 30, 50, 200]
    for neighbor_num in neighbor_nums
        trial(x, cluster_num, neighbor_num=neighbor_num, sigma=sigma)
    end

    # fix neighbor_num = 10/20/30, sigma varies
    neighbor_num = 30
    sigmas = [0.01, 0.1, 0.5, 1, 2, 5, 10, 15]
    for sigma in sigmas
        trial(x, cluster_num, neighbor_num=neighbor_num, sigma=sigma)
    end

end

main()
