include("spectral.jl")

function compute_accuracy(nearest)
    count = zeros(2, 2)
    for (i, c_algo) in nearest
        c_truth = i <= 100 ? 1 : 2
        count[c_truth, c_algo] += 1
    end
    accA = (count[1,1] + count[2,2]) / 200.0
    accB = (count[2,1] + count[1,2]) / 200.0
    println(count)
    println("$accA $accB")
    accA, accB
end

function main()
    x = include("spiral_data.jl");
    nearest = spectral_clustering(x, 2)

    println(nearest)

    compute_accuracy(nearest)
end

main()
