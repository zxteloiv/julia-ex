w11 = [0.1, 6.8, -3.5, 2.0, 4.1, 3.1, -0.8, 0.9, 5.0, 3.9];
w12 = [1.1, 7.1, -4.1, 2.7, 2.8, 5.0, -1.3, 1.2, 6.4, 4.0];
w21 = [7.1, -1.4, 4.5, 6.3, 4.2, 1.4, 2.4, 2.5, 8.4, 4.1];
w22 = [4.2, -4.3, 0.0, 1.6, 1.9, -3.2, -4.0, -6.1, 3.7, -2.2];
w31 = [-3.0, 0.5, 2.9, -0.1, -4.0, -1.3, -3.4, -4.1, -5.1, 1.9];
w32 = [-2.9, 8.7, 2.1, 5.2, 2.2, 3.7, 6.2, 3.4, 1.6, 5.1];
w41 = [2.0, 8.9, 4.2, 8.5, 6.7, 0.5, 5.3, 8.7, 7.1, 8.0] * -1;
w42 = [8.4, -0.2, 7.7, 3.2, 4.0, 9.2, 6.7, 6.4, 9.7, 6.3] * -1;
const SAMPLE = [w11 w12 w21 w22 w31 w32 w41 w42];


function normalize_data(positive, negative)
    " normalization adds additional one column to data "
    normalize = data -> [data ones(size(data)[1])]

    positive = normalize(positive)
    negative = normalize(negative) * -1
    assert(size(positive) == size(negative))
    data = transpose([positive; negative])

    d, n = size(data)
    weight = zeros(d)

    data, weight
end

function relax_procedure(positive, negative; eta=0.5, margin=0)
    data, weight = normalize_data(positive, negative)
    d, n = size(data)

    println("data = $data")

    function compute_error(data, weight, eta, margin)
        error_samples = [y for y in filter([data[:, i] for i = 1:n]) do y
            dot(y, weight) <= margin
        end]
    end

    step, error_samples = 1, compute_error(data, weight, eta, margin)
    while length(error_samples) > 0 #&& step <= 5
        println("===============================================")
        println("error_samples = $error_samples")
        println("weight = $weight")

        gradient = sum(error_samples) do y
            println(y);
            (margin - dot(weight, y) / dot(y, y)) * y
        end

        weight += eta * gradient
        println("new weight $weight\t<= $gradient")

        step += 1
        error_samples = compute_error(data, weight, eta, margin)
    end

    weight, step
end

function batch_perception(positive, negative)
    data, weight = normalize_data(positive, negative)
    d, n = size(data)

    function compute_error(data, weight)
        error_samples = [y for y in filter([data[:, i] for i = 1:n]) do y
            dot(weight, y) <= 0
        end]
    end

    step, error_samples = 1, compute_error(data, weight)

    while length(error_samples) > 0
        gradient = sum(error_samples)
        weight += gradient
        println("new weight $weight\t<= $gradient")

        step += 1
        error_samples = compute_error(data, weight)
    end

    weight, step
end

"""
Write a program to implement the “batch perception” algorithm
(see page 44 or 45 in PPT).

(a). Starting with a = 0, apply your program to the training data from 1 and 2.
Note that the number of iterations required for convergence.

(b). Apply your program to the training data from 3 and 2.
Again, note that the number of iterations required for convergence.

(c). Explain the difference between the iterations required in the above two cases.
"""
function hw2_1()
    weight, step = batch_perception(SAMPLE[:, 1:2], SAMPLE[:, 3:4])
    println("batch perception w1 and w2: iteration steps: $step, final weights: $weight")
    weight, step = batch_perception(SAMPLE[:, 3:4], SAMPLE[:, 5:6])
    println("batch perception w2 and w3: iteration steps: $step, final weights: $weight")
end

if !isinteractive()
    #hw2_1()
    println(relax_procedure(SAMPLE[:, 1:2], SAMPLE[:, 3:4]))
end

