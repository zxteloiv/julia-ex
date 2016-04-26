include("activation.jl")

abstract Layer

"""
a perceptron layer with any number
"""
type PerceptronLayer <: Layer
    weights::Matrix{Float64}
    outputs::Vector{Float64}
    inputs::Vector{Float64}
    bias::Vector{Float64}
    activation::Function

    PerceptronLayer(input_num::Integer, output_num::Integer, activation::Function=sigmoid) = begin
        # weights * input = output, therefore the row should be output nodes
        weights = rand(output_num, input_num)
        for i = 1:output_num
            weights[i, :] ./= sum(weights[i, :])
        end

        new(weights, zeros(output_num), zeros(input_num), rand(output_num), activation)
    end
end

"""
Forward computing in PerceptronLayer
"""
function forward(layer::PerceptronLayer, inputs)
    layer.inputs = inputs
    net = layer.weights * layer.inputs + layer.bias
    layer.outputs = layer.activation(net)
end

"""
Backward error update for PerceptronLayer
"""
function backward(layer::PerceptronLayer, error; eta=0.03)
    activation_derivatives = activation_partial(layer)
    output_num, input_num = size(layer.weights)

    # build new error to propagate backward
    new_error = [ sum([error[j] * activation_derivatives[j] * layer.weights[j, i] for j = 1:output_num]) for i = 1:input_num ]

    for i = 1:input_num, j = 1:output_num
        gradient = error[j] * activation_derivatives[j] * layer.inputs[i]

        layer.weights[j, i] -= eta * gradient
    end

    new_error
end

"""
Partial derivative value of activation functions w.r.t. layer_weight
"""
function activation_partial(layer::PerceptronLayer)
    map(layer.outputs) do z
        if layer.activation == sigmoid
            partial_sigmoid(z)
        elseif layer.activation == tanh
            partial_tanh(z)
        else
            warn("partial for activation not implemented")
            z
        end
    end
end

