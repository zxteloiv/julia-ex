abstract Layer

"""
a perceptron layer with any number
"""
type PerceptronLayer <: Layer
    weights::Matrix{Number}
    outputs::Vector{Number}
    inputs::Vector{Number}
    bias::Vector{Number}
    activation::Function

    PerceptronLayer(input_num::Integer, output_num::Integer, activation::Function) = begin
        # weights * input = output, therefore the row should be output nodes
        weights = rand(output_num, input_num)
        for i = 1:output_num
            weights[i, :] ./= sum(weights[i, :])
        end

        new(weights, zeros(output_num), zeros(input_num), rand(output_num), activation)
    end
end

"""
sigmoid activation function
"""
sigmoid(z::Number) = 1 / (1 + e^(-z))
sigmoid{T<:Number}(arr::Array{T}) = map(sigmoid, arr)

"""
tanh activation function
"""
tanh = Base.tanh

"""
ReLU activation function
"""
relu(x::Real) = max(0, x)
relu{T<:Real}(arr::Array{T}) = map(relu, arr)

"""
Activation in PerceptronLayer
"""
function activate(layer::PerceptronLayer, f::Function)
    layer.outputs = f(layer.outputs)
end

"""
Partial derivative value of activation functions w.r.t. layer_weight
"""
function activation_partial(layer::PerceptronLayer)
    map(layer.outputs) do z
        if layer.activation == sigmoid
            z * (1 - z)
        elseif layer.activation == tanh
            2 / (e^z + e^-z)
        else
            warn("partial for activation not implemented")
            z
        end
    end
end

"""
Forward computing in PerceptronLayer
"""
function forward(layer::PerceptronLayer, inputs::Vector{Number})
    layer.inputs = inputs
    net = layer.weights * layer.inputs + layer.bias
    layer.outputs = activate(net, layer.activation)
end

"""
Backward error update for PerceptronLayer
"""
function backward(layer::PerceptronLayer, error::Vector{Number}; eta=0.05)
    activation_derivatives = activation_partial(layer)

    # build new error to propagate backward
    new_error = [ sum([error[j] * activation_derivatives[j] * layer.weights[j, i] for j = 1:output_num]) for i = 1:input_num ]

    (output_num, input_num) = size(layer)
    for i = 1:input_num, j = 1:output_num
        gradient = error[j] * activation_derivatives[j] * layer.input[i]

        layer.weights[j, i] -= eta * gradient
    end

    new_error
end

"""
Square error loss function
"""
square_error(output, label) = (error = output - label; 1/2 * dot(error, error))

"""
Compute partial derivatives of the loss function with respect to output
"""
function partial_error(foo::Function, output, label)
    if foo == square_error
        return output - label
    else
        return output - label
    end
end

"""
StackNet is network with layers structure like a stack
"""
type StackNet
    layers::Array{Layer, 1}
    loss_func::Function

    StackNet(loss_func::Function) = new(Array{Layer, 1}(), loss_func)
end

add_layer(net::StackNet, layer::Layer) = push!(net.layers, layer)

"""
do the forward computation on the entire stack network
"""
function forward(sample::Vector{Number}, net::StackNet)
    outputs = sample
    for i = 1:length(net.layers)
        outputs = forward(net.layers[i], outputs)
    end
    outputs = net.loss_func(outputs)
end

"""
correct the network with a single sample error
"""
function backward(net::StackNet, last_output, label)
    error = partial_error(net.loss_func, last_output, label)
    for i = length(net.layers):-1:1
        error = backward(net.layers[i], error)
    end
end

function train(X::Vector{Vector{Number}}, Y::Vector{Vector{Number}}, net::StackNet)
    for i = 1:length(X)
        output = forward(X[i], Y[i], net)
    end
end

function test(X::Vector{Vector{Number}}, Y::Vector{Vector{Number}}, net::StackNet)
end

println(sigmoid([1,2,3]))
