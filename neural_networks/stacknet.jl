import Lumberjack
dbglog(x::Any) = Lumberjack.debug(string(x))
dbglog(x::Any...) = Lumberjack.debug(string(x))
infolog(x::Any) = Lumberjack.info(string(x))
infolog(x::Any...) = Lumberjack.info(string(x))

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
function forward(layer::PerceptronLayer, inputs)
    layer.inputs = inputs
    net = layer.weights * layer.inputs + layer.bias
    layer.outputs = layer.activation(net)
end

"""
Backward error update for PerceptronLayer
"""
function backward(layer::PerceptronLayer, error; eta=0.05)
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
function forward(net::StackNet, sample)
    outputs = sample
    for i = 1:length(net.layers)
        outputs = forward(net.layers[i], outputs)
    end
    outputs
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

"""
show the net
"""
function shownet(net::StackNet)
    head = """
    net::StackNet
    loss: $(net.loss_func)
    Layers: $(length(net.layers))
    """
    layer = ["""
    -----------------
        $(net.layers[i].activation)
        weights $(net.layers[i].weights)
        bias $(net.layers[i].bias)
        inputs $(net.layers[i].inputs)
        outputs $(net.layers[i].outputs)
    """ for i in 1:length(net.layers)]
    head * join(layer, "\n")
end

function train(X, Y, net::StackNet)
    for epoch = 1:500
        for i = 1:size(X)[2]
            # every single sample
            outputs = forward(net, X[:, i])
            loss = net.loss_func(outputs)
            dbglog("loss $(loss[1])")

            error = backward(net, outputs, Y[i])
        end
    end
end

function test(X, Y, net::StackNet)
    for i = 1:size(X)[2]
        # every single sample
        outputs = forward(net, X[:, i])
        #dbglog("loss $(loss[1])")

        infolog(outputs[1], Y[i])
    end
end

# network testing

function main()

    X = [
    1 2 0 0 1 0 0 -1 -2 -1;
    0 0 -1 -2 -1 1 2 0 0 1
    ]

    testX = [
    2 1 -1 -2;
    -1 -2 2 1
    ]

    testY = [ 1 1 -1 -1 ]

    Y = [ 1 1 1 1 1 -1 -1 -1 -1 -1 ]


    dbglog(X, Y)

    net = StackNet(sigmoid)
    add_layer(net, PerceptronLayer(2, 3))
    add_layer(net, PerceptronLayer(3, 1))

    dbglog(shownet(net))
    test(testX, testY, net)

    train(X, Y, net)
    dbglog(shownet(net))

    test(testX, testY, net)

end

if !isinteractive()
    main()
end


