include("logger.jl")

include("layer.jl")

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
        error("not implemented for this partial $foo")
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


