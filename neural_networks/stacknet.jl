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

end


sigmoid(z::Number) = 1 / (1 + e^(-z))
sigmoid{T<:Number}(arr::Array{T}) = map(sigmoid, arr)
tanh = Base.tanh
relu(x::Real) = max(0, x)
relu{T<:Real}(arr::Array{T}) = map(relu, arr)


function activate(layer::PerceptronLayer, f::Function)
    layer.outputs = f(layer.outputs)
end

function forward(layer::PerceptronLayer)
    net = layer.weights * layer.inputs + layer.bias
    layer.outputs = activate(net, layer.activation)
end

function backward(layer::PerceptronLayer, error::Vector{Number})
    error * partial * w
end

type StackNet
    layers::Array{Layer, 1}
    loss_func::Function
end

function train(X::Vector{Vector{Number}}, Y::Vector{Vector{Number}}, net::StackNet)
end

function test(X::Vector{Vector{Number}}, Y::Vector{Vector{Number}}, net::StackNet)
end

println(sigmoid(1))
println(sigmoid([1,2,3]))
