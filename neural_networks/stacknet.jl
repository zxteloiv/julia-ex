abstract Layer

"""
a perceptron layer with any number
"""
type PerceptronLayer <: Layer
    weights::Array{Number, 2}
    outputs::Array{Number, 1}
    inputs::Array{Number, 1}
    bias::Array{Number, 1}
    activation::Function

end

sigmoid(z::Number) = 1 / (1 + e^(-z))
sigmoid{T<:Number}(arr::Array{T}) = map(sigmoid, arr)
tanh = Base.tanh
relu(x::Real) = max(0, x)

function activate(layer::PerceptronLayer, f::Function)
    layer.outputs = f(layer.outputs)
end

function forward(layer::PerceptronLayer)
    layer.outputs = activate(layer.weights * inputs + bias, layer.activation)
end

function backward{T}(layer::PerceptronLayer{T}, error)
    error * partial * w
end

type StackNet
    layers::Array{Layer, 1}
    loss_func::AbstractString
end

function train(X::Array{Array{Number, 1}, 1}, Y::Array{Array{Number, 1}, 1}, net::StackNet)
end

function test(X::Array{Array{Number, 1}, 1}, Y::Array{Array{Number, 1}, 1}, net::StackNet)
end

println(sigmoid(1))
println(sigmoid([1,2,3]))
