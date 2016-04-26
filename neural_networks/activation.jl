"""
sigmoid activation function
"""
sigmoid(z::Number) = 1 / (1 + e^(-z))
sigmoid{T<:Number}(arr::Array{T}) = map(sigmoid, arr)

partial_sigmoid(z::Number) = z * (1 - z)

"""
tanh activation function
"""
tanh = Base.tanh

partial_tanh(z::Number) = 2 / (e^z + e^-z)

"""
ReLU activation function
"""
relu(x::Real) = max(0, x)
relu{T<:Real}(arr::Array{T}) = map(relu, arr)


