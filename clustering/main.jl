include("load_data.jl")

# visualize raw data
using Gadfly
plot(x=x[1, :], y=x[2, :])

