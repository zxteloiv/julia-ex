include("stacknet.jl")
include("logger.jl")

function testing()

    function train(X, Y, net::StackNet)
        const MAXITER = 20000
        const BATCHSIZE = 5

        for epoch = 1:MAXITER
            loss = 0
            for i = 1:size(X)[2]
                # every single sample
                outputs = forward(net, X[:, i])
                loss += net.loss_func(outputs, Y[i])
                backward(net, outputs, Y[i])

                if i % BATCHSIZE == 0
                    batch_update(net, eta=0.03)
                    #dbglog(shownet(net))
                end
            end
            batch_update(net, eta=0.03)
            #dbglog("loss $(loss[1])")
        end
    end

    function test(X, Y, net::StackNet)
        for i = 1:size(X)[2]
            # every single sample
            outputs = forward(net, X[:, i])
            dbglog("loss $(net.loss_func(outputs, Y[i]))")

            infolog(outputs[1], Y[i])
        end
    end

    X = [
    1 2 0 0 1 0 0 -1 -2 -1;
    0 0 -1 -2 -1 1 2 0 0 1
    ]

    testX = [
    2 1 -1 -2;
    -1 -2 2 1
    ]

    testY = [ 1 1 -1 -1 ]
    testY = [ 1 1 0 0 ]

    Y = [ 1 1 1 1 1 -1 -1 -1 -1 -1 ]
    Y = [ 1 1 1 1 1 0 0 0 0 0 ]

    net = StackNet(square_error)
    add_layer(net, PerceptronLayer(2, 3, tanh))
    add_layer(net, PerceptronLayer(3, 1, sigmoid))

    dbglog(shownet(net))
    test(testX, testY, net)

    train(X, Y, net)
    dbglog(shownet(net))

    test(testX, testY, net)

end

# homework main execution
function main()
    X = Array[
    [ 1.58,  2.32,  -5.8],   [ 0.67,  1.58,  -4.78],  [ 1.04,  1.01,  -3.63],  
    [-1.49,  2.18,  -3.39],  [-0.41,  1.21,  -4.73],  [1.39,  3.16,  2.87],

    [ 0.21,  0.03,  -2.21],   [ 0.37,  0.28,  -1.8],  [ 0.18,  1.22,  0.16],  
    [-0.24,  0.93,  -1.01],  [-1.18,  0.39,  -0.39],  [0.74,  0.96,  -1.16],

    [-1.54,  1.17,  0.64],   [5.41,  3.45,  -1.33],  [ 1.55,  0.99,  2.69],  
    [1.86,  3.19,  1.51],    [1.68,  1.79,  -0.87],  [3.51,  -0.22,  -1.39],

    ]

    testX = Array[
    [ 1.20,  1.40,  -1.89],  [-0.92,  1.44,  -3.22],  [ 0.45,  1.33,  -4.38],
    [-0.76,  0.84,  -1.96],
    [-0.38,  1.94,  -0.48],  [0.02,  0.72,  -0.17],  [ 0.44,  1.31,  -0.14],
    [ 0.46,  1.49,  0.68],
    [1.40,  -0.44,  -0.92],  [0.44,  0.83,  1.97],  [ 0.25,  0.68,  -0.99],
    [ 0.66,  -0.45,  0.08]
    ]

    Y = Array[
    [1, 0, 0], [1, 0, 0], [1, 0, 0],
    [1, 0, 0], [1, 0, 0], [1, 0, 0],

    [0, 1, 0], [0, 1, 0], [0, 1, 0],
    [0, 1, 0], [0, 1, 0], [0, 1, 0],

    [0, 0, 1], [0, 0, 1], [0, 0, 1],
    [0, 0, 1], [0, 0, 1], [0, 0, 1]
    ]

    testY = Array[
    [1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0],
    [0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0],
    [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]
    ]

    X = [X; testX]
    Y = [Y; testY]

    const MAXITER = 2000
    const BATCHSIZE = 2
    const ETA = 0.03
    const HIDDEN = 10

    net = StackNet(square_error)
    add_layer(net, PerceptronLayer(3, HIDDEN, tanh))
    add_layer(net, PerceptronLayer(HIDDEN, 3, sigmoid))

    function train()
        for epoch = 1:MAXITER
            loss = 0
            for i = 1:length(X)
                
                outputs = forward(net, X[i])
                loss += net.loss_func(outputs, Y[i])
                backward(net, outputs, Y[i])

                if i % BATCHSIZE == 0
                    batch_update(net, eta=ETA)
                end

            end
            batch_update(net, eta=ETA)

            dbglog("train loss: $HIDDEN, $BATCHSIZE, $ETA, $epoch, $loss")
        end
    end

    function test()
        for i = 1:length(testX)
            dbglog("------------\n test sample number: $i $(testX[i])")
            outputs = forward(net, testX[i])
            dbglog("loss $(net.loss_func(outputs, testY[i]))")
            infolog(outputs, testY[i])
        end
    end

    #dbglog(shownet(net))
    #test()

    train()

    #dbglog(shownet(net))
    #test()
end

if !isinteractive()
    main()
end


