include("stacknet.jl")

function train(X, Y, net::StackNet)
    for epoch = 1:20000
        loss = 0
        for i = 1:size(X)[2]
            # every single sample
            outputs = forward(net, X[:, i])
            loss += net.loss_func(outputs)

            error = backward(net, outputs, Y[i])
        end
        #dbglog("loss $(loss[1])")
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
    testY = [ 1 1 0 0 ]

    Y = [ 1 1 1 1 1 -1 -1 -1 -1 -1 ]
    Y = [ 1 1 1 1 1 0 0 0 0 0 ]


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


