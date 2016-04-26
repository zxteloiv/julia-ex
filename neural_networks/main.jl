include("stacknet.jl")

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

    net = StackNet(square_error)
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


