using Flux
using MLDatasets
using Plots
using CSV, DataFrames
using Images
using Flux: crossentropy, onecold, onehotbatch, train!

using LinearAlgebra, Random, Statistics

Random.seed!(1)

# Load data sets

X_train_raw, y_train_raw = MLDatasets.MNIST.traindata(Float32)
X_test_raw, y_test_raw = MLDatasets.MNIST.testdata(Float32)

X_train_raw

describe(X_train_raw)

index = 1

image = X_train_raw[:, :, index]

colorview(Gray, image')

# Flatten input data

X_train = Flux.flatten(X_train_raw)

X_test = Flux.flatten(X_test_raw)


# one hot encoding

y_train = onehotbatch(y_train_raw, 0:9)

y_train_raw

y_test = onehotbatch(y_test_raw, 0:9)

# creating the model

model = Chain(
    Dense(28 * 28, 32, relu), # 28*28 = 784 input layers. 784 => 32
    Dense(32, 10), # 32 hidden layers, 10 output layers. 32 => 10
    softmax # optimizer?
)

# loss
loss(x,y) = crossentropy(model(x), y)

ps = Flux.params(model)

lr = 0.01
opt = ADAM(lr)

# train model
loss_history = []

epochs = 500

for epoch in 1:epochs
    train!(loss, ps, [(X_train, y_train)], opt)
    train_loss = loss(X_train, y_train)
    push!(loss_history, train_loss)
    println("Epoch = $epoch : Training loss = $train_loss")
end

loss_history

y_hat_raw = model(X_test)

index = 1

image = X_test_raw[:, :, index]

colorview(Gray, image')

y_hat = onecold(y_hat_raw) .- 1

y = y_test_raw

mean(y_hat .== y)