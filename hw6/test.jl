using Pkg
using DataFrames
using CSV
using Statistics
using Flux

train_file = "C:/Users/harun/Projects/julia/hw6/test/train.csv"
test_file = "C:/Users/harun/Projects/julia/hw6/test/test.csv"

train_data = DataFrame(CSV.File(train_file))
test_data = DataFrame(CSV.File(test_file))

# Display the first few rows of the data
println("First few rows of train data:")
first(train_data, 5) |> println
println("First few rows of test data:")
first(test_data, 5) |> println

# Check for missing values in each column
missing_train = sum(ismissing, eachcol(train_data))
missing_test = sum(ismissing, eachcol(test_data))

println("Missing values in train data:")
println(missing_train)
println("Missing values in test data:")
println(missing_test)

# Because there is no missing value we can continue

# Separate features and target variable for training data
X_train = select(train_data, Not(:price_range)) |> Matrix
y_train = train_data.price_range |> Vector

# Use all columns for test data as features
X_test = Matrix(test_data)

# Normalize the data (if necessary)
X_train = (X_train .- mean(X_train, dims=1)) ./ std(X_train, dims=1)
X_test = (X_test .- mean(X_test, dims=1)) ./ std(X_test, dims=1)

# Convert data to Float32
X_train = Float32.(X_train)
X_test = Float32.(X_test)

# Define the model
model = Chain(
    Dense(size(X_train, 2), 64, relu),
    Dense(64, 32, relu),
    Dense(32, 4),
    softmax
)

# Define the loss function and optimizer
loss_fn = Flux.Losses.crossentropy
opt = ADAM(0.001)

# Convert the labels to one-hot encoding
y_train_onehot = Flux.onehotbatch(y_train, unique(y_train))

# Function to ensure each sample is reshaped correctly
function reshape_sample(x)
    return reshape(x, (size(x, 1), 1))  # Ensure each sample is a column vector
end

# Function to compute accuracy
function accuracy(model, X, y)
    y_pred = argmax(model(X), dims=1)
    y_true = argmax(y, dims=1)
    return mean(y_pred .== y_true)
end

# Training function
function train(model, X_train, y_train, X_test, y_test, opt, epochs)
    for epoch in 1:epochs
        for i in 1:size(X_train, 2)
            x = reshape_sample(X_train[:, i])
            y_ = y_train[:, i]
            gs = gradient(() -> loss_fn(model(x), y_), Flux.params(model))
            Flux.Optimise.update!(opt, Flux.params(model), gs)
        end
        train_loss = loss_fn(model(X_train), y_train)
        test_loss = loss_fn(model(X_test), y_test)
        test_acc = accuracy(model, X_test, y_test)
        println("Epoch $epoch: Train Loss = $train_loss, Test Loss = $test_loss, Test Accuracy = $test_acc")
    end
end

# Ensure X_train and X_test are transposed to have samples as columns
X_train = X_train'
X_test = X_test'

# Convert y_test to one-hot encoding for accuracy calculation using the same labels as y_train
y_test_onehot = Flux.onehotbatch(test_data.price_range, unique(y_train))

# Training the model
train(model, X_train, y_train_onehot, X_test, y_test_onehot, opt, 10)
