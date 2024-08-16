using LinearAlgebra, Zygote, ForwardDiff, Printf
using CSV, DataFrames
using StatsBase: mean, countmap
using Parameters
using Distributions
using Random
using Flux
using MLUtils
using NNlib
using HTTP.Base64
using PyCall
using HTTP
using JSON
using CodecZlib
using Tar
using CategoricalArrays
using ZipFile
using Pkg
using MLDataPattern


# Pkg.add(["HTTP", "JSON", "CSV", "CategoricalArrays", "Flux"])
# Pkg.build("PyCall")
# Pkg.add("CodecZlib")
# Pkg.add("ZipFile")
# Pkg.add("MLDataPattern")

#### ----- ###
#### Before getting started you should write your student_number in integer format
const student_number::Int64 = 28532046358  ## <---replace 0 by your student_number
### ---- ###
## In this HW you are on your own,

# Hocam path to your kaggle.json should be here.
ENV["KAGGLE_CONFIG_DIR"] = ""

function download_kaggle_dataset(owner, dataset, destination)
    kaggle_json_path = joinpath(ENV["KAGGLE_CONFIG_DIR"], "kaggle.json")
    kaggle_creds = JSON.parsefile(kaggle_json_path)

    user = kaggle_creds["username"]
    key = kaggle_creds["key"]

    url = "https://www.kaggle.com/api/v1/datasets/download/$owner/$dataset"

    headers = [
        "Content-Type" => "application/json",
        "Authorization" => "Basic $(base64encode("$user:$key"))",
        "Content-Disposition" => "attachment"
    ]

    response = HTTP.get(url, headers)

    if response.status == 200
        open(destination, "w") do f
            write(f, response.body)
        end
        println("Downloaded dataset to $destination")
    else
        println("Failed to download. Status: ", response.status)
    end
end

owner = "uciml"
dataset = "pima-indians-diabetes-database"
destination = "diabetes.zip"

download_kaggle_dataset(owner, dataset, destination)

function unzip_file(zip_path, extract_to)
    zarchive = ZipFile.Reader(zip_path)
    for file in zarchive.files
        dest = joinpath(extract_to, file.name)
        mkpath(dirname(dest))
        open(dest, "w") do out_file
            write(out_file, read(file))
        end
    end
end

unzip_file("./diabetes.zip", "./")

data = CSV.read("diabetes.csv", DataFrame)
println("First few rows of the dataset:")
println(first(data, 5))

describe(data)

# Proceed with your data preprocessing and model training as before

println("First few rows of the dataset:")
println(first(data, 5))

describe(data)

# When we inspect the data, some columns have values 0, indicating missing data.
# For example a person having 0 SkinThickness, or 0 BMI (body mass index) is impossible
# Glucose, BloodPressure, SkinThickness, Insulin, BMI colums' 0 rows should be imputed.
columns_to_impute = [:Glucose, :BloodPressure, :SkinThickness, :Insulin, :BMI]

# Converting those columns to Float64, so I can insert the mean (a floating point number)
for col in columns_to_impute
    data[!, col] = convert(Vector{Float64}, data[!, col])
end

# Replace zero values with the mean of non-zero values
for col in columns_to_impute
    non_zero_mean = mean(data[data[!, col] .!= 0, col])
    data[data[!, col] .== 0, col] .= non_zero_mean
end

# Verify that there are no zero values left in the specified columns
for col in columns_to_impute
    println("Column $col - Min Value: ", minimum(data[!, col]))
end

# Checking if the target variable is balanced
outcome_counts = countmap(data.Outcome)
println("Class distribution in the Outcome column after imputation: ", outcome_counts)
# It is imbalanced. There are 500 0s and 268 1s.

# Splitting the target column from the others
y = data[!, :Outcome]
X = select(data, Not(:Outcome))

# Normalize the numeric columns
function normalize(X::DataFrame)
    for col in names(X, Number)
        X[!, col] = (X[!, col] .- mean(X[!, col])) ./ std(X[!, col])
    end
    return X
end

X = normalize(X)

# Convert DataFrame to matrix
function dataframe_to_matrix(df::DataFrame)
    mat = hcat([Vector{Float64}(df[!, col]) for col in names(df)]...)
    return mat
end

X = dataframe_to_matrix(X)
y = convert(Vector{Float64}, y)

# Function to balance the dataset by undersampling
function undersample_data(X, y)
    class_1_indices = findall(y .== 1)
    class_0_indices = findall(y .== 0)
    
    num_samples = min(length(class_1_indices), length(class_0_indices))

    sampled_class_1_indices = sample(class_1_indices, num_samples, replace=false)
    sampled_class_0_indices = sample(class_0_indices, num_samples, replace=false)

    balanced_indices = vcat(sampled_class_1_indices, sampled_class_0_indices)

    shuffled_indices = shuffle(balanced_indices)

    return X[shuffled_indices, :], y[shuffled_indices]
end

# Balance the dataset
X_balanced, y_balanced = undersample_data(X, y)

# Set seed for reproducibility
Random.seed!(42)

# Split the data into training and testing sets
function split_data(X, y, test_ratio=0.2)
    n = size(X, 1)
    indices = shuffle(1:n)
    test_size = round(Int, test_ratio * n)
    test_indices = indices[1:test_size]
    train_indices = indices[(test_size + 1):end]
    return X[train_indices, :], y[train_indices], X[test_indices, :], y[test_indices]
end

X_train, y_train, X_test, y_test = split_data(X_balanced, y_balanced)

println("Training set size: ", size(X_train, 1))
println("Testing set size: ", size(X_test, 1))
println("Class distribution in training set: ", countmap(y_train))
println("Class distribution in testing set: ", countmap(y_test))

# Transposing the matrices to match the expected input shape for Flux. When I don't do it Flux throws an error about Dimension Mismatch
X_train = X_train'
X_test = X_test'
y_train = y_train'
y_test = y_test'

println("Size of X_train_resampled: ", size(X_train))

# Setup 1: Defining the model here
function create_model(input_size)
    model = Chain(
        Dense(input_size, 256, relu), Dropout(0.5),
        Dense(256, 128, relu), Dropout(0.5),
        Dense(128, 64, relu), Dropout(0.5),
        Dense(64, 1, σ)
    )
    return model
end
opt = Flux.Optimise.Adam(0.01)  # Adam optimizer with a learning rate of 0.01

# Setup 2: More layers and lower dropout rate
function create_model_setup1(input_size)
    model = Chain(
        Dense(input_size, 128, relu), Dropout(0.2),
        Dense(128, 64, relu), Dropout(0.2),
        Dense(64, 32, relu), Dropout(0.2),
        Dense(32, 16, relu), Dropout(0.2),
        Dense(16, 1, σ)
    )
    return model
end
opt = Flux.Optimise.Adam(0.001)  # Adam optimizer with a learning rate of 0.001

# Setup 3: Less layers and higher dropout rate
function create_model_setup2(input_size)
    model = Chain(
        Dense(input_size, 512, relu), Dropout(0.6),
        Dense(512, 256, relu), Dropout(0.6),
        Dense(256, 1, σ)
    )
    return model
end
opt = Flux.Optimise.Adam(0.01)  # Adam optimizer with a learning rate of 0.01

# Use the actual number of features from X_train
input_size = size(X_train, 1)
model = create_model_setup1(input_size)

loss_history = []
accuracy_history = []
test_accuracy_history = []

# Train the model with batch processing
function train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32)
    train_data = DataLoader((X_train, y_train), batchsize=batch_size, shuffle=true)
    test_data = DataLoader((X_test, y_test), batchsize=batch_size, shuffle=false)
    
    for epoch in 1:epochs
        epoch_loss = 0.0
        train_acc = 0.0
        num_batches = 0

        # Training phase
        trainmode!(model)
        for (x_batch, y_batch) in train_data
            y_batch = reshape(y_batch, 1, :)
            val, grads = Zygote.withgradient(Flux.params(model)) do
                y_pred = model(x_batch)
                Flux.Losses.binarycrossentropy(y_pred, y_batch)
            end
            Flux.Optimise.update!(opt, Flux.params(model), grads)
            epoch_loss += val
            y_pred_binary = model(x_batch) .> 0.5
            train_acc += mean(y_pred_binary .== y_batch)
            num_batches += 1
        end

        # Calculate average training loss and accuracy
        avg_train_loss = epoch_loss / num_batches
        avg_train_acc = train_acc / num_batches

        # Evaluation
        test_acc = 0.0
        num_test_batches = 0
        testmode!(model)
        for (x_batch, y_batch) in test_data
            y_batch = reshape(y_batch, 1, :)
            y_pred_binary = model(x_batch) .> 0.5
            test_acc += mean(y_pred_binary .== y_batch)
            num_test_batches += 1
        end

        avg_test_acc = test_acc / num_test_batches

        # Saving the history
        push!(loss_history, avg_train_loss)
        push!(accuracy_history, avg_train_acc)
        push!(test_accuracy_history, avg_test_acc)

        if epoch % 10 == 0
            println("Epoch $epoch: Loss = $avg_train_loss, Train Accuracy = $avg_train_acc, Test Accuracy = $avg_test_acc")
        end
    end
end

# Train the model
train_model(model, X_train, y_train, X_test, y_test, 100, 32)

# Plotting loss and accuracy history
plot(1:length(loss_history), loss_history, label="Training Loss", xlabel="Epoch", ylabel="Loss", title="Training Loss History")
plot(1:length(accuracy_history), accuracy_history, label="Training Accuracy", xlabel="Epoch", ylabel="Accuracy", title="Training Accuracy History")
plot(1:length(test_accuracy_history), test_accuracy_history, label="Test Accuracy", xlabel="Epoch", ylabel="Accuracy", title="Test Accuracy History")


# No need to run below.
if abspath(PROGRAM_FILE) == @__FILE__
    @assert student_number != 0
    println("Seems everything is ok!!!")
end
