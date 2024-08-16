using Zygote

function objective(x, y)
    return x + y
end

function constraint(x, y)
    return x^2 + y^2 == 1
end

function project(x, y)
    return (x/sqrt(x^2 + y^2), y/sqrt(x^2 + y^2))
end

lr = 0.1

function converge(objective::Function, iterations, lr=lr)
    x = 0.1
    y = 0.1

    x_cpy = copy(x)
    y_cpy = copy(y)

    for iter in 1:iterations
        grad = Zygote.gradient(objective, x, y)
        x_cpy = x_cpy - lr * grad[1]
        y_cpy = y_cpy - lr * grad[2]
        if !constraint(x, y)
            x_cpy, y_cpy = project(x_cpy, y_cpy)
        end
    end

    return (x_cpy, y_cpy)
end

points = converge(objective, 1)
println(points)

using ForwardDiff

f(x) = x^3 - x

slope = ForwardDiff.derivative(f, 2)

(3/4)^4 - (3/4)^3 + (3/4)^4 - (3/4)^3

function f(x,y,z)
    return (1-x^3)^4 + 10*(1-y^2)^2 +(x*z)^2 + 10
end
using ForwardDiff

# Define the function
function f(vars)
    x, y, z = vars
    return (1 - x^3)^4 + 10 * (1 - y^2)^2 + (x * z)^2 + 10
end

# Define the gradient function using ForwardDiff
gradient_f = x -> ForwardDiff.gradient(f, x)

# Gradient descent implementation
function gradient_descent(f, gradient_f, initial_guess; lr = 0.01, n_iter = 10000, tol = 1e-6)
    x = initial_guess
    for i in 1:n_iter
        grad = gradient_f(x)
        x_new = x - lr * grad
        x = x_new
    end
    return x, f(x)
end

# Initial guess
initial_guess = [2.0, 4.0, 2.0]

# Perform gradient descent
optimal_point, optimal_value = gradient_descent(f, gradient_f, initial_guess, lr = 0.01)

println("Optimal value: $optimal_value")
println("Optimal point: $optimal_point")


function q8(x,y)
    return x^2 + y^2 -2*x + 1
end 

grad = Zygote.gradient(q8, 2,0)

using Optim
using Pkg

Pkg.add("Optim")

# Define the function
function f(vars)
    x, y, z = vars
    return (1 - x^3)^4 + 10 * (1 - y^2)^2 + (x * z)^2 + 10
end

# Initial guess
initial_guess = [9.0, 23.0, 0.0]

# Perform optimization
result = Optim.optimize(f, initial_guess)

# Get the optimal value and point
optimal_value = Optim.minimum(result)
optimal_point = Optim.minimizer(result)

println("Optimal value: $optimal_value")
println("Optimal point: $optimal_point")
