using LinearAlgebra, Zygote, ForwardDiff, Printf




#### ----- ###
#### Before getting started you should write your student_number in integer format
const student_number::Int64 = 28532046358  ## <---replace 0 by your student_number 
### ---- ###
## In this HW you are on your own, 


# q1

A = randn(3, 3)

A[1] = 3.0
A[2] = -1.0
A[3] = 2.0
A[4] = -1.0
A[5] = 1.0
A[6] = 0.0
A[7] = 2.0
A[8] = 0.0
A[9] = 2.0

A

b = randn(3, 1)

b[1] = 2.0
b[2] = -2.0
b[3] = 0.0

b


function q1(x)
    return 1 / 2 * transpose(x) * A * x - transpose(b) * x
end

function penalty(x)
    return 4*max(LinearAlgebra.norm(x) - 1, 0)
end

function new_objective(x)
    return q1(x) .+ penalty(x)
end

# norm(X) <= 1

lr = 0.1

x_matrix = randn(3, 1)

x_matrix

x_matrix_old = x_matrix

for i in 1:100
    grad = Zygote.jacobian(new_objective, x_matrix)
    x_matrix .-= lr * grad[1]'
end

x_matrix
x_matrix_old

norm(x_matrix)

result = q1(x_matrix)
result
# q3
function q3(x)
    return x - (cos(x))^2
end
function q3_der(x)
    return 1 - 2 * cos(x) * (-sin(x))
end

test = x -> cos(x)^2

test2 = x -> cos(x)
test3 = x -> sin(x)
test4 = x -> sin(x)^2
test5 = x -> cos(x)^2

test2(45)
test3(45)
result = test4(45) + test(45)

ForwardDiff.derivative(test5, 45)

x_init = 1.0
x = x_init
for i in 1:5
    println(i)
    first_der = ForwardDiff.derivative(q3, x)
    second_der = ForwardDiff.derivative(q3_der, x)
    update = first_der / second_der
    x -= update
end

x

# No need to run below.
if abspath(PROGRAM_FILE) == @__FILE__
    @assert student_number != 0
    println("Seems everything is ok!!!")
end
