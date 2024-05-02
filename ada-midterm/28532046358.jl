using LinearAlgebra
using Zygote
using ForwardDiff
using Pkg
## Your student number here!!!
const std_number::Int = 28532046358

ForwardDiff.derivative(f, 2)

ForwardDiff.derivative(test, 2)

Zygote.gradient(test, [2])

function test(x)
    return x*2
end

function f(x)
    return x*x + x*x - 2 * x + 1
end

function y(y)
    return y^2
end

println(27 / 128)

function q2(x, y)
    return (x^(1 / 2) + y + abs(x + 6))^(1 / 2)
end

function q5(x, y)
    return x^4 - x^3 + y^4 - y^3
end

function q9(x, y, z)
    return (1 - x^3)^4 + 10 * (1 - y^2)^2 + (x * z)^2 + 10
end