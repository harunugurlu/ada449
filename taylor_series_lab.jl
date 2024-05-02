using ForwardDiff

function f(x)
    return x^3*sin(x) + x^5
end

function forward_diff(x::Int64, h::Float64=0.01)
    result = (f(x+h) - f(x))/h
    return result
end

fw_diff_approx = forward_diff(0)

function backward_diff(x::Int64, h::Float64=0.01)
    result = (f(x) - f(x-h))/h
    return result
end

bw_diff_approx = backward_diff(0)

exact_value = ForwardDiff.derivative(f, 0)

error_forward_diff = abs(fw_diff_approx - exact_value)
println("error_forward_diff: $error_forward_diff")
error_backward_diff = abs(bw_diff_approx - exact_value)
println("error_backward_diff: $error_backward_diff")

