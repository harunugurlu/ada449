using ForwardDiff
using Pkg

function newton_raphson(f::Function, maxiter::Int64, eps::Float64 = 1e-5)
    x_n = 4
    for k in 1:maxiter
        df = ForwardDiff.derivative(f,x_n)
        ddf = ForwardDiff.derivative(t -> ForwardDiff.derivative(f, t), x_n)
        if abs(df) < eps
            return x_n
        end
        x_n1 = x_n - df/ddf
        x_n = x_n1
    end
    return x_n
end

function y(x::Real)
    return x^3-3*x-5
end

result = newton_raphson(y, 1000, 0.3)
