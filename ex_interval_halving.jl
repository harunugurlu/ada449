function interval_halving(f::Function, a::Int64, b::Int64, maxiter::Int64, eps::Float64 = 1e-5)::Tuple{Float64, Float64}
    k = 0
    L = (b-a)
    x1 = a + (L/4)
    x2 = b - (L/4)
    alpha = (a+b)/2
    f1 = f(x1)
    f2 = f(x2)
    fa = f(alpha)
    for k in 1:maxiter
        L = (b-a)
        x1 = a + (L/4)
        x2 = b - (L/4)
        f1 = f(x1)
        f2 = f(x2)
        if f1 < fa
            b = alpha
            alpha = x1
            fa = f1
        else
            if f2 < fa
                a = alpha
                alpha = x2
                fa = f2
            else
                a = x1
                b = x2
            end
        end
        if L < eps
            return (a,b)
        end
    end
    return (a,b)
end

function y(x::Float64)
    return x^4-14*x^3+60*x^2-70*x
end

result = interval_halving(y,0,2,1000,0.3)