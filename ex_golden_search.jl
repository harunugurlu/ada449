function golden_search(f::Function, a::Float64, b::Float64, maxiter::Int64, eps::Float64 = 1e-5)::Tuple{Float64, Float64}
    r = (3-sqrt(5))/2
    x1 = a+r*(b-a)
    x2 = b-r*(b-a)
    f1 = f(x1)
    f2 = f(x2)
    for k in 1:maxiter
        if f1 > f2
            a = x1
            x1 = x2
            f1 = f2
            x2 = r*a+(1-r)*b
            f2 = f(x2)
        else
            b = x2
            x2 = x1
            f2 = f1
            x1 = r*b+(1-r)*a
            f1 = f(x1)
        end
        if abs(b-a) < eps
            return a,b
        end
    end
    return a,b
end

function y(x::Float64)
    return x^4-14*x^3+60*x^2-70*x
end

result = golden_search(y, 0.0, 2.0, 1000, 0.3)