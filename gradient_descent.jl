using Zygote

function f(x, y)
    return (x^2 + y - 11)^2 + (x + y^2 - 7)^2
end

function gradient_descent(f::Function)
    lr = 0.01
    max_iter = 1000
    x, y = 0.0, 0.0
    for i in 1:max_iter
        grad_x, grad_y = Zygote.gradient(f, x, y)
        x -= lr * grad_x
        y -= lr * grad_y
    end
    println("Optimized x = $x, y = $y, f(x, y) = $(f(x, y))")
end

gradient_descent(f)