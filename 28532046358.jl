## Dear Comrade, in this HW you have two tasks
##You shall now implement quadratic search method for finding minima of a fucntion


### Run the following part to see if there is any package that you need to install
### if something goes wrong, watch the error message and install the needed package by 
### Pkg.add("missing_package")
using Pkg
using BSON
import Base: isapprox
using Test
### Run the following function as it is needed for comparison purposses
function isapprox((a,b)::Tuple{T, T}, (c,d)::Tuple{L, L}; rtol = 1e-2) where {T <: Real, L <: Real}
    return abs(a-c) < rtol && abs(b-d) < rtol
end
###
###  
### Scroll down for your HW
### Before getting started you should write your student_number in integer format
const student_number::Int64 =  28532046358 ## <---replace 0 by your student_number 
###
###
###

square(n::Real) = n * n

println(square(2))

function take_power(x::Real, n::Int)::Real
    result = 1 
    base = x
    for i in 1:n
        result = result * base
    end
    return result
end

println(take_power(2,3))

function means_of_determinants(x_1::Real, x_2::Real, x_3::Real, y_1::Real, y_2::Real, y_3::Real)::Tuple{Float64, Float64}
    a_1 = ((x_2 - x_3)^2 * (y_1 - y_2) + (x_3 - x_1)^2 * (y_2 - y_3) + (x_1 - x_2)^2 * (y_3 - y_1)) /
          ((x_1 - x_2) * (x_1 - x_3) * (x_2 - x_3))

    a_2 = ((x_2 - x_3) * (y_1 - y_2) + (x_3 - x_1) * (y_2 - y_3) + (x_1 - x_2) * (y_3 - y_1)) /
          ((x_1 - x_2) * (x_1 - x_3) * (x_2 - x_3))

    return (a_1, a_2)
end



function minimum_x(a_1, a_2)::Float64
    return (-a_1/(2*a_2))
end


function find_minimum_quadratic_search(f::Function, 
    x_1::Real, 
    x_3::Real;
    max_iter::Int = 100, 
    ϵ::Float64 = 1e-5)::Tuple{Float64, Float64}
    ## Beginning of the function ##
    ## Your code goes here your function should return a tuple of the form α, f(α), 
    ## where α is the point where the minimum is attained. 
    x_const = take_power(10,20)
    
    
    for i in 1:max_iter
        x_2 = (x_1 + x_3) / 2
        y_1 = f(x_1)
        y_2 = f(x_2)
        y_3 = f(x_3)
        tup = means_of_determinants(x_1, x_2, x_3, y_1, y_2, y_3)
    
        x_min = minimum_x(tup[1], tup[2])
    
        y_min = f(x_min)
        if x_1 < x_min && x_min < x_2
            if y_min <= y_2
                x_3 = x_2
                y_3 = y_2

                x_2 = x_min
                y_2 = y_min
            else
                x_1 = x_min
                y_1 = y_min
            end
        end
        
        if x_2 < x_min && x_min < x_3
            if y_min <= y_2
                x_1 = x_2
                y_1 = y_2

                x_2 = x_min
                y_2 = y_min
            else
                x_3 = x_min
                y_3 = y_min
            end
        end

        if abs(x_min - x_const) < ϵ
            return (x_min, f(x_min))
        end
        x_const = x_min
    end
    
    return (x_min, f(x_min))
end

## Before going to unit_test run the next cell see what ya doin?
find_minimum_quadratic_search(x->x^2-1, 0.0, pi)
### 

## Unit test for bisection ###
function unit_test_for_quadratic_search()
    @assert student_number != 0 "Mind your student number please!!!!"
    @assert isa(find_minimum_quadratic_search(x->x^2 , -1, 1), Tuple{Float64, Float64}) "Return type should be a tuple of Float64"
    try
        @assert isapprox(find_minimum_quadratic_search(x->x^2 -1 , -1, 1), (0.0, -1.0); rtol = 1e-2)
        @assert isapprox(find_minimum_quadratic_search(x->-sin(x), 0.0, pi), (pi/2, -1.0); rtol = 1e-2)
        @assert isapprox(find_minimum_quadratic_search(x->x^4+x^2+x, -1, 1; max_iter = 1000), (-0.3859, -0.2148); rtol = 1e-2)              
    catch AssertionError
        @info "Something went wrong buddy checkout your implementation"
        throw(AssertionError)
    end
    @info "This is it pal!!!!, you are done!!!"
    return 1
end

## Run the unit_test_for_bisection to see if your doing goood!!!
unit_test_for_quadratic_search()
###

#### Seesm that we are done here congrats comrade, you have completed this task successsssfuly.

##### No need to run anything below!!!!
if abspath(PROGRAM_FILE) == @__FILE__

    G::Int64 = unit_test_for_quadratic_search()
    dict_ = Dict("std_ID"=>student_number, "G"=>G)
    try
        BSON.@save "$(student_number).res" dict_ 
        catch Exception 
            println("something went wrong with", Exception)
    end
end