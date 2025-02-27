
include("Peripheral_subspace_structure.jl")
using Plots
using BenchmarkTools



# Function to create the tensor product of n identity matrices
function kron_identity(n)
    if n == 0
        return 1
        
    elseif  n == 1
        return I(2)
    else
        return kron(I(2), kron_identity(n-1))
    end
end

# Function to create the X, Y, Z operators for n qubits
function create_operators(n)
    X = [0 1; 1 0]
    Y = [0 -im; im 0]
    Z = [1 0; 0 -1]
    
    X_ops = [kron(kron_identity(i-1), kron(X, kron_identity(n-i))) for i in 1:n]
    Y_ops = [kron(kron_identity(i-1), kron(Y, kron_identity(n-i))) for i in 1:n]
    Z_ops = [kron(kron_identity(i-1), kron(Z, kron_identity(n-i))) for i in 1:n]
    
    return sum(X_ops), sum(Y_ops), sum(Z_ops)
end

# Function to create the Kraus operators
function kraus_operators_collective(n)
    X_sum, Y_sum, Z_sum = create_operators(n)
    
    E_x = (1 / sqrt(3)) * exp(1im * X_sum)
    E_y = (1 / sqrt(3)) * exp(1im * Y_sum)
    E_z = (1 / sqrt(3)) * exp(1im * Z_sum)
    
    return [E_x, E_y, E_z]
end
kraus_ops= kraus_operators_collective(4);

@btime P_C, P,basis,U_matrix=structure_of_peripheral_subspace(kraus_ops,1e-4,false);
@btime P_C, P,basis,U_matrix=structure_of_fixed_subspace(kraus_ops,1e-4,false);

#633.918 ms (44652 allocations: 651.39 MiB)
# 172.361 ms (37698 allocations: 444.80 MiB)