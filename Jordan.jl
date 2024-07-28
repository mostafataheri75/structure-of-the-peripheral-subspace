using LinearAlgebra

# Function to compute the k-th generalized eigenvectors of matrix M for eigenvalue λ
function kth_generalized_eigenvectors(M, λ, k, thereshold=1e-6)
    # Compute (M - λI)
    M_shifted = M - λ * I
    
    # Compute null space of (M - λI)^k
    gen_eig_space_k = nullspace(M_shifted^k)
    
    # Compute null space of (M - λI)^(k-1)
    if k > 1
        gen_eig_space_k_minus_1 = nullspace(M_shifted^(k-1), atol=thereshold)
    else
        gen_eig_space_k_minus_1 = zeros(size(M, 1), 0)
    end

    # Find vectors in gen_eig_space_k that are not in gen_eig_space_k_minus_1
    kth_gen_eigvecs = []
    for v in eachcol(gen_eig_space_k)
        if k == 1 || norm(M_shifted^(k-1) * v) != 0
            push!(kth_gen_eigvecs, v)
        end
    end
    
    return kth_gen_eigvecs
end

# Function to find the maximum k for which there are non-trivial k-th generalized eigenvectors
function max_k_for_generalized_eigenvectors(M, λ)
    k = 1
    gen_eigvecs = kth_generalized_eigenvectors(M, λ, k)
    while size(gen_eigvecs, 1) > 0
        k += 1
        gen_eigvecs = kth_generalized_eigenvectors(M, λ, k)
    end
    return k - 1
end

# Function to check if two vectors are parallel within a given threshold
function are_parallel(v1, v2, threshold)
    # Normalize the vectors
    v1_normalized = normalize(v1)
    v2_normalized = normalize(v2)
    
    # Check if the directions are the same or opposite within the threshold
    if abs(dot(v1_normalized, v2_normalized)) >= 1 - threshold 
        return true
    else
        return false
    end
end

# Function to compute the Jordan canonical form of matrix A
function jordan_canonical_form(A, threshold=1e-6)
    # Eigenvalues
    eigenvalues = eigvals(A)

    # Find unique eigenvalues with a threshold
    threshold = 1e-3
    unique_eigenvalues = Set{eltype(eigenvalues)}()
    for val in eigenvalues
        is_unique = all(abs(val - x) > threshold for x in unique_eigenvalues)
        if is_unique
            push!(unique_eigenvalues, val)
        end
    end
    unique_eigenvalues = collect(unique_eigenvalues)
    
    # Initialize empty modal matrix P and Jordan matrix J
    P = zeros(ComplexF64, size(A))
    
    col = 1
    chains = []
    
    for λ in unique_eigenvalues
        # Algebraic multiplicity
        alg_mult = count(x -> x == λ, eigenvalues)
        # Geometric multiplicity
        eig_space = kth_generalized_eigenvectors(A, λ, 1)
        geom_mult = size(eig_space, 1)
        
        if geom_mult < alg_mult
            # Maximum k for generalized eigenvectors
            K_ge = max_k_for_generalized_eigenvectors(A, λ)
            gen_eig_space = []
            for i in 1:K_ge
                gen_eig_i = kth_generalized_eigenvectors(A, λ, i)
                push!(gen_eig_space, gen_eig_i)
            end
            temp = K_ge
            while temp > 0
                while size(gen_eig_space[temp], 1) > 0
                    chain = []
                    V = gen_eig_space[temp][1]
                    deleteat!(gen_eig_space[temp], 1)
                    push!(chain, V)
                    for j in 1:temp-1
                        V = (A - λ * I) * V
                        push!(chain, V)
                        for i in 1:size(gen_eig_space[temp-j], 1)
                            if are_parallel(gen_eig_space[temp-j][i], V, 1e-6)
                                deleteat!(gen_eig_space[temp-j], i)
                                break
                            end
                        end
                    end
                    push!(chains, chain)
                end
                temp -= 1
            end
        else
            for i in 1:geom_mult
                chain = [eig_space[i]]
                push!(chains, chain) 
            end
        end
    end
    
    for chain in chains
        for j in 1:size(chain, 1)
            P[:, col] = chain[j]
            col += 1
        end
    end
    
    return P
end