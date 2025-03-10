using LinearAlgebra
include("Jordan.jl")


"""
    structure_of_peripheral_subspace(kraus_ops::Vector{Matrix{Complex{Float64}}}, threshold::Float64=1e-4)

Compute the structure of the peripheral subspace given Kraus operators.

# Arguments
- `kraus_ops::Vector{Matrix{Complex{Float64}}}`: A vector of Kraus operator matrices.
- `threshold::Float64`: The threshold for numerical errors (default is 1e-4).

# Returns
- `central_projections::Vector{Matrix{Complex{Float64}}}`: Central projections.
- `minimal_projections::Vector{Vector{Matrix{Complex{Float64}}}}`: Minimal projections within each central projection.
"""
 function structure_of_peripheral_subspace(kraus_ops::Vector, threshold::Float64=1e-4,print_structure=true)
    d = size(kraus_ops[1], 1)
    
    # Compute the peripheral projector
    T_p = compute_peripheral_projector(kraus_ops, threshold)
    
    # Get right and left eigenvectors with eigenvalue modulus 1
    right_eigvecs_fixed, left_eigvecs_fixed = left_and_right_eigenvectors_with_eigenvalue_one(T_p, threshold)
    
    # Make  eigenvectors Hermitian and orthogonalize them
    right_eigvecs_fixed = make_hermitian_and_orthogonalize(right_eigvecs_fixed, threshold)

    left_eigvecs_fixed = make_hermitian_and_orthogonalize(left_eigvecs_fixed, threshold)
    
    # Compute the projector to the support of the subspace spanned by left eigenvectors
    left_eigvecs_fixed_support_projector= projector_to_support_of_subspace(left_eigvecs_fixed, threshold)
    
    # Project the right eigenvectors in the support basis
    A_basis = project_matrices(right_eigvecs_fixed, left_eigvecs_fixed_support_projector)
    
    # Determine the structure of the algebra
    central_projections, minimal_projections = structure_of_algebra(A_basis, threshold)
    
    # Print the number of sectors and dimensions
    if print_structure
        println("Number of Sectors K= ", size(central_projections)[1]) 
        for i in 1:size(central_projections)[1]
            println("d_$i = ", size(minimal_projections[i])[1], "\t d'_$i = ", Int(round(real(tr(minimal_projections[i][1])), digits=1)))
        end
    end 

    basis_set,U_matrix = construct_basis(A_basis, central_projections, minimal_projections, threshold)


    return central_projections, minimal_projections, basis_set,U_matrix
end





"""
    structure_of_fixed_subspace(kraus_ops::Vector{Matrix{Complex{Float64}}}, threshold::Float64=1e-4)

Compute the structure of the fixed point subspace given Kraus operators.

# Arguments
- `kraus_ops::Vector{Matrix{Complex{Float64}}}`: A vector of Kraus operator matrices.
- `threshold::Float64`: The threshold for numerical errors (default is 1e-4).

# Returns
- `central_projections::Vector{Matrix{Complex{Float64}}}`: Central projections.
- `minimal_projections::Vector{Vector{Matrix{Complex{Float64}}}}`: Minimal projections within each central projection.
"""
function structure_of_fixed_subspace(kraus_ops::Vector, threshold::Float64=1e-4,print_structure=true)
    d = size(kraus_ops[1], 1)
    d2 = d * d
    
    # Compute the super operator from the Kraus operators
    T = zeros(Complex{Float64}, d2, d2)
    
    # Sum Ti ⊗ Ti* for all Kraus operators
    for Ti in kraus_ops
        T += kron(Ti, conj(Ti))
    end
    
    # Get right and left eigenvectors with eigenvalue modulus 1
    right_eigvecs_fixed, left_eigvecs_fixed = left_and_right_eigenvectors_with_eigenvalue_one(T, threshold)
    
    # Make  eigenvectors Hermitian and orthogonalize them
    right_eigvecs_fixed = make_hermitian_and_orthogonalize(right_eigvecs_fixed, threshold)

    left_eigvecs_fixed = make_hermitian_and_orthogonalize(left_eigvecs_fixed, threshold)
    # Compute the projector to the support of the subspace spanned by left eigenvectors
    left_eigvecs_fixed_support_projector= projector_to_support_of_subspace(left_eigvecs_fixed, threshold)
    
    # Project the right eigenvectors in the support basis
    A_basis = project_matrices(right_eigvecs_fixed, left_eigvecs_fixed_support_projector)
    
    # Determine the structure of the algebra
    central_projections, minimal_projections = structure_of_algebra(A_basis, threshold)

    
    # Print the number of sectors and dimensions
    if print_structure
        println("Number of Sectors K= ", size(central_projections)[1]) 
        for i in 1:size(central_projections)[1]
            println("d_$i = ", size(minimal_projections[i])[1], "\t d'_$i = ", Int(round(real(tr(minimal_projections[i][1])), digits=1)))
        end
    end
    

    basis_set,U_matrix = construct_basis(A_basis, central_projections, minimal_projections, threshold)


    return central_projections, minimal_projections, basis_set,U_matrix
end




"""
    construct_basis(A_matrices::Vector{Matrix{Complex{Float64}}}, 
                    central_projections::Vector{Matrix{Complex{Float64}}}, 
                    minimal_projections::Vector{Vector{Matrix{Complex{Float64}}}}, 
                    threshold::Float64=1e-4)

Construct the basis vectors e_{k,i,j} for the algebra  mathcal{A} .

# Arguments
- `A_matrices::Vector{Matrix{Complex{Float64}}}`: Matrices spanning the algebra mathcal{A} .
- `central_projections::Vector{Matrix{Complex{Float64}}}`: Orthogonal minimal central projections {P_k}.
- `minimal_projections::Vector{Vector{Matrix{Complex{Float64}}}}`: Minimal projections{P_{k,i}}  for each  P_k .
- `threshold::Float64`: Numerical threshold for approximations (default is 1e-4).

# Returns
- `basis_set::Vector{Vector{Vector{Matrix{Complex{Float64}}}}}`: Basis vectors e_{k,i,j} .
- unitary_matrix: A unitary matrix (d x d) constructed for changing the basis.

"""
function construct_basis(A_matrices,central_projections,minimal_projections,threshold=1e-4,print_structure=true)
    # Number of minimal central projections
    K = length(central_projections)
    
    # Initialize d[k] and d_prime[k]
    d = [length(minimal_projections[k]) for k in 1:K]
    d_prime = [Int(round(real(tr(minimal_projections[k][1])), digits=1)) for k in 1:K]
    
    # Step 1: Compute eigenvectors for minimal projections
    eigenVectors = []
    for k in 1:K
        push!(eigenVectors, [])
        for i in 1:d[k]
            evals, evecs = eigen(minimal_projections[k][i])
            unit_eigenvecs = [evecs[:, j] for j in 1:length(evals) if abs(evals[j] - 1) < threshold]
            push!(eigenVectors[k], unit_eigenvecs)
        end
    end
    
    # Step 2: Construct matrices U^{k,1,n}
    U = []
    for k in 1:K
        push!(U, [])
        for n in 1:d[k]
            V = zeros(Complex{Float64}, d_prime[k], d_prime[k])
            found_valid = false
            while !found_valid
                for A in A_matrices
                    for i in 1:d_prime[k], j in 1:d_prime[k]
                        V[i, j] = eigenVectors[k][n][j]' * A * eigenVectors[k][1][i]
                    end
                    if norm(V) > threshold
                        found_valid = true
                        break
                    end
                end
            end
            normalize_factor = sqrt(real(tr(V' * V)))
            push!(U[k], V / normalize_factor)
        end
    end
    # Step 3: Construct basis vectors e_{k,i,j}
    basis_set = []
    unitary_matrix = []  # Collect all basis vectors to form the unitary matrix

    for k in 1:K
        push!(basis_set, [])
        for i in 1:d[k]
            push!(basis_set[k], [])
            for j in 1:d_prime[k]
                basis_vector = zeros(Complex{Float64}, size(A_matrices[1], 1))
                for m in 1:d_prime[k]
                    basis_vector += U[k][i][j, m] * eigenVectors[k][i][m]
                end
                push!(basis_set[k][i], basis_vector)
                push!(unitary_matrix, basis_vector)  # Add the basis vector to the unitary matrix

            end
        end
    end
    unitary_matrix = hcat(unitary_matrix...)  # Stack vectors as columns

    return basis_set, unitary_matrix
end





"""
    structure_of_algebra(A_matrices::Vector{Matrix{Complex{Float64}}}, threshold::Float64=1e-4)

Find the structure of the algebra generated by `A_matrices`. Returns the minimal central projections, and their decomposition into minimal projections.

# Arguments
- `A_matrices::Vector{Matrix{Complex{Float64}}}`: A vector of matrices that generate the algebra.
- `threshold::Float64`: The threshold for numerical approximations (default is 1e-4).

# Returns
- `central_projections::Vector{Matrix{Complex{Float64}}}`: A vector of central projection matrices.
- `minimal_projections::Vector{Vector{Matrix{Complex{Float64}}}}`: A vector of vectors of minimal projection matrices.
"""

function structure_of_algebra(A_matrices::Vector, threshold::Float64=1e-4)
    d = size(A_matrices[1], 1)
    I_d = Matrix{Complex{Float64}}(I, d, d)  # Identity matrix of size d
    # Compute the center of the algebra
    A_matrices=make_hermitian_and_orthogonalize(A_matrices, threshold)
    center_generators = compute_center_of_algebra(A_matrices)
    C_matrices=make_hermitian_and_orthogonalize(center_generators, threshold)
    
    # Find the minimal projections in the center
    # ######## instead of using Identity we should use the projector to support of center
    I_d=projector_to_support_of_subspace(A_matrices, threshold)
    C_matrices=project_matrices(C_matrices, I_d) # this is for the zero part in the algebra 
    #println("trace Id=",tr(I_d))
    central_projections = find_minimal_projections(I_d, C_matrices, threshold)

    # Find the minimal projections in the algebra
    minimal_projections = []
    for P in central_projections
        projections = find_minimal_projections(P, A_matrices, threshold)
        push!(minimal_projections, projections)
    end

    return central_projections, minimal_projections
end




"""
    compute_peripheral_projector(kraus_ops::Vector{Matrix}; threshold::Float64=1e-4)

Compute the peripheral projector of the superoperator from the Kraus operators.

# Arguments
- `kraus_ops::Vector{Matrix}`: A vector of Kraus operator matrices.
- `threshold::Float64`: The threshold for determining eigenvalue modulus (default is 1e-4).

# Returns
- `T_P::Matrix{Complex{Float64}}`: The peripheral projector matrix.
"""
function compute_peripheral_projector(kraus_ops::Vector, threshold::Float64=1e-4)
    d = size(kraus_ops[1], 1)  # Dimension of the Kraus operators
    d2 = d * d  # Dimension of the resulting matrix

    # Initialize T as a zero matrix of size d2 × d2
    T = zeros(Complex{Float64}, d2, d2)
    
    # Sum Ti ⊗ Ti* for all Kraus operators
    for Ti in kraus_ops
        T += kron(Ti, conj(Ti))  # Kronecker product of Ti and its complex conjugate
    end
    
    # Compute the Jordan canonical form of T
    P = jordan_canonical_form(T)  # Matrix P for Jordan decomposition
    J = inv(P) * T * P  # Jordan form J

    # Initialize S as a zero matrix of size d2 × d2
    S = zeros(Complex{Float64}, d2, d2)

    # Iterate over the diagonal elements of J
    for i in 1:d2
        if abs(J[i, i]) > 1 - threshold  # Check if the eigenvalue modulus is approximately 1
            S[i, i] = 1  # Set the corresponding element in S to 1
        end
    end

    # Compute the peripheral projector T_P
    T_P = P * S * inv(P)
    
    return T_P
end



"""
    left_and_right_eigenvectors_with_modulus_one(T::Matrix{Complex{Float64}}, threshold::Float64=1e-4)

Compute the left and right eigenvectors and eigenvalues of a matrix `T` with eigenvalues of modulus 1.

# Arguments
- `T::Matrix{Complex{Float64}}`: The input matrix.
- `threshold::Float64`: The threshold for determining eigenvalue modulus (default is 1e-4).

# Returns
- `right_eigvecs_mod1::Vector{Matrix{Complex{Float64}}}`: The right eigenvectors of `T` with eigenvalue modulus 1, reshaped to d*d matrices.
- `left_eigvecs_mod1::Vector{Matrix{Complex{Float64}}}`: The left eigenvectors of `T` with eigenvalue modulus 1, reshaped to d*d matrices.
"""
function left_and_right_eigenvectors_with_eigenvalue_one(T::Matrix{Complex{Float64}}, threshold::Float64=1e-4)
    d = Int(sqrt(size(T, 1)))
    
    # Compute right eigenvalues and eigenvectors
    right_eigen = eigen(T)
    right_eigvals = right_eigen.values
    right_eigvecs = right_eigen.vectors
    
    # Compute left eigenvalues and eigenvectors by transposing T
    left_eigen = eigen(T')
    left_eigvals = left_eigen.values
    left_eigvecs = left_eigen.vectors
    
    # Select right eigenvectors with eigenvalue modulus 1
    right_indices = findall(x -> abs(x - 1) < threshold, right_eigvals)
    right_eigvecs_mod1 = [reshape(right_eigvecs[:, i], d, d) for i in right_indices]
    
    # Select left eigenvectors with eigenvalue modulus 1
    left_indices = findall(x -> abs(x - 1) < threshold, left_eigvals)
    left_eigvecs_mod1 = [reshape(left_eigvecs[:, i], d, d) for i in left_indices]
    
    return right_eigvecs_mod1, left_eigvecs_mod1
end


"""
    make_hermitian_and_orthogonalize(A_matrices::Vector, threshold::Float64=1e-4)

Takes a set of matrices, generates a new set of Hermitian matrices `M + M'` and `M - iM'`,
and then orthogonalizes them using the Gram-Schmidt process.

# Arguments
- `A_matrices::Vector`: A vector of matrices.
- `threshold::Float64`: The threshold for determining linear independence (default is 1e-4).

# Returns
- `orthogonal_matrices::Vector`: A vector of orthogonal matrices.
"""
function make_hermitian_and_orthogonalize(A_matrices::Vector, threshold::Float64=1e-4)
    new_matrices = []

    # Generate new Hermitian matrices M + M' and i(M - M')
    for M in A_matrices
        push!(new_matrices, M + M')
        push!(new_matrices, 1im*(M - M'))
    end

    # Perform the Gram-Schmidt process to orthogonalize the new set of Hermitian matrices
    orthogonal_matrices = gram_schmidt_independent_matrices(new_matrices, threshold)
    
    return orthogonal_matrices
end



"""
    gram_schmidt_independent_matrices(A_matrices::Vector, threshold::Float64=1e-4)

Perform the Gram-Schmidt process on a set of matrices to produce a set of linearly independent matrices.

# Arguments
- `A_matrices::Vector`: A vector of matrices to be orthogonalized.
- `threshold::Float64`: The threshold for determining linear independence (default is 1e-4).

# Returns
- `independent_matrices::Vector`: A vector of linearly independent matrices.
"""
function gram_schmidt_independent_matrices(A_matrices::Vector, threshold::Float64=1e-4)
    n = length(A_matrices)
    Q_matrices = copy(A_matrices)
    independent_matrices = []

    for j in 1:n
        # Orthogonalize matrix j with respect to previous matrices
        for i in 1:(j-1)
            if maximum(abs.(Q_matrices[i]))>threshold

            # Compute the projection of Q_matrices[j] onto Q_matrices[i]
            proj_factor = tr(Q_matrices[i]' * Q_matrices[j]) / tr(Q_matrices[i]' * Q_matrices[i])
            Q_matrices[j] -= proj_factor * Q_matrices[i]
            end
        end
        # Check if the matrix is non-zero (i.e., linearly independent) using maximum absolute value
        norm_factor = maximum(abs.(Q_matrices[j]))
        if norm_factor > threshold
            # Normalize matrix j
            Q_matrices[j] /= norm_factor
            push!(independent_matrices, Q_matrices[j])
        end
    end
    
    return independent_matrices
end




"""
    projector_to_support_of_subspace(subspace_matrices::Vector{Matrix{Complex{Float64}}}, threshold::Float64=1e-4)

Compute the projector to the support of the subspace spanned by a set of Hermitian matrices.

# Arguments
- `subspace_matrices::Vector{Matrix{Complex{Float64}}}`: A vector of Hermitian matrices representing the subspaces.
- `threshold::Float64`: The threshold for numerical errors (default is 1e-4).

# Returns
- `P::Matrix{Complex{Float64}}`: The projector matrix.
- `basis::Vector{Vector{Complex{Float64}}}`: The orthogonal basis vectors.
"""
function projector_to_support_of_subspace(subspace_matrices::Vector, threshold::Float64=1e-4)
    evec_set = Set{Vector{Complex{Float64}}}()  # Set to store unique eigenvectors

    # Find all support's basis
    for f in subspace_matrices
        eval, evec = eigen(f)
        for i in 1:length(eval)
            if abs(eval[i]) > threshold
                push!(evec_set, evec[:, i])
            end
        end
    end

    # Convert set to vector for Gram-Schmidt
    evec_vector = collect(evec_set)

    # Orthogonalize the eigenvectors using Gram-Schmidt
    basis = gram_schmidt(evec_vector)

    # Initialize the projector matrix as zero matrix
    P = zeros(Complex{Float64}, size(subspace_matrices[1]))

    # Sum the projectors for the orthonormal basis vectors
    for b in basis
        P += b * adjoint(b)
    end

    return P
end

# Gram-Schmidt orthogonalization process
function gram_schmidt(V::Vector)
    U = []
    for v in V
        u = copy(v)
        for w in U
            u -= (w ⋅ v) * w
        end
        if norm(u) > 1e-10
            u /= norm(u)
            push!(U, u)
        end
    end
    return U
end



"""
    project_matrices(matrices::Vector{Matrix{Complex{Float64}}}, projector::Matrix{Complex{Float64}}, threshold::Float64=1e-4)

Compute the projected matrices by applying a projector to a set of matrices.

# Arguments
- `matrices::Vector{Matrix{Complex{Float64}}}`: A vector of matrices to be projected.
- `projector::Matrix{Complex{Float64}}`: The projector matrix.
- `threshold::Float64`: The threshold for numerical errors (default is 1e-4).

# Returns
- `projected_matrices::Vector{Matrix{Complex{Float64}}}`: The set of projected matrices.
"""
function project_matrices(matrices, projector::Matrix{Complex{Float64}})
    projected_matrices = []  # Vector to store the projected matrices

    # Compute the projected matrices
    for matrix in matrices
        projected_matrix = projector * matrix * projector
        push!(projected_matrices, projected_matrix)
    end

    return projected_matrices
end















"""
    construct_spectral_projections(A::AbstractMatrix, threshold::Float64=1e-4)

Construct the spectral projections of matrix `A`, considering eigenvalues within `threshold` as degenerate.

# Arguments
- `A::AbstractMatrix`: The input Hermitian matrix for which spectral projections are to be computed.
- `threshold::Float64`: The threshold for considering eigenvalues as degenerate.

# Returns
- `spectral_projections::Vector{Matrix{Complex{Float64}}}`: A vector of spectral projection matrices.
"""
function construct_spectral_projections(A::AbstractMatrix, threshold::Float64=1e-4)
    eigenvalues, eigenvectors = eigen(A)
    
    # Find unique eigenvalues with a threshold
    unique_eigenvalues = Set{eltype(eigenvalues)}()
    for val in eigenvalues
        is_unique = all(abs(val - x) > threshold for x in unique_eigenvalues)
        if is_unique
            push!(unique_eigenvalues, val)
        end
    end
    unique_eigenvalues = collect(unique_eigenvalues)
    
    # Number of unique eigenvalues
    n_unique = length(unique_eigenvalues)
    
    # Construct spectral projection operators
    spectral_projections = []
    
    for i in 1:n_unique
        eigenvalue = unique_eigenvalues[i]
        
        if abs(eigenvalue) > threshold
            # Find the indices of degenerate eigenvalues within the threshold
            indices = findall(x -> abs(x - eigenvalue) < threshold, eigenvalues)
            
            # Initialize the projection matrix
            P = zeros(Complex{Float64}, size(A)...)
            P_set = []
            
            for j in indices
                push!(P_set, eigenvectors[:, j])
            end
            
            # Orthogonalize the set of eigenvectors
            o_p_set = gram_schmidt(P_set)
            
            for j in o_p_set
                P += j * adjoint(j)
            end
            
            # Add the projection matrix to the list
            push!(spectral_projections, P)
        end
    end
    
    return spectral_projections
end






"""
    compute_center_of_algebra(A_matrices::Vector{Matrix{Complex{Float64}}})

Compute the center of the algebra generated by a set of matrices `A_matrices`.

# Arguments
- `A_matrices::Vector{Matrix{Complex{Float64}}}`: A vector of matrices that generate the algebra.

# Returns
- `center_generators::Vector{Matrix{Complex{Float64}}}`: A vector of matrices that generate the center of the algebra.
"""
function compute_center_of_algebra(A_matrices)
    d = size(A_matrices[1], 1)
    d2 = d^2
    I_d = Matrix{Complex{Float64}}(I, d, d)  # Identity matrix of size d
    
    # Initialize Γ as a zero matrix of size d2 × d2
    Gamma = zeros(Complex{Float64}, d2, d2)
    
    # Loop through each matrix and update Γ
    for Ai in A_matrices
        Gamma += kron(Ai' * Ai, I_d) - kron(Ai', transpose(Ai)) - kron(Ai,conj(Ai) ) + kron(I_d, conj(Ai) * transpose(Ai))
    end
    
    # Compute the null space of Γ
    B = nullspace(Gamma)
    Gamma_p = zeros(Complex{Float64}, d2, d2)
    
    # Loop through each basis element of the null space
    for Bi in eachcol(B)
        Bi_mat = reshape(Bi, d, d)
        Gamma_p += kron(Bi_mat' * Bi_mat, I_d) - kron(Bi_mat', transpose(Bi_mat)) - kron(Bi_mat,conj(Bi_mat) ) + kron(I_d, conj(Bi_mat) * transpose(Bi_mat))
    end
    
    # Compute the null space of Γ + Γ′
    C = nullspace(Gamma + Gamma_p)
    
    # Convert the basis vectors to square matrices
    center_generators = []
    for i in 1:size(C, 2)
        mat = reshape(C[:, i], d, d)
        push!(center_generators, mat)
    end
    
    # Return the set of square matrices
    return center_generators
end




"""
    reduce_projection(P::Matrix{Complex{Float64}}, A_matrices::Vector{Matrix{Complex{Float64}}}, threshold::Float64=1e-4)

Find a projection smaller than `P`.

# Arguments
- `P::Matrix{Complex{Float64}}`: The input projection matrix.
- `A_matrices::Vector{Matrix{Complex{Float64}}}`: A vector of matrices that generate the algebra.
- `threshold::Float64`: The threshold for numerical approximations (default is 1e-4).

# Returns
- `P_bar::Matrix{Complex{Float64}}`: A projection matrix smaller than `P` if `P` is not minimal, else returns `P`.
"""
function reduce_projection(P::Matrix{Complex{Float64}}, A_matrices::Vector, threshold::Float64=1e-4)
    isMinimal = true
    
    for Ai in A_matrices
        temp = P * Ai * P
        if isapprox(temp, (tr(temp) / tr(P)) * P, atol=threshold)
            continue
        else
            isMinimal = false
            spectral_projections = construct_spectral_projections(temp, threshold)
            # Choose the smallest spectral projection
            P_bar = spectral_projections[argmin([abs(tr(proj)) for proj in spectral_projections])]
            return P_bar
        
        end
    end
    
    if isMinimal
        return P
    end
end



 """
    find_one_minimal_projection(P::Matrix{Complex{Float64}}, A_matrices::Vector{Matrix{Complex{Float64}}}, threshold::Float64=1e-4)

Find one minimal projection in the range of projection `P`.

# Arguments
- `P::Matrix{Complex{Float64}}`: The input projection matrix.
- `A_matrices::Vector{Matrix{Complex{Float64}}}`: A vector of matrices that generate the algebra.
- `threshold::Float64`: The threshold for numerical approximations (default is 1e-4).

# Returns
- `Q::Matrix{Complex{Float64}}`: A minimal projection matrix.
"""
function find_one_minimal_projection(P::Matrix{Complex{Float64}}, A_matrices::Vector, threshold::Float64=1e-4)
    Q = reduce_projection(P, A_matrices, threshold)
    while !isapprox(P, Q, atol=threshold)
        P = Q
        Q = reduce_projection(P, A_matrices, threshold)
    end
    return Q
end


"""
    find_minimal_projections(P::Matrix{Complex{Float64}}, A_matrices::Vector{Matrix{Complex{Float64}}}, threshold::Float64=1e-4)

Decompose `P` into minimal projections in the algebra `A`.

# Arguments
- `P::Matrix{Complex{Float64}}`: The input projection matrix.
- `A_matrices::Vector{Matrix{Complex{Float64}}}`: A vector of matrices that generate the algebra.
- `threshold::Float64`: The threshold for numerical approximations (default is 1e-4).

# Returns
- `minimal_projections::Vector{Matrix{Complex{Float64}}}`: A vector of minimal projection matrices.
"""
function find_minimal_projections(P::Matrix{Complex{Float64}}, A_matrices::Vector, threshold::Float64=1e-4)
    minimal_projections = []
    while !iszero(P)
        Q = find_one_minimal_projection(P, A_matrices, threshold)
        push!(minimal_projections, Q)
        P -= Q
    end
    return minimal_projections
end







