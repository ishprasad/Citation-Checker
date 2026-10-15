# Review: Real-Time Krylov Theory for Quantum Computing Algorithms

Field: Physics: quantum computing (drawn from the machine-learning stratum) · job `759d71ca65f3` · 6 flagged citations on untouched sentences.

Mark each one in annotations.json as `genuine` (a real citation problem), `false_alarm`, or `debatable`, with a one-line note.

## s2 → 13: Quantum Computation and Quantum Information
*Reference text: full* · `.cache/fulltext/a5d771784aab36359253c31bd3c6d10a.txt`

> As one of the most prominent algorithms, quantum phase estimation (QPE) [13] resolves the core task of Hamiltonian diagonalization but necessitates a relatively high simulation cost.

Before: However, given current limitations on the assembly and control of scalable quantum computers, efficient usage of quantum resources for specific tasks [1, 2, 3, 4, 5, 6, 7, 8, 9] is considered essential in the noisy intermediate-scale quantum (NISQ) era [10, 11, 12].

- **not_found** (0.90) · escalated
  - Claim (finding): Quantum phase estimation resolves the core task of Hamiltonian diagonalization
  - Attributed: Quantum phase estimation resolves the core task of Hamiltonian diagonalization
  - Explanation: The cited work describes quantum phase estimation as an algorithm for estimating the phase of eigenvalues of unitary operators and mentions its use in Shor's factoring algorithm, but it does not state that quantum phase estimation resolves Hamiltonian diagonalization.
  - Quote: -
  - Passage (0.45, abstract): One of the most cited books in physics of all time, Quantum Computation and Quantum Information remains the best textbook in this exciting field of science. This 10th anniversary edition includes an introduction from the authors setting the work in context. This comprehensive textbook describes such remarkable effects as fast quantum algorithms, quantum teleportation, quantum cryptography and quantum error-correction. Quantum mechanics and computer science are introduced before moving on to describe what a quantum computer is, how it can be used to solve problems faster than 'classical' comput
  - Passage (0.71, Phase Estimation): Quantum algorithms are of random nature in the sense that they are able to produce correct answers only with some probabilities. Consider quantum phase estimation which provides the key to many quantum algorithms. Assume that a unitary operator U has an eigenvector |x ⟩ with eigenvalue e^2 π i φ. The phase φ of the eigenvalue is unknown and the goal of the phase estimation algorithm is to estimate φ based on the assumption that the state |x ⟩ can be prepared and the controlled-U^2^j operations [see Section § for control gate] can be performed for suitable nonnegative integers j.

The registers
  - Passage (0.65, Phase Estimation): We obtain a perfect estimate of φ.

Now we consider the case that φ cannot be expressed exactly with a b bit binary expansion. Take 0 ≤η< 2^b to be the integer that its binary fraction η/2^b=0.η1η2⋯ηb is the first b bit representation in the binary expansion of φ, which satisfies 0 ≤φ-η/2^b ≤2^-b.

Perform the inverse quantum Fourier transform on the first register given by (§), which is obtained in the first stage results, and get where amplitudes of |(η+ ) (mod 2^b) ⟩ are Assume that the result of the final measurement from phase estimation is η and dividing the result by 2^b gives φ=η/2^b. 

## s7 → 29: Circumventing the Heisenberg principle: A rigorous demonstration of filter-diagonalization on a LiCN model
*Reference text: abstract_only*

> VQPE shares the merits of variational approaches and bypasses conventional optimization procedures by solving generalized eigenvalue equations with information gathered from real-time evolution [27, 28, 29, 30], which is unitary and thereby native to quantum hardware.

Before: One representative approach is the so-called variational quantum phase estimation (VQPE) studied and developed recently [50].

- **unverifiable** (0.90) · off_topic_citation
  - Claim (finding): VQPE shares the merits of variational approaches and bypasses conventional optimization procedures by solving generalized eigenvalue equations with information gathered from real-time evolution
  - Attributed: VQPE shares the merits of variational approaches and bypasses conventional optimization procedures by solving generalized eigenvalue equations with information gathered from real-time evolution
  - Explanation: The abstract of the cited work describes filter diagonalization using short-time real-time propagation to avoid large-matrix diagonalization, but it does not mention VQPE, variational approaches, or solving generalized eigenvalue equations in that context.
  - Quote: -
  - Note: off_topic_citation
  - Passage (0.46, abstract): In a previous paper [J. Chem. Phys. 93, 2611 (1990)] a new method, filter diagonalization, was introduced for extracting highly excited rovibrational states from an arbitrary Hamiltonian, in any desired energy range. In the method, an arbitrary initial wave packet is propagated for a short time and during the propagation a ‘‘short time filter’’ of the wave packet is accumulated at various energies in any desired ‘‘window,’’ yielding a small set of functions which span the eigenfunctions of the Hamiltonian in the desired range. A small Hamiltonian matrix is then evaluated in the filtered-functi
- **unverifiable** (0.40) · off_topic_citation
  - Claim (finding): VQPE is unitary and thereby native to quantum hardware
  - Attributed: Real-time evolution is unitary and thereby native to quantum hardware.
  - Explanation: The abstract of the cited work discusses time‑dependent propagation of wave packets in a chemical physics context but does not mention the unitary nature of the evolution or its suitability for quantum hardware.
  - Quote: -
  - Note: off_topic_citation
  - Passage (0.36, abstract): In a previous paper [J. Chem. Phys. 93, 2611 (1990)] a new method, filter diagonalization, was introduced for extracting highly excited rovibrational states from an arbitrary Hamiltonian, in any desired energy range. In the method, an arbitrary initial wave packet is propagated for a short time and during the propagation a ‘‘short time filter’’ of the wave packet is accumulated at various energies in any desired ‘‘window,’’ yielding a small set of functions which span the eigenfunctions of the Hamiltonian in the desired range. A small Hamiltonian matrix is then evaluated in the filtered-functi

## s15 → 41: Hamiltonian simulation using linear combinations of unitary operations
*Reference text: full* · `.cache/fulltext/37f2f7e3a960c86be366ae40107757dd.txt`

> The creation of the state from which the flagged state can be sampled comes with a low probability of success, at least when employing a linear combination of unitaries (LCU) type of preparation, due to the low target amplitude typically limited to the initial state. [41]

Before: We note that this result does not change the analysis of unstructured search problem in regards to previous bounds established in the literature [40].

- **not_found** (0.95) · escalated
  - Claim (finding): The creation of the state from which the flagged state can be sampled comes with a low probability of success, at least when employing a linear combination of unitaries (LCU) type of preparation, due to the low target amplitude typically limited to the initial state
  - Attributed: The creation of the state from which the flagged state can be sampled comes with a low probability of success, at least when employing a linear combination of unitaries (LCU) type of preparation, due to the low target amplitude typically limited to the initial state.
  - Explanation: The cited work analyzes success probabilities for implementing linear combinations of unitary operations in the context of Hamiltonian simulation, expressing them in terms of parameters such as κ, Δ, and γ. It does not discuss 'flagged states,' 'target amplitude,' or an 'initial state' in the context of state preparation for sampling, nor does it attribute low success probability to low target amplitude limited to the initial state.
  - Quote: -
  - Passage (0.65, abstract): We present a new approach to simulating Hamiltonian dynamics based on implementing linear combinations of unitary operations rather than products of unitary operations. The resulting algorithm has superior performance to existing simulation algorithms based on product formulas and, most notably, scales better with the simulation error than any known Hamiltonian simulation technique. Our main tool is a general method to nearly deterministically implement linear combinations of nearby unitary operations, which we show is optimal among a large class of methods.
  - Passage (0.70, Optimality of the linear combination procedure): The goal of this appendix is to show that no protocol for implementing linear combinations of unitary operations in a large family of such protocols can have failure probability less than where κ is defined in main:nonunitary. Specifically, we consider protocols of the form shown in appendix:generalcircuit. Our result shows that the protocol of main:nonunitary is optimal among such all protocols in the limit of small Δ (i.e., when the unitary operations being combined are all similar).

Any protocol for implementing V=∑q=0^k Cq Uq using a circuit of the form of appendix:generalcircuit must fai
  - Passage (0.67, Adding and Subtracting Unitary Operations Using Quantum Computers): For example, the sum U1+U2+U3+U4 is implemented as (((U1+U2)+U3)+U4). Implementing the sums in A and B requires k-1 addition operations, so V can be implemented using k-1 addition operations and one subtraction operation (assuming that all the control qubits are measured to be zero at the end of the protocol).

According to main:2case, the probability of failing to implement V, given that we successfully implement A and B, is The probability of failing to perform the k-1 sums needed to construct A and B obeys where the last step follows from the fact that we can take κ ≥ 1 without loss of gene

## s17 → 42: Structured Backward Error and Condition of Generalized Eigenvalue Problems
*Reference text: full* · `.cache/fulltext/6cd9d5b716893fba8ee312eac84f087a.txt`

> In practice, we simply solve Eq. § on a truncated subspace for which sj ≥ s SV to avoid numerical instabilities [42] and filter out noise.

Before: For multi-step VQPE, fast convergence relies on a suitable time grid t. We want the real-time states |Φj ⟩ j=0^NT to be sufficiently independent in the sense that the singular values sj ∈ [0, NT + 1] of the overlap S stay bounded, for example, below by a threshold value s SV.

- **not_found** (0.98) · escalated
  - Claim (finding): sj ≥ s SV to avoid numerical instabilities
  - Attributed: In practice, one solves the equation on a truncated subspace for which singular values sj ≥ s_SV to avoid numerical instabilities.
  - Explanation: The cited work (Higham & Higham, 1998) develops structured backward discrepancy and condition numbers for generalized eigenvalue problems. It does not discuss truncating subspaces based on a singular-value threshold sj ≥ s_SV, nor does it state that such truncation avoids numerical instabilities. The claim describes a specific numerical technique used in the citing paper's VQPE method, which is not present in the cited work.
  - Quote: -
  - Passage (0.52, abstract): Backward errors and condition numbers are defined and evaluated for eigenvalues and eigenvectors of generalized eigenvalue problems. Both normwise and componentwise measures are used. Unstructured problems are considered first, and then the basic definitions are extended so that linear structure in the coefficient matrices (for example, Hermitian, Toeplitz, Hamiltonian, or band structure) is preserved by the perturbations.
  - Passage (0.50, text): The quantities derived in Theorems 4.1 and 4.2 are now upper bounds for the condition numbers; the bounds are attained for p = 1 and therefore are within a factor t of being attained for other values of p.

5. Numerical experiments. To illustrate our results we present some numerical examples. All computations were carried out in Matlab, which has unit roundoff u = 2−53 ≈1.1 × 10−16. In each example we used the QZ algorithm [23] to compute the eigensystem. Since Matlab’s implementation of the QZ algorithm does not provide left eigenvectors, we computed left eigenvectors using inverse iteration
  - Passage (0.50, text): The choice of the various parameters is summarized as follows:

• Normalization for eigenvector condition numbers: g = y (left eigenvector). • Normwise backward error η and condition numbers κ: 2-norm with E = A and F = B. • Componentwise backward error ω and condition numbers cond: E = |A| and F = |B|. • Structured backward error ω and condition number cond: g = ∥a∥2, giving “normwise” measure, and g = |a|, giving “componentwise” measure, using Σ = diag(∥x∥p) and p = 2. The results are shown in Tables 5.1 and 5.2. Several features are worth noting. For the symmetric problem we see that the co

## s19 → 47: Orthogonal Polynomials
*Reference text: full* · `.cache/fulltext/eb980557eb12eaf1774cab0738d83abf.txt`

> Let q ∈ (0,1] and now we will consider the handy choice of complex-valued Rogers-Szegő polynomials [46, 47], over the circle S^1,q = z: |z| = q^-1/2.

Before: Note that here λn = (-in) for n = En Δ t ∈ [0, 2π) so we seek a family of polynomials defined over the unit circle S^1 = z: |z| = 1 ⊂ C to bound the fraction.

- **not_found** (0.95) · escalated
  - Claim (method): Rogers-Szegő polynomials are complex-valued polynomials defined over the unit circle S^1,q = z: |z| = q^-1/2
  - Attributed: Rogers-Szegő polynomials are complex-valued polynomials defined over the circle S^1,q = {z: |z| = q^{-1/2}}.
  - Explanation: The cited work is a survey on orthogonal polynomials that explicitly states it does not discuss orthogonal polynomials on the unit circle. It mentions Szegő in connection with orthogonal polynomials on the unit circle and references Szegő's book, but it does not define or describe Rogers-Szegő polynomials, nor does it define any polynomials over a circle of radius q^{-1/2}. The claim is not present in the text.
  - Quote: -
  - Passage (0.60, V. Totik 72): with special functions, combinatorics and algebra, and it is mainly devoted to concrete orthogonal systems or hierarchies of systems such as the Jacobi, Hahn, Askey-Wilson, . . . polynomials. All the discrete polynomials and the qanalogues of classical ones belong to this theory. We will not treat this part; the interested reader can consult the three recent excellent monographs [39] by M. E. H. Ismail, [28] by W. Gautschi and [6] by G. E. Andrews, R. Askey and R. Roy. Much of the present state of the theory of orthogonal polynomials of several variables lies also close to this algebraic part 
  - Passage (0.60, V. Totik 108): It turns out that then the situation is similar to adding these discrete measures to µ0 (the new measure will also be in the same Nevai class), and considering standard orthogonality with respect to this new measure. For ecample, if r = 1, then

lim n→∞

Qn(z) qn(µ0 + µ1, z) = 1

holds uniformly on compact subsets of C \ supp(µ0 + µ1). Thus, the Sobolev orthogonal polynomials differ from those of the measure µ0, but not more than what happens when adding mass points to µ0.

In this discrete case the Qn’s satisfy a higher order recurrence relation, hence this case is also related to matrix orth
  - Passage (0.59, V. Totik 74): are the complex moments of µ. This matrix is nonsingular: if some linear combination with coefficients c0, . . . , cn−1 of the rows is zero, then the polynomial Pn−1(z) = c0 + · · · + cn−1zn−1 is orthogonal to every zj, j < n, and hence it is orthogonal to itself, i.e.,

Z

|Pn−1|2dµ =

Z

Pn−1Pn−1dµ = 0,

which implies Pn−1(z) ≡0. Thus, c0 = · · · = cn−1 = 0, which shows the nonsingularity of (σi,j). Therefore, the system (2.1) has a unique solution for the non-leading coefficients of pn(z)/κn (note that the leading coefficient is 1), and finally κn comes from normalization.

In particular, t

## s30 → 41: Hamiltonian simulation using linear combinations of unitary operations
*Reference text: full* · `.cache/fulltext/37f2f7e3a960c86be366ae40107757dd.txt`

> Childs and Wiebe in [41] provide the probability of applying linear combinations of unitary operations (LCU), thus yielding a success probability of the first iterative step, where Ex specifies the center of spectral decay.

Before: In particular, our phase cancellation intuition suggests that each IVQPE step acts as a sum of two unitary gates, I + (i)U(Δ tj), where the interfering phase changes after each iterative step.

- **overstated** (0.90)
  - Claim (finding): Childs and Wiebe provide the probability of applying linear combinations of unitary operations (LCU), thus yielding a success probability of the first iterative step, where Ex specifies the center of spectral decay
  - Attributed: Childs and Wiebe provide the probability of applying linear combinations of unitary operations (LCU), thus yielding a success probability of the first iterative step, where Ex specifies the center of spectral decay.
  - Explanation: The cited work discusses success/failure probabilities for implementing linear combinations of unitary operations, but it does not mention Ex or the center of spectral decay, so the claim adds unsupported detail.
  - Quote (verified): Our goal is to maximize the success probability, which is equivalent to maximizing K over all choices of the unitary operations A and B satisfying appendix:a3
  - Passage (0.60, abstract): We present a new approach to simulating Hamiltonian dynamics based on implementing linear combinations of unitary operations rather than products of unitary operations. The resulting algorithm has superior performance to existing simulation algorithms based on product formulas and, most notably, scales better with the simulation error than any known Hamiltonian simulation technique. Our main tool is a general method to nearly deterministically implement linear combinations of nearby unitary operations, which we show is optimal among a large class of methods.
  - Passage (0.67, Optimality of the linear combination procedure): The goal of this appendix is to show that no protocol for implementing linear combinations of unitary operations in a large family of such protocols can have failure probability less than where κ is defined in main:nonunitary. Specifically, we consider protocols of the form shown in appendix:generalcircuit. Our result shows that the protocol of main:nonunitary is optimal among such all protocols in the limit of small Δ (i.e., when the unitary operations being combined are all similar).

Any protocol for implementing V=∑q=0^k Cq Uq using a circuit of the form of appendix:generalcircuit must fai
  - Passage (0.64, Adding and Subtracting Unitary Operations Using Quantum Computers): For example, the sum U1+U2+U3+U4 is implemented as (((U1+U2)+U3)+U4). Implementing the sums in A and B requires k-1 addition operations, so V can be implemented using k-1 addition operations and one subtraction operation (assuming that all the control qubits are measured to be zero at the end of the protocol).

According to main:2case, the probability of failing to implement V, given that we successfully implement A and B, is The probability of failing to perform the k-1 sums needed to construct A and B obeys where the last step follows from the fact that we can take κ ≥ 1 without loss of gene

