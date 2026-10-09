# Review: Learning correlated noise in a 39-qubit quantum processor

Field: Computer science: machine learning · job `081c663e0eaf` · 5 flagged citations on untouched sentences (current and old pipeline).

Mark each one in annotations.json as `genuine` (a real citation problem), `false_alarm`, or `debatable`, with a one-line note.

## s11 → Huang et al., 2021: Efficient Estimation of Pauli Observables by Derandomization
*Flagged by: old pipeline (da5f9e3)*

> Accordingly, such graphical models can directly instantiate tensor networks, allowing utilisation of the contraction methods currently proposed as a means of decoding syndrome measurements (Chubb, 2021; Chubb and Flammia, 2021; Huang et al., 2021; Farrelly et al., 2022).

Before: For instance, ref. (Robeva and Seigal, 2018) explores the duality between such graphical models and tensor networks.

**current** · reference text: full · `.cache/fulltext/d9ac8b9dd2a575a1efddf74ce43344c5.txt`
- **unverifiable** (0.95) · escalated
  - Claim (method): Graphical models can directly instantiate tensor networks, allowing utilisation of the contraction methods currently proposed as a means of decoding syndrome measurements
  - Attributed: Graphical models can directly instantiate tensor networks, allowing utilisation of contraction methods currently proposed as a means of decoding syndrome measurements (with (Huang et al., 2021) cited as one of several works supporting this general statement).
  - Explanation: The cited work ((Huang et al., 2021)) presents a derandomization procedure for estimating Pauli observables in variational quantum algorithms. It discusses randomized and deterministic Pauli measurement protocols, confidence bounds, and numerical experiments on molecular ground-state energy estimation. It does not mention graphical models, tensor networks, contraction methods, or syndrome measurement decoding. As one of four references cited for a general statement about graphical models and tensor networks, (Huang et al., 2021) is not a reasonable example of that statement.
  - Quote: -
  - Passage (0.47, abstract): We consider the problem of jointly estimating expectation values of many Pauli observables, a crucial subroutine in variational quantum algorithms. Starting with randomized measurements, we propose an efficient derandomization procedure that iteratively replaces random single-qubit measurements by fixed Pauli measurements; the resulting deterministic measurement procedure is guaranteed to perform at least as well as the randomized one. In particular, for estimating any L low-weight Pauli observables, a deterministic measurement on only of order log(L) copies of a quantum state suffices. In som
  - Passage (0.47, Introduction): Noisy Intermediate-Scale Quantum (NISQ) devices are becoming available [39]. Though less powerful than fully error-corrected quantum computers, NISQ devices used as coprocessors might have advantages over classical computers for solving some problems of practical interest. For example, variational algorithms using NISQ hardware have potential applications to chemistry, materials science, and optimization [38, 36, 27, 18, 19, 7, 40, 20, 3].

In a typical NISQ variational algorithm, we need to estimate expectation values for a specified set of operators O1,O2, …, OL in a quantum state ρ that can
  - Passage (0.47, Conclusion and outlook): We consider the problem of predicting many Pauli expectation values from few Pauli measurements. Derandomization [33, 1, 43] provides an efficient procedure that replaces originally randomized single-qubit Pauli measurements by specific Pauli assignments. The resulting Pauli measurements are deterministic, but inherit all advantages of a fully randomized measurement protocol. Furthermore, the derandomization procedure could accurately capture the fine-grained structure of the observables in question. Predicting molecular ground state energies based on derandomized Pauli measurements scales fav
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/d9ac8b9dd2a575a1efddf74ce43344c5.txt`
- **unverifiable** (0.95) · escalated · off_topic_citation
  - Claim (method): Graphical models can directly instantiate tensor networks, allowing utilisation of the contraction methods currently proposed as a means of decoding syndrome measurements
  - Attributed: This work is an example of contraction methods currently proposed as a means of decoding syndrome measurements.
  - Explanation: The cited work presents a derandomization procedure for efficient estimation of Pauli observables in variational quantum algorithms and does not discuss graphical models, tensor networks, syndrome measurements, or contraction methods for decoding.
  - Quote: -
  - Note: off_topic_citation
  - Passage (0.47, abstract): We consider the problem of jointly estimating expectation values of many Pauli observables, a crucial subroutine in variational quantum algorithms. Starting with randomized measurements, we propose an efficient derandomization procedure that iteratively replaces random single-qubit measurements by fixed Pauli measurements; the resulting deterministic measurement procedure is guaranteed to perform at least as well as the randomized one. In particular, for estimating any L low-weight Pauli observables, a deterministic measurement on only of order log(L) copies of a quantum state suffices. In som
  - Passage (0.48, Introduction): Noisy Intermediate-Scale Quantum (NISQ) devices are becoming available [39]. Though less powerful than fully error-corrected quantum computers, NISQ devices used as coprocessors might have advantages over classical computers for solving some problems of practical interest. For example, variational algorithms using NISQ hardware have potential applications to chemistry, materials science, and optimization [38, 36, 27, 18, 19, 7, 40, 20, 3].

In a typical NISQ variational algorithm, we need to estimate expectation values for a specified set of operators O1,O2, …, OL in a quantum state ρ that can
  - Passage (0.47, Conclusion and outlook): We consider the problem of predicting many Pauli expectation values from few Pauli measurements. Derandomization [33, 1, 43] provides an efficient procedure that replaces originally randomized single-qubit Pauli measurements by specific Pauli assignments. The resulting Pauli measurements are deterministic, but inherit all advantages of a fully randomized measurement protocol. Furthermore, the derandomization procedure could accurately capture the fine-grained structure of the observables in question. Predicting molecular ground state energies based on derandomized Pauli measurements scales fav

## s12 → Bravyi et al., 2014: Efficient algorithms for maximum likelihood decoding in the surface code
*Flagged by: current*

> We measure the logical failure rate assuming a code capacity model (i.e., no measurement and reset errors) and utilising an approximate maximum likelihood decoder (a tensor network decoder (Bravyi et al., 2014) with bond dimension 8).

Before: We fit several graphical models having increasing expressive power to these data to see how correlations affect quantum error correction.

**current** · reference text: full · `.cache/fulltext/998809748b1eaafc06f0dd0de805dd42.txt`
- **overstated** (0.95)
  - Claim (method): Approximate maximum likelihood decoder is a tensor network decoder with bond dimension 8
  - Attributed: Approximate maximum likelihood decoder is a tensor network decoder with bond dimension 8
  - Explanation: The cited work describes an approximate maximum likelihood decoder based on matrix product states (a tensor network) with bond dimension χ as a tunable parameter, but does not fix the bond dimension to 8. Therefore the claim adds a specific bond dimension value not present in the source.
  - Quote (verified): Secondly, we show how to implement MLD approximately for more general noise models using matrix product states (MPS). Our implementation has running time $O(nχ^3)$ where $χ$ is a parameter that controls the approximation precision. The key step of our algorithm, borrowed from the DMRG method, is a subroutine for contracting a tensor network on the two-dimensional grid. The subroutine uses MPS with a bond dimension $χ$ to approximate the sequence of tensors arising in the course of contraction
  - Passage (0.72, abstract): We describe two implementations of the optimal error correction algorithm known as the maximum likelihood decoder (MLD) for the 2D surface code with a noiseless syndrome extraction. First, we show how to implement MLD exactly in time $O(n^2)$, where $n$ is the number of code qubits. Our implementation uses a reduction from MLD to simulation of matchgate quantum circuits. This reduction however requires a special noise model with independent bit-flip and phase-flip errors. Secondly, we show how to implement MLD approximately for more general noise models using matrix product states (MPS). Our i
  - Passage (0.74, Abstract): We describe two implementations of the optimal error correction algorithm known as the maximum likelihood decoder (MLD) for the 2D surface code with a noiseless syndrome extraction. First, we show how to implement MLD exactly in time O(n^2), where n is the number of code qubits. Our implementation uses a reduction from MLD to simulation of matchgate quantum circuits. This reduction however requires a special noise model with independent bit-flip and phase-flip errors. Secondly, we show how to implement MLD approximately for more general noise models using matrix product states (MPS). Our imple
  - Passage (0.69, Introduction): In the case of the depolarizing noise we observed that the MPS-based decoder is far superior than the MWM decoder offering more than two orders of magnitude reduction of the logical error probability even for small values of χ. This can be attributed to the fact that the MWM decoder often fails to find the minimum weight error consistent with the syndrome since it ignores correlations between X and Z errors [32].

The rest of the paper is organized as follows. We formally define the maximum likelihood decoder, the studied noise models, and the surface code in Sections §,§, and § respectively. 
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/998809748b1eaafc06f0dd0de805dd42.txt`
- **supported** (0.95)
  - Claim (method): A tensor network decoder with bond dimension 8 is an approximate maximum likelihood decoder
  - Attributed: A tensor network decoder with bond dimension 8 is an approximate maximum likelihood decoder.
  - Explanation: The cited work explicitly describes an approximate implementation of the maximum likelihood decoder using matrix product states (a tensor network) with bond dimension χ, confirming that such a decoder is an approximate MLD.
  - Quote (verified): Secondly, we show how to implement MLD approximately for more general noise models using matrix product states (MPS). Our implementation has running time O(nχ^3) where χ is a parameter that controls the approximation precision. The key step of our algorithm, borrowed from the DMRG method, is a subroutine for contracting a tensor network on the two-dimensional grid. The subroutine uses MPS with a bond dimension χ to approximate the sequence of tensors arising in the course of contraction
  - Passage (0.76, abstract): We describe two implementations of the optimal error correction algorithm known as the maximum likelihood decoder (MLD) for the 2D surface code with a noiseless syndrome extraction. First, we show how to implement MLD exactly in time $O(n^2)$, where $n$ is the number of code qubits. Our implementation uses a reduction from MLD to simulation of matchgate quantum circuits. This reduction however requires a special noise model with independent bit-flip and phase-flip errors. Secondly, we show how to implement MLD approximately for more general noise models using matrix product states (MPS). Our i
  - Passage (0.78, Abstract): We describe two implementations of the optimal error correction algorithm known as the maximum likelihood decoder (MLD) for the 2D surface code with a noiseless syndrome extraction. First, we show how to implement MLD exactly in time O(n^2), where n is the number of code qubits. Our implementation uses a reduction from MLD to simulation of matchgate quantum circuits. This reduction however requires a special noise model with independent bit-flip and phase-flip errors. Secondly, we show how to implement MLD approximately for more general noise models using matrix product states (MPS). Our imple
  - Passage (0.74, Introduction): In the case of the depolarizing noise we observed that the MPS-based decoder is far superior than the MWM decoder offering more than two orders of magnitude reduction of the logical error probability even for small values of χ. This can be attributed to the fact that the MWM decoder often fails to find the minimum weight error consistent with the syndrome since it ignores correlations between X and Z errors [32].

The rest of the paper is organized as follows. We formally define the maximum likelihood decoder, the studied noise models, and the surface code in Sections §,§, and § respectively. 

## s21 → Acharya et al., 2023: Suppressing quantum errors by scaling a surface code logical qubit
*Flagged by: current, old pipeline (da5f9e3)*

> Including such correlations is in fact necessary to obtain near-optimal performance (Acharya et al., 2023; McEwen et al., 2021).

Before: Thus, these correlations can inform a decoder of the changed average error probabilities, improving average performance.

**current** · reference text: full · `.cache/fulltext/0230a04bdb987561b6e5d08307b55fe4.txt`
- **overstated** (0.85)
  - Claim (finding): Including such correlations is necessary to obtain near-optimal performance
  - Attributed: Including such correlations is necessary to obtain near-optimal performance.
  - Explanation: The cited work discusses that pairwise correlations provide fine‑grained information about discrepancy types, but it does not state that including these correlations is required to achieve near‑optimal performance.
  - Quote (verified): We next examine pairwise correlations between detection events, which give us fine-grained information about which types of error are occurring during error correction
  - Passage (0.40, abstract): Abstract Practical quantum computing will require error rates well below those achievable with physical qubits. Quantum error correction 1,2 offers a path to algorithmically relevant error rates by encoding logical qubits within many physical qubits, for which increasing the number of physical qubits enhances protection against physical errors. However, introducing more qubits also increases the number of error sources, so the density of errors must be sufficiently low for logical performance to improve with increasing code size. Here we report the measurement of logical qubit performance scal
  - Passage (0.46, Understanding errors through correlations): We next examine pairwise correlations between detection events, which give us fine-grained information about which types of error are occurring during error correction. Figure 2a illustrates a few examples of pairwise detections that are generated by X or Z errors in the surface code. Measurement and reset errors are detected by the same stabilizer in two consecutive cycles, which we classify as a timelike pair. Data qubits may experience an X (Z) error while idling during measurement that is detected by its neighbouring Z (X) stabilizers in the same cycle, forming a spacelike pair. Errors dur
  - Passage (0.43, Algorithmically relevant error rates): Even as known error sources are suppressed in future devices, new dominant error mechanisms may arise as lower logical error rates are realized. To test the behaviour of codes with substantially lower error rates, we use the bit-flip repetition code, a one-dimensional version of the surface code. The bit-flip repetition code does not correct for phase-flip errors and is thus unsuitable for quantum algorithms. However, correcting only bit-flip errors allows it to achieve much lower logical error probabilities.

Without post-selection, we achieve a logical error per cycle of (1.7 ± 0.3) × 10−6 u
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/f78a589f928f39d12e0a7ec5327b8adc.txt`
- **not_found** (0.90) · escalated
  - Claim (finding): Including such correlations is necessary to obtain near-optimal performance
  - Attributed: Including correlations in decoding is necessary to achieve near-optimal performance.
  - Explanation: The cited work discusses pairwise correlations between detection events and describes two decoders (belief-matching and maximum-likelihood) that both utilize correlation information within a discrepancy hypergraph. However, it does not state that including correlations is necessary for near-optimal performance, nor does it provide evidence comparing decoders with and without correlations to establish such necessity.
  - Quote: -
  - Passage (0.40, abstract): Abstract Practical quantum computing will require error rates well below those achievable with physical qubits. Quantum error correction 1,2 offers a path to algorithmically relevant error rates by encoding logical qubits within many physical qubits, for which increasing the number of physical qubits enhances protection against physical errors. However, introducing more qubits also increases the number of error sources, so the density of errors must be sufficiently low for logical performance to improve with increasing code size. Here we report the measurement of logical qubit performance scal
  - Passage (0.53, Setting the prior distribution): This is most common in one- and two-body correlations, which may be contained in many three- or four-body correlations. To account for this, we enforce a simple averaged depolarizing T1/T2 noise probability floor for one-body and spacelike two-body correlations to avoid unphysically small (or even negative) probabilities.

Because we use device-level data to calibrate the decoder, we must be careful not to fit the parameters of the decoder to the specific data that must be decoded. This can happen e.g. when optimizing minimum-weight perfect matching edge weights using gradient descent with the
  - Passage (0.52, Correlated minimum-weight perfect matching): For simulations involving larger scans through parameter space (e.g. for error budgeting), we employ a minimum-weight perfect matching decoder that uses a variant of the two-pass correlation strategy detailed in (authorFowler and G, n.d.). Because the time cost of this decoder is essentially twice the time cost of minimum-weight perfect matching, we can leverage a fast custom matching engine to perform these larger scans quickly. Matching is also used to decode the repetition code, which does not require a two-pass strategy.

## s37 → Arute et al., 2019: Quantum supremacy using a programmable superconducting processor
*Flagged by: current, old pipeline (da5f9e3)*

> On this processor we have observed only a modest increase in single-qubit gate errors during simultaneous operation (see fig. 2 of Ref. (Arute et al., 2019)) but significant differences are present for isolated and simultaneous readout and two-qubit gate operations.

Before: Figure §(a) shows typical qubit T1 lifetimes at their idle frequencies, and Figure §(b) plots single qubit Pauli errors for π/2 gates characterized with isolated cross entropy benchmarking (XEB), where isolated operation is defined as applying gates to a single qubit with the remaining 52 idling at their idle frequency.

**current** · reference text: full · `.cache/fulltext/4840a0ad7ebb41125afacb5d9ecbfa89.txt`
- **supported** (0.95)
  - Claim (finding): Only a modest increase in single-qubit gate errors during simultaneous operation is observed
  - Attributed: Only a modest increase in single-qubit gate errors during simultaneous operation is observed
  - Explanation: The cited work reports that simultaneous operation leads to a modest increase in discrepancy (approximately 0.003) for single-qubit gate benchmarks, which matches the claim.
  - Quote (verified): Simultaneous operation modestly increases the error, by roughly 0.003
  - Passage (0.46, abstract): The promise of quantum computers is that certain computational tasks might be executed exponentially faster on a quantum processor than on a classical processor1. A fundamental challenge is to build a high-fidelity processor capable of running quantum algorithms in an exponentially large computational space. Here we report the use of a processor with programmable superconducting qubits2–7 to create quantum states on 53 qubits, corresponding to a computational state-space of dimension 253 (about 1016). Measurements from repeated experiments sample the resulting probability distribution, which w
  - Passage (0.74, Speckle purity benchmarking (SPB)): Following each parallel XEB sequence, we measure all the qubits and extract the equivalent XEB data for each pair. Every two-qubit gate can be characterized in these four experiments, regardless of system size. The optimization and classical simulation are also efficient, as each pair can be analyzed individually.

Figure: Parallel XEB. a, Schematics of four device-wide sequences, one for each entangler layer. Black points are active qubits, colored circles are single-qubit gates, and colored lines are two-qubit gates. We cycle between single- and two-qubit gates m times. Compare to Fig. 3a, m
  - Passage (0.72, Characterizing multi-qubit readout): To assess the fidelity of multi-qubit readout, we prepare and measure 150 random classical bitstring states with 53 qubits, with 3000 trials per state. We find that 13.6% of all trials successfully identified the prepared state. We can decompose this overall fidelity in two ways. First, we plot in solid lines in Fig. § the errors for each qubit during simultaneous readout, averaged over the 150 random bitstrings. We find that the median errors increase from 0.97% for |0⟩ and 4.5% for |1⟩ in isolation, to 1.8% and 5.1% for simultaneous readout. We do not yet understand the root causes of this i
- **overstated** (0.78)
  - Claim (finding): Significant differences are present for isolated and simultaneous readout and two-qubit gate operations
  - Attributed: significant differences are present for isolated and simultaneous readout and two-qubit gate operations
  - Explanation: The cited work reports higher discrepancy rates for simultaneous versus isolated readout and two‑qubit gate operations, showing quantitative differences, but it does not characterize those differences as statistically significant or use the term 'significant'. Therefore the claim’s assertion of 'significant' differences goes beyond what the source establishes.
  - Quote (verified): We find that the median errors increase from 0.97% for |0⟩ and 4.5% for |1⟩ in isolation, to 1.8% and 5.1% for simultaneous readout
  - Passage (0.46, abstract): The promise of quantum computers is that certain computational tasks might be executed exponentially faster on a quantum processor than on a classical processor1. A fundamental challenge is to build a high-fidelity processor capable of running quantum algorithms in an exponentially large computational space. Here we report the use of a processor with programmable superconducting qubits2–7 to create quantum states on 53 qubits, corresponding to a computational state-space of dimension 253 (about 1016). Measurements from repeated experiments sample the resulting probability distribution, which w
  - Passage (0.77, Characterizing multi-qubit readout): % | §
1Q XEB e1 isolated / simultaneous | 0.13 / 0.14 | 0.15 / 0.16 | 0.05 / 0.05 | % | 3a (main) §
1Q XEB e1 purity isolated / simultaneous | 0.11 / 0.11 | 0.11 / 0.12 | 0.03 / 0.03 | % | §
2Q XEB e2 isolated / simultaneous | 0.30 / 0.60 | 0.36 / 0.62 | 0.17 / 0.24 | % | 3a (main)
2Q XEB e2c isolated / simultaneous | 0.64 / 0.89 | 0.65 / 0.93 | 0.20 / 0.26 | % | 3a (main)
2Q XEB e2c purity isolated / simultaneous | 0.59 / 0.86 | 0.62 / 0.89 | 0.20 / 0.24 | % | §
Measurement em isolated / simultaneous | 2.83 / 3.50 | 3.05 / 3.77 | 1.09 / 1.61 | % | 3a (main)
  - Passage (0.72, Characterizing multi-qubit readout): To assess the fidelity of multi-qubit readout, we prepare and measure 150 random classical bitstring states with 53 qubits, with 3000 trials per state. We find that 13.6% of all trials successfully identified the prepared state. We can decompose this overall fidelity in two ways. First, we plot in solid lines in Fig. § the errors for each qubit during simultaneous readout, averaged over the 150 random bitstrings. We find that the median errors increase from 0.97% for |0⟩ and 4.5% for |1⟩ in isolation, to 1.8% and 5.1% for simultaneous readout. We do not yet understand the root causes of this i
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/4840a0ad7ebb41125afacb5d9ecbfa89.txt`
- **not_found** (0.85) · escalated
  - Claim (finding): single-qubit gate errors during simultaneous operation are only modestly increased
  - Attributed: The cited work ((Arute et al., 2019)) reports that single-qubit gate errors increase only modestly during simultaneous operation.
  - Explanation: The cited work includes a figure caption stating that it shows the typical distribution of single-qubit gate benchmarking discrepancies for both isolated and simultaneous operation, but the provided text does not contain any explicit statement or quantification characterizing the increase during simultaneous operation as 'modest'. The specific claim about a modest increase is not supported by any passage in the full text.
  - Quote: -
  - Passage (0.43, abstract): The promise of quantum computers is that certain computational tasks might be executed exponentially faster on a quantum processor than on a classical processor1. A fundamental challenge is to build a high-fidelity processor capable of running quantum algorithms in an exponentially large computational space. Here we report the use of a processor with programmable superconducting qubits2–7 to create quantum states on 53 qubits, corresponding to a computational state-space of dimension 253 (about 1016). Measurements from repeated experiments sample the resulting probability distribution, which w
  - Passage (0.73, Speckle purity benchmarking (SPB)): Following each parallel XEB sequence, we measure all the qubits and extract the equivalent XEB data for each pair. Every two-qubit gate can be characterized in these four experiments, regardless of system size. The optimization and classical simulation are also efficient, as each pair can be analyzed individually.

Figure: Parallel XEB. a, Schematics of four device-wide sequences, one for each entangler layer. Black points are active qubits, colored circles are single-qubit gates, and colored lines are two-qubit gates. We cycle between single- and two-qubit gates m times. Compare to Fig. 3a, m
  - Passage (0.69, Characterizing multi-qubit readout): To assess the fidelity of multi-qubit readout, we prepare and measure 150 random classical bitstring states with 53 qubits, with 3000 trials per state. We find that 13.6% of all trials successfully identified the prepared state. We can decompose this overall fidelity in two ways. First, we plot in solid lines in Fig. § the errors for each qubit during simultaneous readout, averaged over the 150 random bitstrings. We find that the median errors increase from 0.97% for |0⟩ and 4.5% for |1⟩ in isolation, to 1.8% and 5.1% for simultaneous readout. We do not yet understand the root causes of this i
- **overstated** (0.85)
  - Claim (finding): significant differences are present for isolated and simultaneous readout and two-qubit gate operations
  - Attributed: significant differences are present for isolated and simultaneous readout and two-qubit gate operations
  - Explanation: The cited work reports that simultaneous operation only modestly increases discrepancy (by about 0.003) and notes increases in readout discrepancy without describing them as significant; thus the claim overstates the magnitude of the differences.
  - Quote (verified): Simultaneous operation modestly increases the error, by roughly 0.003
  - Passage (0.46, abstract): The promise of quantum computers is that certain computational tasks might be executed exponentially faster on a quantum processor than on a classical processor1. A fundamental challenge is to build a high-fidelity processor capable of running quantum algorithms in an exponentially large computational space. Here we report the use of a processor with programmable superconducting qubits2–7 to create quantum states on 53 qubits, corresponding to a computational state-space of dimension 253 (about 1016). Measurements from repeated experiments sample the resulting probability distribution, which w
  - Passage (0.69, Characterizing multi-qubit readout): To assess the fidelity of multi-qubit readout, we prepare and measure 150 random classical bitstring states with 53 qubits, with 3000 trials per state. We find that 13.6% of all trials successfully identified the prepared state. We can decompose this overall fidelity in two ways. First, we plot in solid lines in Fig. § the errors for each qubit during simultaneous readout, averaged over the 150 random bitstrings. We find that the median errors increase from 0.97% for |0⟩ and 4.5% for |1⟩ in isolation, to 1.8% and 5.1% for simultaneous readout. We do not yet understand the root causes of this i
  - Passage (0.67, Speckle purity benchmarking (SPB)): Following each parallel XEB sequence, we measure all the qubits and extract the equivalent XEB data for each pair. Every two-qubit gate can be characterized in these four experiments, regardless of system size. The optimization and classical simulation are also efficient, as each pair can be analyzed individually.

Figure: Parallel XEB. a, Schematics of four device-wide sequences, one for each entangler layer. Black points are active qubits, colored circles are single-qubit gates, and colored lines are two-qubit gates. We cycle between single- and two-qubit gates m times. Compare to Fig. 3a, m

## s45 → Nielsen, 2002: A simple formula for the average gate fidelity of a quantum dynamical operation
*Flagged by: current*

> When the group is the full Clifford group, then the noise becomes a quantum depolarizing channel with the same fidelity as the original channel (Nielsen, 2002), this is one of the essential ingredients of randomized benchmarking (Magesan et al., 2011).

Before: If the group in question is the Pauli group, then the noise being twirled, when averaged over repeated randomised sequences, is the equivalent to a Pauli noise channel, i.e. the noise with all of its coherence removed.

**current** · reference text: full · `.cache/fulltext/595327540f4f43030594b7146e81c845.txt`
- **not_found** (0.95) · escalated
  - Claim (finding): The noise becomes a quantum depolarizing channel with the same fidelity as the original channel
  - Attributed: When the group is the full Clifford group, the noise becomes a quantum depolarizing channel with the same fidelity as the original channel.
  - Explanation: The cited work proves that twirling over the full unitary group (Haar measure) yields a depolarizing channel that preserves average fidelity, but it does not mention the Clifford group or demonstrate that twirling over the Clifford group alone suffices for this result.
  - Quote: -
  - Passage (0.69, 03.67.-a,03.65.-w,89.70.+c): (Footnote: Our definition is a special case of (Schumacher, 1996), which also considered non-maximally entangled states of RQ.)

M., P., and R. Horodecki have presented a beautiful formula (Horodecki et al., 1999) connecting F( E) to F e( E): We now give a proof of Eq. (§), substantially simplifying the proof in (Horodecki et al., 1999). The first step is to define a new, “twirled” operation ET, ET(ρ) ≡ ∫ dU U^ E(Uρ U^) U, where the integral is over the normalized uniform (Haar) measure dU on the space of d × d unitary matrices. Note that ET is a trace-preserving quantum operation. Next, we ar
  - Passage (0.69, 03.67.-a,03.65.-w,89.70.+c): Characterizing the quality of quantum channels and quantum gates is a central task of quantum computation and quantum information (Nielsen and Chuang, 2000). The purpose of this note is to present a simple formula for the average fidelity of a quantum channel or quantum gate.

The average fidelity of a quantum channel described by a trace-preserving quantum operation E (Nielsen and Chuang, 2000) is defined by where the integral is over the uniform (Haar) measure dψ on state space, normalized so ∫ dψ = 1. We assume E acts on a qudit, that is, a d-dimensional quantum system, with d finite. We us
  - Passage (0.67, 03.67.-a,03.65.-w,89.70.+c): The paper concludes with a discussion of how the formula for F( E,U) may be useful for experimentally quantifying the quality of quantum gates and quantum channels.

The present work is a development of the paper of Bowdrey et al (Bowdrey et al., 2002), who obtained a simple formula for F( E,U) when E and U act on qubits. This paper generalizes to the case where E and U act on qudits. Related results were also obtained by Fortunato et al (Fortunato et al., 2002a; Fortunato et al., 2002b) who found a simple and experimentally useful formula for the entanglement fidelity; (Fortunato et al., 2002
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/595327540f4f43030594b7146e81c845.txt`
- **supported** (0.95)
  - Claim (finding): The noise becomes a quantum depolarizing channel with the same fidelity as the original channel
  - Attributed: The noise becomes a quantum depolarizing channel with the same fidelity as the original channel
  - Explanation: The cited work demonstrates that twirling a quantum operation results in a depolarizing channel and argues that twirling does not change the average fidelity, thereby showing that the noise becomes a depolarizing channel with the same fidelity as the original channel.
  - Quote (verified): The simplification is in the next step, namely, showing that ET is a depolarizing channel. That is, there is a p such that ET(ρ) = pI/d + (1-p) ρ for all ρ
  - Passage (0.69, 03.67.-a,03.65.-w,89.70.+c): (Footnote: Our definition is a special case of (Schumacher, 1996), which also considered non-maximally entangled states of RQ.)

M., P., and R. Horodecki have presented a beautiful formula (Horodecki et al., 1999) connecting F( E) to F e( E): We now give a proof of Eq. (§), substantially simplifying the proof in (Horodecki et al., 1999). The first step is to define a new, “twirled” operation ET, ET(ρ) ∫ dU U^ E(Uρ U^) U, where the integral is over the normalized uniform (Haar) measure dU on the space of d × d unitary matrices. Note that ET is a trace-preserving quantum operation. Next, we argu
  - Passage (0.69, 03.67.-a,03.65.-w,89.70.+c): Characterizing the quality of quantum channels and quantum gates is a central task of quantum computation and quantum information (Nielsen and Chuang, 2000). The purpose of this note is to present a simple formula for the average fidelity of a quantum channel or quantum gate.

The average fidelity of a quantum channel described by a trace-preserving quantum operation E (Nielsen and Chuang, 2000) is defined by where the integral is over the uniform (Haar) measure dψ on state space, normalized so ∫ dψ = 1. We assume E acts on a qudit, that is, a d-dimensional quantum system, with d finite. We us
  - Passage (0.67, 03.67.-a,03.65.-w,89.70.+c): The paper concludes with a discussion of how the formula for F( E,U) may be useful for experimentally quantifying the quality of quantum gates and quantum channels.

The present work is a development of the paper of Bowdrey et al (Bowdrey et al., 2002), who obtained a simple formula for F( E,U) when E and U act on qubits. This paper generalizes to the case where E and U act on qudits. Related results were also obtained by Fortunato et al (Fortunato et al., 2002a; Fortunato et al., 2002b) who found a simple and experimentally useful formula for the entanglement fidelity; (Fortunato et al., 2002

