# Review: Numerical Implementation of Just-In-Time Decoding in Novel Lattice Slices Through the Three-Dimensional Surface Code

Field: Computer science: machine learning · job `fa5e07cc13a7` · 10 flagged citations on untouched sentences (current and old pipeline).

Mark each one in annotations.json as `genuine` (a real citation problem), `false_alarm`, or `debatable`, with a one-line note.

## s8 → 22: 2D quantum computation with 3D topological codes
*Flagged by: current, old pipeline (da5f9e3)*

> As noted by Bombín in [22] this need for repeated measurements results in a discrepancy between constant-time circuits in 3D topological codes and linear-time circuits in 2D topological codes: in principle these both have spacetime cost O(d^3), but in practise the 2D code will also require O(d) measurement rounds between each set of operations, resulting in a time cost of O(d^2) and a spacetime cost of O(d^4).

Before: It is believed that single-shot decoding is not possible for 2D topological codes [19] and so measurement errors must instead be counteracted by repeated rounds of stabiliser measurement, with fault-tolerance in a distance d code requiring O(d) repeats [9].

**current** · reference text: full · `.cache/fulltext/188688734f7e8f659cf3d74c87643a27.txt`
- **overstated** (0.90) · scope_inflation
  - Claim (finding): Bombín notes that repeated measurements result in a discrepancy between constant-time circuits in 3D topological codes and linear-time circuits in 2D topological codes, with spacetime cost O(d^3) in principle and O(d^4) in practice, requiring O(d) measurement rounds between each set of operations
  - Attributed: Bombín notes that repeated measurements result in a discrepancy between constant-time circuits in 3D topological codes and linear-time circuits in 2D topological codes, with spacetime cost O(d^3) in principle and O(d^4) in practice, requiring O(d) measurement rounds between each set of operations.
  - Explanation: The cited work states that 3D topological codes allow constant-time universal operations while 2D codes do not achieve universality even when extended in time, but it does not provide the specific asymptotic spacetime costs O(d^3) principle, O(d^4) practice, or the O(d) measurement rounds mentioned in the claim.
  - Quote (verified): for three spatial dimensions a universal set of topological operations can be carried out in constant time, and yet for two spatial dimensions universality cannot be achieved even if operations extend over time
  - Note: scope_inflation
  - Passage (0.58, abstract): I present a fault-tolerant quantum computing method for 2D architectures that is particularly appealing for photonic qubits. It relies on a crossover of techniques from topological stabilizer codes and measurement based quantum computation. In particular, it is based on 3D color codes and their transversal operations.
  - Passage (0.69, A dimensional puzzle): In three dimensions, by contrast, local operations, supplemented with global classical computation, are enough to achieve universal quantum computation. This is true in particular for color codes, a class of topological stabilizer codes with optimal transversality properties for every spatial dimensionality (Bombin and Martin-Delgado, 2006; Bombin and Martin-Delgado, 2007; Bombin et al., 2013; Bombin, 2015a; Anonymous, n.d.). (Footnote: Transversal gates are by no means unique to topological methods. However, in the topological context they naturally generalize to finite depth quantum circuits
  - Passage (0.65, Delayed measurements): Figure: A 2D cut of the 3D MBQC scheme with time as a dimension. Black triangles represent tetrahedra. The colored areas represent regions of the resource state at a point in time. The white one is yet to be created. The grey one is being created. The red one has been created and its ancillas measured. The blue one has been created and all its qubits measured.

As discussed in section §, time cannot be naively exchanged with one of the dimensions of the 3D MBQC scheme of section §. This is due to causal obstructions. The obstacle lies in the flow of classical information, in particular within 
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/188688734f7e8f659cf3d74c87643a27.txt`
- **not_found** (0.95) · escalated
  - Claim (finding): Bombín notes that the need for repeated measurements results in a discrepancy between constant-time circuits in 3D topological codes and linear-time circuits in 2D topological codes, with spacetime cost O(d^3) in principle and O(d^4) in practice, requiring O(d) measurement rounds between each set of operations
  - Attributed: Bombín notes that the need for repeated measurements results in a discrepancy between constant-time circuits in 3D topological codes and linear-time circuits in 2D topological codes: in principle these both have spacetime cost O(d^3), but in practise the 2D code will also require O(d) measurement rounds between each set of operations, resulting in a time cost of O(d^2) and a spacetime cost of O(d^4).
  - Explanation: The cited work discusses the dimensional puzzle that 3D topological codes allow constant-time universal operations while 2D codes do not, and introduces just-in-time (JIT) decoding as a scalable 2D scheme. However, it does not contain the specific quantitative claims about spacetime costs O(d^3) versus O(d^4), O(d) measurement rounds, O(d^2) time cost, or a characterization of 2D circuits as linear-time.
  - Quote: -
  - Passage (0.58, abstract): I present a fault-tolerant quantum computing method for 2D architectures that is particularly appealing for photonic qubits. It relies on a crossover of techniques from topological stabilizer codes and measurement based quantum computation. In particular, it is based on 3D color codes and their transversal operations.
  - Passage (0.70, A dimensional puzzle): In three dimensions, by contrast, local operations, supplemented with global classical computation, are enough to achieve universal quantum computation. This is true in particular for color codes, a class of topological stabilizer codes with optimal transversality properties for every spatial dimensionality (Bombin and Martin-Delgado, 2006; Bombin and Martin-Delgado, 2007; Bombin et al., 2013; Bombin, 2015a; Anonymous, n.d.). (Footnote: Transversal gates are by no means unique to topological methods. However, in the topological context they naturally generalize to finite depth quantum circuits
  - Passage (0.66, A dimensional puzzle): Topological error correction, originally introduced in the foundational work of Kitaev (Kitaev, 1997), is likely to play an important role in the ongoing technological race to build a quantum computer. This is particularly true for topological methods lying at the crossover with stabilizer-based approaches, a sweet spot where the versatility and simplicity of stabilizer techniques (Gottesman, 1996) joins hands with the locality and scalability of topological error correction (Dennis et al., 2002).

The qubits of a topological stabilizer code form a lattice such that (i) the quantum operations 

## s10 → 5: Topological Quantum Distillation
*Flagged by: current, old pipeline (da5f9e3)*

> This scheme was based on the 3D colour code [5, 23, 24] and the ideas presented there were translated to the 3D surface code by Brown [17], who used JIT decoding to prove a threshold for a linear-time CCZ between three 2D surface codes.

Before: In its original formulation, Bombín used the idea of JIT decoding to circumvent causal restrictions encountered when attempting to translate a (3+0)-dimensional measurement-based computing scheme to a (2+1)-dimensional one.

**current** · reference text: full · `.cache/fulltext/d83ca2cd04659f8583c3e6792361cd3f.txt`
- **not_found** (0.95) · escalated
  - Claim (method): The 3D colour code is a colour code
  - Attributed: The 3D colour code is a colour code
  - Explanation: The cited work introduces color codes on two-dimensional surfaces (closed surfaces, planar layers) and describes their properties, but it does not mention a "3D colour code" or any three-dimensional variant of color codes. Therefore, it does not support the claim that a 3D colour code is a colour code.
  - Quote: -
  - Passage (0.53, abstract): We construct a class of topological quantum codes to perform quantum entanglement distillation. These codes implement the whole Clifford group of unitary operations in a fully topological manner and without selective addressing of qubits. This allows us to extend their application also to quantum teleportation, dense coding, and computation with magic states.
  - Passage (0.69, 03.67.-a, 03.67.Lx): If its length is n and S has s generators, it will encode k=n-s qubits. Let Z be the centralizer of S in n, i.e., the set of operators in n that commute with the elements of S. The distance of the code C is the minimal length among the elements of Z not contained in S up to a sign.

Suppose that we have a 2-dimensional lattice embedded in a torus of arbitrary genus such that three links meet at each site and plaquettes can be 3-colored, see Fig. § for a example in a torus of genus one. We will take red, green and blue as colors (RGB). Notice that we can attach a color to the links in the latti
  - Passage (0.65, 03.67.-a, 03.67.Lx): We call these topological codes triangular codes. In addition, they have two virtues: 1/ there is no need for selective addressing and 2/ there is no need for braiding quasiparticles. The first property means that we do not have to address any particular qubit or set of qubits in order to implement a gate. The second one means that all we use are ground state operators, not quasiparticle excitations.

In order to achieve these goals, we shall proceed in several stages. First, we introduce a new class of topological quantum error correcting codes that we call color codes. Unlike the original to
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/d83ca2cd04659f8583c3e6792361cd3f.txt`
- **not_found** (0.95) · escalated
  - Claim (method): The 3D colour code is a colour code
  - Attributed: The 3D colour code is a colour code (i.e., the cited work introduces or describes a three-dimensional colour code).
  - Explanation: The cited work (Topological Quantum Distillation, 2007) describes color codes defined on two-dimensional lattices embedded in surfaces (e.g., torus, sphere) and their extension to triangular codes with borders. It does not mention a three-dimensional colour code or any colour code in three dimensions. The full text is substantial and on the topic of color codes, yet contains no support for the existence of a 3D colour code.
  - Quote: -
  - Passage (0.53, abstract): We construct a class of topological quantum codes to perform quantum entanglement distillation. These codes implement the whole Clifford group of unitary operations in a fully topological manner and without selective addressing of qubits. This allows us to extend their application also to quantum teleportation, dense coding, and computation with magic states.
  - Passage (0.69, 03.67.-a, 03.67.Lx): If its length is n and S has s generators, it will encode k=n-s qubits. Let Z be the centralizer of S in n, i.e., the set of operators in n that commute with the elements of S. The distance of the code C is the minimal length among the elements of Z not contained in S up to a sign.

Suppose that we have a 2-dimensional lattice embedded in a torus of arbitrary genus such that three links meet at each site and plaquettes can be 3-colored, see Fig. § for a example in a torus of genus one. We will take red, green and blue as colors (RGB). Notice that we can attach a color to the links in the latti
  - Passage (0.65, 03.67.-a, 03.67.Lx): We call these topological codes triangular codes. In addition, they have two virtues: 1/ there is no need for selective addressing and 2/ there is no need for braiding quasiparticles. The first property means that we do not have to address any particular qubit or set of qubits in order to implement a gate. The second one means that all we use are ground state operators, not quasiparticle excitations.

In order to achieve these goals, we shall proceed in several stages. First, we introduce a new class of topological quantum error correcting codes that we call color codes. Unlike the original to

## s10 → 17: A fault-tolerant non-Clifford gate for the surface code in two dimensions
*Flagged by: current*

> This scheme was based on the 3D colour code [5, 23, 24] and the ideas presented there were translated to the 3D surface code by Brown [17], who used JIT decoding to prove a threshold for a linear-time CCZ between three 2D surface codes.

Before: In its original formulation, Bombín used the idea of JIT decoding to circumvent causal restrictions encountered when attempting to translate a (3+0)-dimensional measurement-based computing scheme to a (2+1)-dimensional one.

**current** · reference text: full · `.cache/fulltext/3c0faf6d7b79b2a16d4b0e845a2e0041.txt`
- **overstated** (0.90) · scope_inflation
  - Claim (finding): Brown used JIT decoding to prove a threshold for a linear-time CCZ between three 2D surface codes
  - Attributed: Brown used JIT decoding to prove a threshold for a linear-time CCZ between three 2D surface codes.
  - Explanation: The cited work proves a threshold for a non-Clifford operation using a just-in-time decoder, but it does not specify that the operation is a linear-time CCZ between three 2D surface codes or that the threshold proof concerns exactly three 2D surface codes.
  - Quote (verified): Here, we prove that the non-Clifford operation will perform arbitrarily well as we scale the size of the system, provided that the physical error rate on the qubits is suitably low. We outline an error correction procedure as we undergo the controlled-controlled-phase operation. The argument requires two main components. We require a just-in-time decoder that controls the spread of an error during the gauge fixing
  - Note: scope_inflation
  - Passage (0.64, abstract): Fault-tolerant logic gates will consume a large proportion of the resources of a two-dimensional quantum computing architecture. Here we show how to perform a fault-tolerant non-Clifford gate with the surface code; a quantum error-correcting code now under intensive development. This alleviates the need for distillation or higher-dimensional components to complete a universal gate set. The operation uses both local transversal gates and code deformations over a time that scales with the size of the qubit array. An important component of the gate is a just-in-time decoder. These decoding algori
  - Passage (0.67, INTRODUCTION): The procedure makes essential use of just-in-time gauge fixing, a concept recently introduced by Bombín in ([16]). This enables us to recover the three-dimensional surface code model using parity measurements of weight no greater than 4. Research on such technology is presently under intensive development ([6], 7), as these are the minimal requirements to realize the surface code model.

The non-Clifford gate presented here circumvents fundamental limitations of two-dimensional models ([17]–21) by dynamically preparing a three-dimensional system using a two-dimensional array of active qubits. 
  - Passage (0.65, Just-in-time gauge fixing): We denote the failure rates of the high-performance (just-in-time) decoder P¯HP(P¯JIT). Both decay rapidly with system size below threshold, but we suppose P¯HP≪P¯JIT. In the event that the decoders disagree, we discard the state. To leading order, this occurs with likelihood ∼P¯JIT. In the case that the decoders agree, the state that we output is logically incorrect with likelihood ∼P¯HP. The use of a high-performance decoder therefore improves the fidelity of the postselected output states. The failure rate of the just-in-time decoder then only determines the rate at which magic states shoul
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/46f61173258dcfb00d9de92f59647d26.txt`
- **supported** (0.90) · escalated
  - Claim (finding): Brown used JIT decoding to prove a threshold for a linear-time CCZ between three 2D surface codes
  - Attributed: Brown used JIT decoding to prove a threshold for a linear-time CCZ between three 2D surface codes.
  - Explanation: The cited paper (Brown 2020) describes a fault-tolerant non-Clifford (controlled-controlled-phase) gate implemented with three overlapping copies of the 2D surface code, where each 2D array replicates a 3D slice over a time scaling with system size (linear time). It introduces a just-in-time (JIT) decoder for gauge fixing and proves a threshold for the gate using a renormalisation-group decoder and a constant spread discrepancy model for the JIT decoder (Lemma 2 and 3). The Discussion explicitly states they have proved a threshold for the controlled-controlled-phase gate in a 2D surface-code architecture.
  - Quote (verified): To summarise, we have shown how to perform a fault-tolerant controlled-controlled-phase gate with a

two-dimensional surface-code architecture and we have proved it has a threshold
  - Passage (0.64, abstract): Fault-tolerant logic gates will consume a large proportion of the resources of a two-dimensional quantum computing architecture. Here we show how to perform a fault-tolerant non-Clifford gate with the surface code; a quantum error-correcting code now under intensive development. This alleviates the need for distillation or higher-dimensional components to complete a universal gate set. The operation uses both local transversal gates and code deformations over a time that scales with the size of the qubit array. An important component of the gate is a just-in-time decoder. These decoding algori
  - Passage (0.67, I. INTRODUCTION): The procedure makes essential use of just-in-time gauge fixing; a concept recently introduced by Bomb´ın in Ref. [27]. This enables us to recover the three-dimensional surface-code model using parity measurements of weight no greater than four. Research into such technology is presently under intensive development [11–13], as these are the minimal require-

ments to realise the surface-code model.

The non-Clifford gate presented here circumvents fundamental limitations of two-dimensional models [28–32] by dynamically preparing a three-dimensional system using a two-dimensional array of active
  - Passage (0.67, II. RESULTS): We propose a renormalisation-group [47, 48] just-in-time decoder [27] that will defer the pairing of defects such that the spread of errors is controlled. Broadly speaking, we find that a just-in-time decoder will work if the pairing of two defects is deferred until both defects have existed for a time proportional to their separation in spacetime. We make this statement precise in Methods IV C and prove it controls the spread of errors. Moreover, we discuss how the decoder is modified to find a correction close to the boundary of the surface code.

Supposing an independent and identically dis

## s18 → 30: Dimensional jump in quantum error correction
*Flagged by: current, old pipeline (da5f9e3)*

> The ability to swap between a thin slice of 3D surface code and a 2D surface code (a process termed “dimension jumping [30]”) is at the heart of Brown's procedure.

**current** · reference text: full · `.cache/fulltext/833086ecdc3614041513c2154b113cbd.txt`
- **overstated** (0.90)
  - Claim (method): Dimension jumping is a process that swaps between a thin slice of 3D surface code and a 2D surface code
  - Attributed: Dimension jumping is a process that swaps between a thin slice of 3D surface code and a 2D surface code
  - Explanation: The cited work describes dimension jumping as a procedure that switches fault‑tolerantly between 3D and 2D color codes. It does not mention a thin slice or specifically surface codes, so the claim adds details not present in the source.
  - Quote (verified): In particular, the procedure allows to switch back and forth fault-tolerantly between 3D and 2D color codes
  - Passage (0.68, abstract): Topological stabilizer codes with different spatial dimensions have complementary properties. Here I show that the spatial dimension can be switched using gauge fixing. Combining 2D and 3D gauge color codes in a 3D qubit lattice, fault-tolerant quantum computation can be achieved with constant time overhead on the number of logical gates, up to efficient global classical computation, using only local quantum operations. Single-shot error correction plays a crucial role.
  - Passage (0.78, Introduction): This comes at a cost: spatial locality can only be attained in 4D due to two-qubit logical gates. In addition, 3D color codes require O(n^3/2) qubits to correct the same number of errors as an n-qubit 2D color code.

Dimensional jumps solve these problems, at least to a large extent. As the name suggests, in a dimensional jump the spatial dimension of a local code is switched in constant time, or more precisely via a quantum-local operation, where locality refers to a 3D layout. In particular, the procedure allows to switch back and forth fault-tolerantly between 3D and 2D color codes.

Dimens
  - Passage (0.75, Dimensional Jumps): This section contains the main results of the paper. Namely, (i) that a tetrahedral 3D gauge color code can be split into a triangular 2D color code and another 3D gauge color code with no encoded qubits and (ii) that switching fault-tolerantly back and forth between the 2D and 3D codes only requires quantum-local operations. The generalization to higher dimensions is also briefly discussed.
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/833086ecdc3614041513c2154b113cbd.txt`
- **not_found** (0.95) · escalated
  - Claim (method): Dimension jumping is a process that swaps between a thin slice of 3D surface code and a 2D surface code
  - Attributed: Dimension jumping is a process that swaps between a thin slice of 3D surface code and a 2D surface code
  - Explanation: The cited work describes dimensional jumps as a procedure to switch fault-tolerantly between 3D gauge color codes and 2D color codes (specifically tetrahedral 3D and triangular 2D color codes). It does not mention surface codes or a thin slice of a 3D surface code. The citing paper itself notes in the next sentence that Bombín studied the process for colour codes and Brown adapted it for surface codes, confirming the distinction.
  - Quote: -
  - Passage (0.68, abstract): Topological stabilizer codes with different spatial dimensions have complementary properties. Here I show that the spatial dimension can be switched using gauge fixing. Combining 2D and 3D gauge color codes in a 3D qubit lattice, fault-tolerant quantum computation can be achieved with constant time overhead on the number of logical gates, up to efficient global classical computation, using only local quantum operations. Single-shot error correction plays a crucial role.
  - Passage (0.78, Introduction): This comes at a cost: spatial locality can only be attained in 4D due to two-qubit logical gates. In addition, 3D color codes require O(n^3/2) qubits to correct the same number of errors as an n-qubit 2D color code.

Dimensional jumps solve these problems, at least to a large extent. As the name suggests, in a dimensional jump the spatial dimension of a local code is switched in constant time, or more precisely via a quantum-local operation, where locality refers to a 3D layout. In particular, the procedure allows to switch back and forth fault-tolerantly between 3D and 2D color codes.

Dimens
  - Passage (0.75, Dimensional Jumps): This section contains the main results of the paper. Namely, (i) that a tetrahedral 3D gauge color code can be split into a triangular 2D color code and another 3D gauge color code with no encoded qubits and (ii) that switching fault-tolerantly back and forth between the 2D and 3D codes only requires quantum-local operations. The generalization to higher dimensions is also briefly discussed.

## s22 → 29: Three-dimensional surface codes: Transversal gates and fault-tolerant architectures
*Flagged by: current, old pipeline (da5f9e3)*

> This restriction on the string-like logical means that as long as we ensure that its associated boundaries are on the sides of the slice and not on the top and bottom we are guaranteed logical operators with weight at least d. It is this restriction that forces at least one of the three codes to have a different time direction to the other two, since the string-like logical operators of three 3D surface codes which admit a transversal CCZ are all perpendicular [29] so there is no way to define a consistent time direction for all three (in three dimensions) such that slices in all three codes have this property.

Before: The ability to construct slices satisfying this relies on a specific property of the 3D surface code (the string-like logical operator of code is required to run between a particular pair of boundaries) which is not present in general 3D codes.

**current** · reference text: full · `.cache/fulltext/76c96fdd5b9e5df4eeb710fb75e528c2.txt`
- **not_found** (0.90) · escalated
  - Claim (finding): The string-like logical operators of three 3D surface codes which admit a transversal CCZ are all perpendicular
  - Attributed: The string-like logical operators of three 3D surface codes which admit a transversal CCZ are all perpendicular.
  - Explanation: The cited work analyzes stacks of three 3D surface codes and describes their canonical Z (string-like) logical operators as lying along the intersections of different boundary pairs in the rectified cubic lattice, but it does not explicitly state that these operators are all perpendicular to each other.
  - Quote: -
  - Passage (0.73, abstract): One of the leading quantum computing architectures is based on the two-dimensional (2D) surface code. This code has many advantageous properties such as a high error threshold and a planar layout of physical qubits where each physical qubit need only interact with its nearest neighbors. However, the transversal logical gates available in 2D surface codes are limited. This means that an additional (resource-intensive) procedure known as magic state distillation is required to do universal quantum computing with 2D surface codes. Here, we examine three-dimensional (3D) surface codes in the conte
  - Passage (0.77, Transversal CCZ): We now prove that CCZ is transversal for stacks rectified cubic codes. We first write the surface code kets in a form inspired by a proof in (Paetznick and Reichardt, 2013). Let Hc^x be the (classical) parity check matrix of the X stabilizers of SCc. That is, Hc^x is an m by n binary matrix with m equal to the number of X stabilizer generators in SCc and n equal to the number of physical qubits in the code. Each row of Hc^x has a 1 at column j if the stabilizer generator corresponding to that row acts non-trivially on qubit qj. If the stabilizer generator acts trivially then the entry is equal
  - Passage (0.75, Introduction): This relationship has implications for quantum computing with 3D surface codes and 3D color codes. For example, using the mapping between the two code families, we can use 3D surface code decoders to decode 3D color codes (Aloshious and Sarvepalli, 2018). This is useful because efficient 3D color code decoders are difficult to construct. Color codes tend to have a larger range of transversal logical gates when compared with surface codes (Bombín and Martin-Delgado, 2007a; Kubica and Beverland, 2015). This implies that we can use the relationship between surface codes and color codes to realize
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/76c96fdd5b9e5df4eeb710fb75e528c2.txt`
- **not_found** (0.90) · escalated
  - Claim (finding): The string-like logical operators of three 3D surface codes which admit a transversal CCZ are all perpendicular
  - Attributed: The string-like logical operators of three 3D surface codes which admit a transversal CCZ are all perpendicular.
  - Explanation: The cited work describes the geometry of three 3D surface codes on a rectified cubic lattice and associates each code with a spatial axis (x, y, z), but it does not explicitly state that their string-like logical operators are all perpendicular. The term "perpendicular" does not appear in the text, and no passage makes this claim directly.
  - Quote: -
  - Passage (0.73, abstract): One of the leading quantum computing architectures is based on the two-dimensional (2D) surface code. This code has many advantageous properties such as a high error threshold and a planar layout of physical qubits where each physical qubit need only interact with its nearest neighbors. However, the transversal logical gates available in 2D surface codes are limited. This means that an additional (resource-intensive) procedure known as magic state distillation is required to do universal quantum computing with 2D surface codes. Here, we examine three-dimensional (3D) surface codes in the conte
  - Passage (0.77, Transversal CCZ): We now prove that CCZ is transversal for stacks rectified cubic codes. We first write the surface code kets in a form inspired by a proof in (Paetznick and Reichardt, 2013). Let Hc^x be the (classical) parity check matrix of the X stabilizers of SCc. That is, Hc^x is an m by n binary matrix with m equal to the number of X stabilizer generators in SCc and n equal to the number of physical qubits in the code. Each row of Hc^x has a 1 at column j if the stabilizer generator corresponding to that row acts non-trivially on qubit qj. If the stabilizer generator acts trivially then the entry is equal
  - Passage (0.75, Introduction): This relationship has implications for quantum computing with 3D surface codes and 3D color codes. For example, using the mapping between the two code families, we can use 3D surface code decoders to decode 3D color codes (Aloshious and Sarvepalli, 2018). This is useful because efficient 3D color code decoders are difficult to construct. Color codes tend to have a larger range of transversal logical gates when compared with surface codes (Bombín and Martin-Delgado, 2007a; Kubica and Beverland, 2015). This implies that we can use the relationship between surface codes and color codes to realize

## s32 → 17: A fault-tolerant non-Clifford gate for the surface code in two dimensions
*Flagged by: current, old pipeline (da5f9e3)*

> Our error threshold estimate is many orders of magnitude larger than the theoretical value of ∼ 10^-17 obtained by Brown in his proof [17].

Before: We observe evidence of an error threshold in the region of pc ∼ 0.1 % for all three codes; see Fig. §.

**current** · reference text: full · `.cache/fulltext/3c0faf6d7b79b2a16d4b0e845a2e0041.txt`
- **overstated** (0.70) · number_drift
  - Claim (finding): Brown obtained a theoretical value of ∼ 10^-17
  - Attributed: Brown obtained a theoretical value of ∼ 10^-17
  - Explanation: The cited work derives a lower bound on the threshold discrepancy rate of approximately 10^−17, matching the attributed claim.
  - Quote (verified): we obtain a lower bound on the threshold error rate of ε ∼ 10−17
  - Note: number_drift
  - Passage (0.22, abstract): Fault-tolerant logic gates will consume a large proportion of the resources of a two-dimensional quantum computing architecture. Here we show how to perform a fault-tolerant non-Clifford gate with the surface code; a quantum error-correcting code now under intensive development. This alleviates the need for distillation or higher-dimensional components to complete a universal gate set. The operation uses both local transversal gates and code deformations over a time that scales with the size of the qubit array. An important component of the gate is a just-in-time decoder. These decoding algori
  - Passage (0.39, A threshold theorem with a spread error): This verifies condition 1.

The argument given above says that all errors are corrected on well-separated containers that are much smaller than the size of the system provided Dm < L/3. Given that there are no level-m + 1 errors, all the errors supported on containers of size Dm will be untethered and therefore corrected at the largest length scale. Therefore, we bound the failure probability by predicting the probability that an error of size Qm+1 occurs. Bravyi and Haah ([32]) gives a formula stating that the likelihood that a level-m chunk occurs on an L × L × L lattice is

Demanding that (
  - Passage (0.35, Just-in-time gauge fixing): We consider the permissibility of pairing u contained within Bj,α to v contained in Bk,β. For Q ≥ 33, we find that such a pairing will never be permissible before all defects are paired locally within their isolated boxes. We suppose that, without loss of generality, the diameter of Bj,α is less than or equal to the diameter of Bk,β. Given that Bj,α is separated from Bk,β by a distance greater than Qj+1/3 − 2, it will not be permissible to pair u with v within the lifetime of u before it is paired to a boundary or another defect in Bj,α provided 3(Qj + 2) ≤ Qj+1/3 − 2. This is satisfied for al
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/46f61173258dcfb00d9de92f59647d26.txt`
- **contradicted** (0.47) · escalated
  - Claim (finding): Brown obtains a theoretical error threshold of approximately 10^-17
  - Attributed: Brown's proof obtains a theoretical error threshold of approximately 10^-17.
  - Explanation: The cited work (Brown's own paper) derives a lower-bound threshold discrepancy rate of ε ∼6 × 10^−15, not ∼10^−17. The claim attributes a value of ∼10^−17 to Brown's proof, but the proof reported in the cited work yields a different, incompatible number.
  - Quote: Taking Q = 87 using s = 8 and r = 2, and we have that the number of qubits per site is N = 120 from IV A 1, we obtain a lower-bound on the threshold error rate of ε ∼6 · 10−15.
  - Passage (0.42, abstract): Fault-tolerant logic gates will consume a large proportion of the resources of a two-dimensional quantum computing architecture. Here we show how to perform a fault-tolerant non-Clifford gate with the surface code; a quantum error-correcting code now under intensive development. This alleviates the need for distillation or higher-dimensional components to complete a universal gate set. The operation uses both local transversal gates and code deformations over a time that scales with the size of the qubit array. An important component of the gate is a just-in-time decoder. These decoding algori
  - Passage (0.61, 1. Notation and terminology): Given that Dj > 2p we have that r(s + 2)Qj > 2p+1

for r = 2 at the level-(p + 1) error-correction subroutine. Therefore Error Correct(p + 1) will not find any components of E outside of the container ˜Cl,α. As such a correction will be returned entirely on ˜Cl,α, verifying condition 3.

We finally consider the support of the correction operator. If the error is tethered, the correction returned for Cj,α lies on some container ˜Ck,β with k > j to which it is tethered. In the case of untethered errors the correction for each connected component supported on Cj,α, and the correction for the smal
  - Passage (0.55, 1. Notation and terminology): In the proof given in Ref. [47] the authors make use of Lemma 1 to show that the renormalisation-group decoder will not introduce a logical failure. This is assured given that all of the errors are small and well separated in a way that is made precise by Lemma 1. With the errors of the spread error model now supported in containers as much as a factor s larger than the initial connected components of the error, the connected components are now much closer together, and in some cases overlap with one another. We have to check that the noise will not introduce a logical failure given sufficient

## s37 → 22: 2D quantum computation with 3D topological codes
*Flagged by: old pipeline (da5f9e3)*

> It is worth emphasising the significant difference between the JIT decoding scheme we have examined in this work and Bombín's original JIT decoding proposal [22].

**current** · reference text: full · `.cache/fulltext/188688734f7e8f659cf3d74c87643a27.txt`
- **supported** (0.95)
  - Claim (method): Bombín's original JIT decoding proposal is a JIT decoding scheme
  - Attributed: Bombín's original JIT decoding proposal is a JIT decoding scheme
  - Explanation: The cited work includes a section titled “Naive JIT decoder” that introduces a JIT decoder, confirming that Bombín's original proposal is indeed a JIT decoding scheme.
  - Quote (verified): This section introduces a JIT decoder that, at each step and using conventional decoders,

• estimates a portion of the X-error syndrome with the available information, and

• compensates for the difference between the current estimate and the one made in the previous step.

This ‘naive' approach is fault tolerant
  - Passage (0.33, abstract): I present a fault-tolerant quantum computing method for 2D architectures that is particularly appealing for photonic qubits. It relies on a crossover of techniques from topological stabilizer codes and measurement based quantum computation. In particular, it is based on 3D color codes and their transversal operations.
  - Passage (0.64, Naive JIT decoder): This section introduces a JIT decoder that, at each step and using conventional decoders,

• estimates a portion of the X-error syndrome with the available information, and

• compensates for the difference between the current estimate and the one made in the previous step.

This ‘naive' approach is fault tolerant.
  - Passage (0.60, The challenge): The goal of JIT decoding is to map, in real time and as information becomes available, the noisy ancilla outcomes φ to a corrected syndrome φ. Each step fixes a partial value for the syndrome φ. The input available for this task at the i-th step is For simplicity we assume that this information is used to fix The partial knowledge of the syndrome φ is then used to choose the Pauli frame fX in the subset of qubits following the prescription of appendix §. The consistency of the procedure is guaranteed if the causality condition is satisfied, i.e. where Sφi denotes the subgroup of SZ generated b
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/188688734f7e8f659cf3d74c87643a27.txt`
- **not_found** (0.95) · escalated
  - Claim (finding): The JIT decoding scheme examined in this work is significantly different from Bombín's original JIT decoding proposal
  - Attributed: The JIT decoding scheme examined in this work is significantly different from Bombín's original JIT decoding proposal.
  - Explanation: The cited work (Bombín, 2018) describes Bombín's own JIT decoding proposal in detail, but it does not mention the citing work's scheme nor make any comparison between the two. The claim of a significant difference is a comparative statement that appears only in the citing paper, not in the cited work.
  - Quote: -
  - Passage (0.55, abstract): I present a fault-tolerant quantum computing method for 2D architectures that is particularly appealing for photonic qubits. It relies on a crossover of techniques from topological stabilizer codes and measurement based quantum computation. In particular, it is based on 3D color codes and their transversal operations.
  - Passage (0.74, The challenge): The goal of JIT decoding is to map, in real time and as information becomes available, the noisy ancilla outcomes φ to a corrected syndrome φ. Each step fixes a partial value for the syndrome φ. The input available for this task at the i-th step is For simplicity we assume that this information is used to fix The partial knowledge of the syndrome φ is then used to choose the Pauli frame fX in the subset of qubits following the prescription of appendix §. The consistency of the procedure is guaranteed if the causality condition is satisfied, i.e. where Sφi denotes the subgroup of SZ generated b
  - Passage (0.73, Naive JIT decoder): This section introduces a JIT decoder that, at each step and using conventional decoders,

• estimates a portion of the X-error syndrome with the available information, and

• compensates for the difference between the current estimate and the one made in the previous step.

This ‘naive' approach is fault tolerant.

## s39 → 28: Unfolding the color code
*Flagged by: current, old pipeline (da5f9e3)*

> In light of recent results regarding transversal CCCZ gates in the 4D surface code [28, 50] it also seems natural to ask if an equivalent process could be used to construct slices of 4D surface code which allow for a linear-time CCCZ in the 3D surface code.

**current** · reference text: full · `.cache/fulltext/cc67a803696595be6d977c18a2a8d5ae.txt`
- **not_found** (0.95) · escalated
  - Claim (finding): Transversal CCCZ gates in the 4D surface code are recent results
  - Attributed: [28] reports recent results regarding transversal CCCZ gates in the 4D surface code.
  - Explanation: The cited work discusses the equivalence between the color code and toric code, transversal implementation of logical gates in the color code, and the d-qubit control-Z gate in the toric code. However, it does not mention 'transversal CCCZ gates' or the '4D surface code' (the term 'surface code' is used only for the 2D toric code with boundaries). No results about transversal CCCZ gates in the 4D surface code are present in the text.
  - Quote: -
  - Passage (0.64, abstract): The topological color code and the toric code are two leading candidates for realizing fault-tolerant quantum computation. Here we show that the color code on a d -dimensional closed manifold is equivalent to multiple decoupled copies of the d -dimensional toric code up to local unitary transformations and adding or removing ancilla qubits. Our result not only generalizes the proven equivalence for d = 2, but also provides an explicit recipe of how to decouple independent components of the color code, highlighting the importance of colorability in the construction of the code. Moreover, for th
  - Passage (0.69, Summary of main results): Note that the d-qubit control-Z gate belongs to the d-th level of the Clifford hierarchy, but is outside of the (d-1)-th level. Thus, a stack of d copies of the d-dimensional toric code saturates the bound by Bravyi and König on fault-tolerant logical gates which are implementable by local unitary transformations (Bravyi and König, 2013). For a definition of the Clifford hierarchy, see (Gottesman and Chuang, 1999; Bravyi and König, 2013; Pastawski and Yoshida, 2015)

We believe that our findings will shed light on the techniques of code deformation (Bombin and Martin-Delgado, 2009) and lattice
  - Passage (0.66, Transversal gates): We have seen that the color code is equivalent to (multiple copies of) the toric code, both in the presence or the absence of boundaries. Our findings hint that there might be non-trivial logical gates from the d-th level of the Clifford hierarchy in the d-dimensional toric code which admit fault-tolerant implementation. In this section, we show that one can implement by local unitary transformations the logical d-qubit control-Z gate on the stack of d copies of the d-dimensional toric code with point-like excitations.
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/cc67a803696595be6d977c18a2a8d5ae.txt`
- **not_found** (0.95) · escalated
  - Claim (finding): Transversal CCCZ gates in the 4D surface code are recent results
  - Attributed: There are recent results regarding transversal CCCZ gates in the 4D surface code.
  - Explanation: The cited work discusses transversal implementation of a d-qubit control-Z gate on the d-dimensional color code and its equivalence to the toric code, but it does not mention 'CCCZ gates', '4D surface code', or characterize these results as recent. The term 'surface code' is used only for the 2D toric code with boundaries.
  - Quote: -
  - Passage (0.64, abstract): The topological color code and the toric code are two leading candidates for realizing fault-tolerant quantum computation. Here we show that the color code on a d -dimensional closed manifold is equivalent to multiple decoupled copies of the d -dimensional toric code up to local unitary transformations and adding or removing ancilla qubits. Our result not only generalizes the proven equivalence for d = 2, but also provides an explicit recipe of how to decouple independent components of the color code, highlighting the importance of colorability in the construction of the code. Moreover, for th
  - Passage (0.69, Summary of main results): Note that the d-qubit control-Z gate belongs to the d-th level of the Clifford hierarchy, but is outside of the (d-1)-th level. Thus, a stack of d copies of the d-dimensional toric code saturates the bound by Bravyi and König on fault-tolerant logical gates which are implementable by local unitary transformations (Bravyi and König, 2013). For a definition of the Clifford hierarchy, see (Gottesman and Chuang, 1999; Bravyi and König, 2013; Pastawski and Yoshida, 2015)

We believe that our findings will shed light on the techniques of code deformation (Bombin and Martin-Delgado, 2009) and lattice
  - Passage (0.66, Transversal gates): We have seen that the color code is equivalent to (multiple copies of) the toric code, both in the presence or the absence of boundaries. Our findings hint that there might be non-trivial logical gates from the d-th level of the Clifford hierarchy in the d-dimensional toric code which admit fault-tolerant implementation. In this section, we show that one can implement by local unitary transformations the logical d-qubit control-Z gate on the stack of d copies of the d-dimensional toric code with point-like excitations.

## s39 → 50: Four-dimensional toric code with non-Clifford transversal gates
*Flagged by: current*

> In light of recent results regarding transversal CCCZ gates in the 4D surface code [28, 50] it also seems natural to ask if an equivalent process could be used to construct slices of 4D surface code which allow for a linear-time CCCZ in the 3D surface code.

**current** · reference text: full · `.cache/fulltext/60a6d7acb4cf593b49dc5911ff1eb452.txt`
- **supported** (0.90) · escalated
  - Claim (finding): Transversal CCCZ gates in the 4D surface code are recent results
  - Attributed: The cited work presents recent results on transversal CCCZ gates in the 4D surface code (specifically, a 4D toric code with a transversal CCCZ gate).
  - Explanation: The cited work (2021) describes a four-dimensional toric code based on the octaplex tessellation that implements a transversal CCCZ gate, which is a recent result. The citing sentence refers to 'recent results regarding transversal CCCZ gates in the 4D surface code'; the cited work's 4D toric code is analogous to a 4D surface code, and the difference in terminology does not affect the support for the claim.
  - Quote (verified): The design of a four-dimensional toric code is explored with the goal of finding a lattice capable of implementing a logical CCCZ gate transversally
  - Passage (0.78, abstract): The design of a four-dimensional toric code is explored with the goal of finding a lattice capable of implementing a logical CCCZ gate transversally.The established lattice is the octaplex tessellation, which is a regular tessellation of four-dimensional Euclidean space whose underlying 4-cell is the octaplex, or hyperdiamond.This differs from the conventional 4D toric code lattice, based on the hypercubic tessellation, which is symmetric with respect to logical X and Z and only allows for the implementation of a transversal Clifford gate.This paper further develops the established connection 
  - Passage (0.79, Motivation): In this work, we explore 4D toric codes and show a particular choice of 4D toric code that has an underlying CCCZ gate, a gate in the fourth level of the Clifford hierarchy. The underlying lattice of the code is not a simple generalization of the cubic lattice, such has a hypercubic tessellation, but a rather more exotic lattice that allows for the appropriate overlap conditions of the underlying stabilizers of the four codeblocks.

The paper is organized as follows: Sec. § is reserved for a review of prior results. In § we review the necessary algebraic conditions for CSS codes to have transv
  - Passage (0.78, Motivation): There is a rich set of literature relating topological codes to logical gates in quantum error correction. Namely, the set of transversal gates accessible to a topological code is related to the spatial dimension in which it is embedded. In the color code family, all Clifford gates are transversal for 2D color codes (Bombín and Martin-Delgado, 2006), the 3D color code has a transversal T gate (e^iπ Z/8), and more generally D-dimensional color codes have transversal e^i π Z/2^D gates (Bombín, 2015; Kubica and Beverland, 2015). It should be noted, while these higher-dimensional codes have more e
- **overstated** (0.85) · scope_inflation
  - Claim (finding): Equivalent processes could be used to construct slices of 4D surface code that allow for a linear-time CCCZ in the 3D surface code
  - Attributed: Equivalent processes could be used to construct slices of 4D surface code that allow for a linear-time CCCZ in the 3D surface code
  - Explanation: The cited work demonstrates a 4D toric code that supports a transversal CCCZ gate, but it does not discuss slicing the 4D code or achieving linear-time CCCZ in a 3D surface code. Thus the claim extends beyond what the source shows.
  - Quote (verified): The design of a four-dimensional toric code is explored with the goal of finding a lattice capable of implementing a logical CCCZ gate transversally
  - Note: scope_inflation
  - Passage (0.70, abstract): The design of a four-dimensional toric code is explored with the goal of finding a lattice capable of implementing a logical CCCZ gate transversally.The established lattice is the octaplex tessellation, which is a regular tessellation of four-dimensional Euclidean space whose underlying 4-cell is the octaplex, or hyperdiamond.This differs from the conventional 4D toric code lattice, based on the hypercubic tessellation, which is symmetric with respect to logical X and Z and only allows for the implementation of a transversal Clifford gate.This paper further develops the established connection 
  - Passage (0.74, Pauli sandwich trick): One advantage of implementing a CCCZ gate transversally is that if the resulting logical gate couples multiple logical qubits, such as the 3D toric code with periodic boundaries or in instances of Pin codes (Vuillot and Breuckmann, 2019), one may still be able to achieve a targeted CCZ gate by repeated uses of the transversal gate. The idea is to insert a logical X gate (which can always be implemented transversally in a stabilizer code) between two instances of the transversal gate implementing logical CCCZ, which results in implementing a CCZ gate on the other three logical codeblocks, see F
  - Passage (0.73, Motivation): In this work, we explore 4D toric codes and show a particular choice of 4D toric code that has an underlying CCCZ gate, a gate in the fourth level of the Clifford hierarchy. The underlying lattice of the code is not a simple generalization of the cubic lattice, such has a hypercubic tessellation, but a rather more exotic lattice that allows for the appropriate overlap conditions of the underlying stabilizers of the four codeblocks.

The paper is organized as follows: Sec. § is reserved for a review of prior results. In § we review the necessary algebraic conditions for CSS codes to have transv
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/60a6d7acb4cf593b49dc5911ff1eb452.txt`
- **supported** (0.95)
  - Claim (finding): Transversal CCCZ gates in the 4D surface code are recent results
  - Attributed: Transversal CCCZ gates in the 4D surface code are recent results.
  - Explanation: The cited work explicitly states that it explored 4D toric codes and demonstrated a choice with an underlying transversal CCCZ gate, confirming the existence of recent results on transversal CCCZ gates in the 4D surface code.
  - Quote (verified): In this work, we explore 4D toric codes and show a particular choice of 4D toric code that has an underlying CCCZ gate, a gate in the fourth level of the Clifford hierarchy
  - Passage (0.78, abstract): The design of a four-dimensional toric code is explored with the goal of finding a lattice capable of implementing a logical CCCZ gate transversally.The established lattice is the octaplex tessellation, which is a regular tessellation of four-dimensional Euclidean space whose underlying 4-cell is the octaplex, or hyperdiamond.This differs from the conventional 4D toric code lattice, based on the hypercubic tessellation, which is symmetric with respect to logical X and Z and only allows for the implementation of a transversal Clifford gate.This paper further develops the established connection 
  - Passage (0.79, Motivation): In this work, we explore 4D toric codes and show a particular choice of 4D toric code that has an underlying CCCZ gate, a gate in the fourth level of the Clifford hierarchy. The underlying lattice of the code is not a simple generalization of the cubic lattice, such has a hypercubic tessellation, but a rather more exotic lattice that allows for the appropriate overlap conditions of the underlying stabilizers of the four codeblocks.

The paper is organized as follows: Sec. § is reserved for a review of prior results. In § we review the necessary algebraic conditions for CSS codes to have transv
  - Passage (0.78, Motivation): There is a rich set of literature relating topological codes to logical gates in quantum error correction. Namely, the set of transversal gates accessible to a topological code is related to the spatial dimension in which it is embedded. In the color code family, all Clifford gates are transversal for 2D color codes (Bombín and Martin-Delgado, 2006), the 3D color code has a transversal T gate (e^iπ Z/8), and more generally D-dimensional color codes have transversal e^i π Z/2^D gates (Bombín, 2015; Kubica and Beverland, 2015). It should be noted, while these higher-dimensional codes have more e

## s40 → 29: Three-dimensional surface codes: Transversal gates and fault-tolerant architectures
*Flagged by: current, old pipeline (da5f9e3)*

> This compares unfavourably with the d^3 scaling allowed by the constant-time set of computationally universal operations in the 3D surface code [29], and so is unlikely to be advantageous.

Before: We expect that such a generalisation should be possible, but note that its spacetime overhead would scale as d^4.

**current** · reference text: full · `.cache/fulltext/76c96fdd5b9e5df4eeb710fb75e528c2.txt`
- **overstated** (0.75) · escalated · scope_inflation
  - Claim (finding): The 3D surface code allows d^3 scaling
  - Attributed: The 3D surface code allows d^3 scaling of spacetime overhead due to a constant-time set of computationally universal operations.
  - Explanation: The cited work states that the number of physical qubits in a 3D surface code scales with d^3 (implied by d^2 vertices per layer and d layers) and that transversal gates are realized by constant-depth (depth one) circuits. However, it does not discuss the number of syndrome extraction rounds or the overall spacetime overhead (space × time). The citing paper's claim that the 3D surface code allows d^3 scaling of spacetime overhead due to constant-time universal operations generalizes beyond what the cited work establishes.
  - Quote (verified): Each chequerboard layer in the lattice has d^2 vertices each and each diamond layer has 2d(d-1) vertices. Therefore, for a distance d lattice, the number of physical qubits in each code is
  - Note: scope_inflation
  - Passage (0.75, abstract): One of the leading quantum computing architectures is based on the two-dimensional (2D) surface code. This code has many advantageous properties such as a high error threshold and a planar layout of physical qubits where each physical qubit need only interact with its nearest neighbors. However, the transversal logical gates available in 2D surface codes are limited. This means that an additional (resource-intensive) procedure known as magic state distillation is required to do universal quantum computing with 2D surface codes. Here, we examine three-dimensional (3D) surface codes in the conte
  - Passage (0.73, Abstract): One of the leading quantum computing architectures is based on the two-dimensional (2D) surface code. This code has many advantageous properties such as a high error threshold and a planar layout of physical qubits where each physical qubit need only interact with its nearest neighbours. However, the transversal logical gates available in 2D surface codes are limited. This means that an additional (resource intensive) procedure known as magic state distillation is required to do universal quantum computing with 2D surface codes. Here, we examine three-dimensional (3D) surface codes in the cont
  - Passage (0.73, Discussion): In this article, we introduced the rectified picture of 3D surface codes. We used the rectified picture to analyse stacks of three 3D surface codes, showing that CCZ is transversal in these codes. In addition, we detailed 3D surface code architectures which allow us to do universal quantum computing without magic state distillation.

As we mentioned in Section §, the large resource cost of magic state distillation has motivated research into alternative implementations of non-Clifford gates in topological codes. To reason about the resource scaling of different architectures we use a spacetime
**old pipeline (da5f9e3)** · reference text: full · `.cache/old-pipeline-cache/fulltext/76c96fdd5b9e5df4eeb710fb75e528c2.txt`
- **overstated** (0.90) · scope_inflation, hypothesis_as_finding
  - Claim (finding): The 3D surface code allows d^3 scaling
  - Attributed: The 3D surface code allows d^3 scaling
  - Explanation: The cited work only states that a 3D surface code needs O(d^3) physical qubits and notes that the number of syndrome‑extraction rounds is unknown (and unlikely to be constant). It does not claim that constant‑time universal operations yield d^3 scaling of spacetime overhead, so the citing sentence overstates what the source establishes.
  - Quote (verified): A distance d 3D surface code requires O(d^3) physical qubits but an unknown number of rounds of syndrome extraction
  - Note: scope_inflation
  - Note: hypothesis_as_finding
  - Passage (0.74, abstract): One of the leading quantum computing architectures is based on the two-dimensional (2D) surface code. This code has many advantageous properties such as a high error threshold and a planar layout of physical qubits where each physical qubit need only interact with its nearest neighbors. However, the transversal logical gates available in 2D surface codes are limited. This means that an additional (resource-intensive) procedure known as magic state distillation is required to do universal quantum computing with 2D surface codes. Here, we examine three-dimensional (3D) surface codes in the conte
  - Passage (0.73, Abstract): One of the leading quantum computing architectures is based on the two-dimensional (2D) surface code. This code has many advantageous properties such as a high error threshold and a planar layout of physical qubits where each physical qubit need only interact with its nearest neighbours. However, the transversal logical gates available in 2D surface codes are limited. This means that an additional (resource intensive) procedure known as magic state distillation is required to do universal quantum computing with 2D surface codes. Here, we examine three-dimensional (3D) surface codes in the cont
  - Passage (0.73, Discussion): In this article, we introduced the rectified picture of 3D surface codes. We used the rectified picture to analyse stacks of three 3D surface codes, showing that CCZ is transversal in these codes. In addition, we detailed 3D surface code architectures which allow us to do universal quantum computing without magic state distillation.

As we mentioned in Section §, the large resource cost of magic state distillation has motivated research into alternative implementations of non-Clifford gates in topological codes. To reason about the resource scaling of different architectures we use a spacetime

