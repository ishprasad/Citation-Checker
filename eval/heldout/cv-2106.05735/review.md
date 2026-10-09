# Review: The Medical Segmentation Decathlon

Field: Computer science: computer vision · job `7b456fc70151` · 8 flagged citations on untouched sentences.

Mark each one in annotations.json as `genuine` (a real citation problem), `false_alarm`, or `debatable`, with a one-line note.

## s4 → Maier-Hein et al., 2018: Why rankings of biomedical image analysis competitions should be interpreted with care
*Reference text: full* · `.cache/fulltext/e7340d27a874f0f4e1b468165f263167.txt`

> It is so far the most widely investigated medical image processing task, with about 70% of all biomedical image analysis challenges dedicated to it (Maier-Hein et al., 2018).

Before: Semantic segmentation refers to the process of transforming raw medical images into clinically relevant, spatially structured information, such as outlining tumor boundaries, and is an essential prerequisite for a number of clinical applications, such as radiotherapy planning (Liang et al., 2019) and treatment response monitoring (Assefa et al., 2010).

- **overstated** (0.90) · corrected_source
  - Claim (finding): About 70% of all biomedical image analysis challenges are dedicated to semantic segmentation
  - Attributed: About 70% of all biomedical image analysis challenges are dedicated to semantic segmentation.
  - Explanation: The cited work reports that 70% of biomedical image analysis tasks (not challenges) were related to segmentation, which does not directly support the claim about challenges.
  - Quote (verified): Most tasks were related to segmentation (70%) and classification (10%)
  - Note: corrected_source
  - Passage (0.68, abstract): International challenges have become the standard for validation of biomedical image analysis methods. Given their scientific impact, it is surprising that a critical analysis of common practices related to the organization of challenges has not yet been performed. In this paper, we present a comprehensive analysis of biomedical image analysis challenges conducted up to now. We demonstrate the importance of challenges and show that the lack of quality control has critical consequences. First, reproducibility and interpretation of the results is often hampered as only a fraction of relevant inf
  - Passage (0.78, 2. Results): 150 biomedical image analysis challenges Up until the end of 2016, 150 biomedical image analysis challenges that met our inclusion criteria (see Online Methods) were conducted with a total of 549 different image analysis tasks (see Fig. 1). 56% of these challenges published their results in journals or conference proceedings. The information used in this paper from the remaining challenges was acquired from websites. Most tasks were related to segmentation (70%) and classification (10%) and were organized within the context of the MICCAI conference (50%), and the IEEE International Symposium o
  - Passage (0.77, Evaluation): M.E., M.S., A.R., P.S., S.O., and P.M.F. acquired the publicly available data of all biomedical challenges conducted up to 2016. A.K.-S, C.S., M.E., A.R., S.O. and L.M.-H. designed and implemented the statistical analysis. F.v.d.S., A.C., G.C.S., B.H.M., S.S., B.A.L., K.S., O.M., G.Z., H.B., A.A.T., C.-W.W., A.P.B., and P.F.N. contributed data from 2015 segmentation challenges. A.R., M.E. and L.M.-H. designed the international questionnaire with additional input from all co-authors and analyzed the results. T.A. and K.H. and B.H.M. provided background on research practices from other fields (c

## s5 → Isensee et al., 2021: nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
*Reference text: full* · `.cache/fulltext/9d2ba66cafa868d78aeab00c0e6351e8.txt`

> With thousands of algorithms published in the field of biomedical image segmentation per year (Isensee et al., 2021), however, it has become challenging to decide on a baseline architecture as starting point when designing an algorithm for a new given clinical problem.

Before: It is so far the most widely investigated medical image processing task, with about 70% of all biomedical image analysis challenges dedicated to it (Maier-Hein et al., 2018).

- **not_found** (0.95) · escalated
  - Claim (finding): Thousands of algorithms are published in the field of biomedical image segmentation per year
  - Attributed: Thousands of algorithms are published in the field of biomedical image segmentation per year.
  - Explanation: The cited work states that approximately 12,000 studies cite the 2015 U-Net architecture on biomedical image segmentation (cumulative since 2015) and describes a thriving research landscape with many competitions and challenge entries. It does not provide a figure for the number of algorithms published per year in biomedical image segmentation.
  - Quote: -
  - Passage (0.65, Introduction): Semantic segmentation transforms raw biomedical image data into meaningful, spatially structured information and thus plays an essential role for scientific discovery in the field falk2019u,hollon2020near. At the same time, semantic segmentation is an essential ingredient to numerous clinical applications aerts2014decoding,nestle2005comparison, including applications of artificial intelligence in diagnostic support systems de2018clinically,ACDCTMI, therapy planning support nikolov2018deep, intra-operative assistance hollon2020near or tumor growth monitoring kickingereder2019automated. The high
  - Passage (0.65, Introduction): As a consequence, the end-user is commonly left with an iterative trial and error process during method design that is mostly driven by their individual experience, only scarcely documented and hard to replicate, inevitably evoking suboptimal segmentation pipelines and methodological findings that do not generalize to other datasets litjens2017survey,bergstra2012random. To further complicate things, there is an unmanageable number of research papers that propose architecture variations and extensions for performance improvement. This bulk of studies is incomprehensible to the non-expert and di
  - Passage (0.64, Abstract): Biomedical imaging is a driver of scientific discovery and core component of medical care, currently stimulated by the field of deep learning. While semantic segmentation algorithms enable 3D image analysis and quantification in many applications, the design of respective specialised solutions is non-trivial and highly dependent on dataset properties and hardware conditions. We propose nnU-Net, a deep learning framework that condenses the current domain knowledge and autonomously takes the key decisions required to transfer a basic architecture to different datasets and segmentation tasks. Wit

## s8 → Maier-Hein et al., 2018: Why rankings of biomedical image analysis competitions should be interpreted with care
*Reference text: full* · `.cache/fulltext/e7340d27a874f0f4e1b468165f263167.txt`

> The MSD challenge aimed to test the ability of machine learning algorithms to accurately segment a large collection of prescribed regions of interest, as defined by ten different data sets, each corresponding to a different anatomical structure (see Fig. §) and to at least one medical imaging task (Maier-Hein et al., 2018).

- **not_found** (0.98) · escalated · corrected_source
  - Claim (finding): The MSD challenge aimed to test the ability of machine learning algorithms to accurately segment a large collection of prescribed regions of interest, as defined by ten different data sets, each corresponding to a different anatomical structure, and to at least one medical imaging task
  - Attributed: The MSD challenge aimed to test the ability of machine learning algorithms to accurately segment a large collection of prescribed regions of interest, as defined by ten different data sets, each corresponding to a different anatomical structure and to at least one medical imaging task.
  - Explanation: The cited work is a meta-analysis of 150 biomedical image analysis challenges conducted up to 2016, focusing on general practices, ranking robustness, and reporting parameters. It does not mention the 'MSD challenge' (Medical Segmentation Decathlon), its ten datasets, specific anatomical structures, or associated medical imaging tasks. The claim describes a specific challenge not covered in the cited work.
  - Quote: -
  - Note: corrected_source
  - Passage (0.59, abstract): International challenges have become the standard for validation of biomedical image analysis methods. Given their scientific impact, it is surprising that a critical analysis of common practices related to the organization of challenges has not yet been performed. In this paper, we present a comprehensive analysis of biomedical image analysis challenges conducted up to now. We demonstrate the importance of challenges and show that the lack of quality control has critical consequences. First, reproducibility and interpretation of the results is often hampered as only a fraction of relevant inf
  - Passage (0.70, Evaluation): M.E., M.S., A.R., P.S., S.O., and P.M.F. acquired the publicly available data of all biomedical challenges conducted up to 2016. A.K.-S, C.S., M.E., A.R., S.O. and L.M.-H. designed and implemented the statistical analysis. F.v.d.S., A.C., G.C.S., B.H.M., S.S., B.A.L., K.S., O.M., G.Z., H.B., A.A.T., C.-W.W., A.P.B., and P.F.N. contributed data from 2015 segmentation challenges. A.R., M.E. and L.M.-H. designed the international questionnaire with additional input from all co-authors and analyzed the results. T.A. and K.H. and B.H.M. provided background on research practices from other fields (c
  - Passage (0.66, 2. Results): 150 biomedical image analysis challenges Up until the end of 2016, 150 biomedical image analysis challenges that met our inclusion criteria (see Online Methods) were conducted with a total of 549 different image analysis tasks (see Fig. 1). 56% of these challenges published their results in journals or conference proceedings. The information used in this paper from the remaining challenges was acquired from websites. Most tasks were related to segmentation (70%) and classification (10%) and were organized within the context of the MICCAI conference (50%), and the IEEE International Symposium o

## s10 → Simpson et al., 2019: A large annotated medical image dataset for the development and evaluation of segmentation algorithms
*Reference text: full* · `.cache/fulltext/bd97c6ab5527d55e49812baf688f9ebe.txt`

> The Decathlon challenge made ten data sets available online (Simpson et al., 2019), where each data set had between one and three region-of-interest (ROI) targets (17 targets in total).

- **not_found** (0.95) · escalated
  - Claim (finding): each data set had between one and three region-of-interest (ROI) targets (17 targets in total)
  - Attributed: The Decathlon challenge made ten data sets available online, where each data set had between one and three region-of-interest (ROI) targets (17 targets in total).
  - Explanation: The cited work describes the ten datasets and their annotations in detail, but the provided full text does not contain an explicit statement that each dataset had between one and three ROI targets or that there were 17 targets in total. The table that might summarize the number of labels per dataset is referenced but not included in the available text.
  - Quote: -
  - Passage (0.42, abstract): Semantic segmentation of medical images aims to associate a pixel with a label in a medical image without human initialization. The success of semantic segmentation algorithms is contingent on the availability of high-quality imaging data with corresponding labels provided by experts. We sought to create a large collection of annotated medical image datasets of various clinically relevant anatomies available under open source license to facilitate the development of semantic segmentation algorithms. Such a resource would allow: 1) objective assessment of general-purpose segmentation methods th
  - Passage (0.51, Datasets): Annotations of the liver and tumours were performed by radiologists.

Task04_Hippocampus. The dataset consisted of MRI acquired in 90 healthy adults and 105 adults with a non-affective psychotic disorder (56 schizophrenia, 32 schizoaffective disorder, and 17 schizophreniform disorder) taken from the Psychiatric Genotype/Phenotype Project data repository at Vanderbilt University Medical Center (Nashville, TN, USA). Patients were recruited from the Vanderbilt Psychotic Disorders Program and controls were recruited from the surrounding community. All participants were assessed with the Structured
  - Passage (0.50, Datasets): An overview of the ten datasets is provided in Table §. The datasets were chosen based on availability and appropriateness for semantic segmentation algorithm development. The rationale for inclusion of each dataset is described in the table. Exemplar images and labels for each task are provided in Figure §.

Task01_BrainTumour. The brain tumour dataset included in the MSD challenge describes a subset of the data used in the 2016 and 2017 Brain Tumour Image Segmentation (BraTS) challenges (authorBakas et al., n.d.; authorBakas and S, n.d.; authorMenze and H, 2024). Specifically, 750 multi-para

## s13 → Bilic et al., 2019: The Liver Tumor Segmentation Benchmark (LiTS)
*Reference text: full* · `.cache/fulltext/55ea30bdf6683b364c6e07f39e036f02.txt`

> The data was acquired in the IRCAD Hôpitaux Universitaires, Strasbourg, France and contained a subset of patients from the 2017 Liver Tumor Segmentation (LiTS) challenge (Bilic et al., 2019).

Before: This data set was selected due to the challenging nature of having significant label unbalance between large (liver) and small (tumor) target region of interests (ROIs).

- **not_found** (0.85) · escalated
  - Claim (method): The data was acquired in the IRCAD Hôpitaux Universitaires, Strasbourg, France and contained a subset of patients from the 2017 Liver Tumor Segmentation (LiTS) challenge
  - Attributed: The data was acquired in the IRCAD Hôpitaux Universitaires, Strasbourg, France and contained a subset of patients from the 2017 Liver Tumor Segmentation (LiTS) challenge.
  - Explanation: The cited work states that the LiTS dataset includes data from IRCAD in France and that the LiTS benchmark dataset constitutes the Medical Segmentation Decathlon's liver tasks. However, it does not mention the specific hospital name 'Hôpitaux Universitaires, Strasbourg' nor that the Decathlon data is a subset of patients from the 2017 LiTS challenge. These specific details attributed to the cited work are not present in the text.
  - Quote: -
  - Passage (0.78, abstract): In this work, we report the set-up and results of the Liver Tumor Segmentation Benchmark (LiTS), which was organized in conjunction with the IEEE International Symposium on Biomedical Imaging (ISBI) 2017 and the International Conferences on Medical Image Computing and Computer-Assisted Intervention (MICCAI) 2017 and 2018. The image dataset is diverse and contains primary and secondary tumors with varied sizes and appearances with various lesion-to-background levels (hyper-/hypo-dense), created in collaboration with seven hospitals and research institutions. Seventy-five submitted liver and liv
  - Passage (0.77, Abstract): In this work, we report the set-up and results of the Liver Tumor Segmentation Benchmark (LiTS), which was organized in conjunction with the IEEE International Symposium on Biomedical Imaging (ISBI) 2017 and the International Conferences on Medical Image Computing and Computer-Assisted Intervention (MICCAI) 2017 and 2018. The image dataset is diverse and contains primary and secondary tumors with varied sizes and appearances with various lesion-to-background levels (hyper-/hypo-dense), created in collaboration with seven hospitals and research institutions. Seventy-five submitted liver and liv
  - Passage (0.76, Introduction): In this paper, we describe the three key contributions to fully automated liver and liver tumor segmentation. First, we generate a new public multi-center dataset of 201 abdominal CT Volumes and the reference segmentations of liver and liver tumors. Second, we present the set-up and the summary of our LiTS benchmarks in three grand challenges. Third, we review, evaluate, rank, and analyze the resulting state-of-the-art algorithms and results. The paper is structured as follows: Section II reviews existing public datasets and state-of-the-art automated liver and liver tumors segmentation. Next,

## s30 → Isensee et al., 2021: nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
*Reference text: full* · `.cache/fulltext/9d2ba66cafa868d78aeab00c0e6351e8.txt`

> The method won 33 out of 53 tasks with a median rank of 1 (interquartile range (IQR) of (1;2)) in the 53 tasks (Isensee et al., 2021), for example being the winning method of the famous BraTS challenge in 2020.

Before: In the two years after the challenge, the winning algorithm, nnU-Net (with sometimes minor modification) competed in a total of 53 further segmentation tasks.

- **not_found** (0.95) · escalated
  - Claim (finding): The method won 33 out of 53 tasks with a median rank of 1 (interquartile range (IQR) of (1;2)) in the 53 tasks
  - Attributed: nnU-Net won 33 out of 53 tasks with a median rank of 1 (interquartile range (IQR) of (1;2)) in the 53 tasks.
  - Explanation: The cited nnU-Net paper reports results on 49 segmentation tasks across 10 challenges, stating it set a new state of the art in 29 of them. It does not mention a set of 53 tasks, 33 wins, a median rank of 1, or an IQR of (1;2). The 53-task statistic appears to originate from the citing paper's own analysis, not from the cited work.
  - Quote: -
  - Passage (0.56, Results): Challenges are international competitions that can be seen as the equivalent to clinical trials for algorithm benchmarking. Typically, they are hosted by individual researchers, institutes, or societies, aiming to assess the performance of multiple algorithms in a standardized environment maier2018rankings. In all segmentation tasks, nnU-Net was trained from scratch using only the provided challenge data. While the methodology behind nnU-Net was developed on the 10 training sets provided by the Medical Segmentation Decathlon decathlonDataPaper, the remaining datasets and tasks were used for in
  - Passage (0.56, Results): The results clearly show that caution is required when drawing methodological conclusions from evaluations that are based on an insufficient number of datasets. While five out of the nine variants achieved rank 1 in at least one of the datasets, neither of them exhibits consistent improvements across the ten tasks. The original nnU-Net configuration shows the best generalization and ranks first when aggregating results of all datasets.

In current research practice, evaluation is rarely performed on more than two datasets and even then the datasets come with largely overlapping properties (suc
  - Passage (0.55, Medical Segmentation Decathlon): Challenge summary. The Medical Segmentation Decathlon SMdecathlonDataPaper is a competition that spans 10 different segmentation tasks. These tasks are selected to cover a large proportion of the dataset variability in the medical domain. The overarching goal of the competition was to encourage researchers to develop algorithms that can work with these datasets out of the box without manual intervention. Each of the tasks comes with respective training and test data. A detailed description of datasets can be found on the challenge homepage. Originally, the challenge was divided into two phases
- **not_found** (0.95) · escalated
  - Claim (finding): The method is the winning method of the BraTS challenge in 2020
  - Attributed: nnU-Net won the BraTS challenge in 2020.
  - Explanation: The cited work details nnU-Net's performance on the Medical Segmentation Decathlon, the KiTS 2019 challenge, ACDC, PROMISE12, LiTS, MSLesion, BCV, and CREMI, but it does not mention a BraTS challenge or a victory in 2020.
  - Quote: -
  - Passage (0.63, Medical Segmentation Decathlon): Challenge summary. The Medical Segmentation Decathlon SMdecathlonDataPaper is a competition that spans 10 different segmentation tasks. These tasks are selected to cover a large proportion of the dataset variability in the medical domain. The overarching goal of the competition was to encourage researchers to develop algorithms that can work with these datasets out of the box without manual intervention. Each of the tasks comes with respective training and test data. A detailed description of datasets can be found on the challenge homepage. Originally, the challenge was divided into two phases
  - Passage (0.63, Acknowledgements): This work was co-funded by the National Center for Tumor Diseases (NCT) in Heidelberg and the Helmholtz Imaging Platform (HIP) of the German Cancer Consortium (DKTK). We thank our colleagues at DKFZ who were involved in the various challenge contributions, especially Andre Klein, David Zimmerer, Jakob Wasserthal, Gregor Koehler, Tobias Norajitra and Sebastian Wirkert who contributed to the Decathlon submission. We also thank the MITK team who supported us in producing all medical dataset visualizations. We are also thankful to all the challenge organizers, who provided an important basis for o
  - Passage (0.62, Results): Despite its generic nature, nnU-Net outperforms most existing semantic segmentation solutions, even though the latter were specifically optimized towards the respective task. Overall, nnU-Net sets a new state of the art in 29 out of 49 target structures and otherwise shows performances on par with or close to the top leaderboard entries.

Figure: nnU-Net outperforms most specialized deep learning pipelines. Quantitative results from all international challenges that nnU-Net competed in. For each segmentation task, results achieved by nnU-Net are highlighted in red, competing teams are shown in

## s31 → Isensee et al., 2021: nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation
*Reference text: full* · `.cache/fulltext/9d2ba66cafa868d78aeab00c0e6351e8.txt`

> For instance, eight nnU-Net derivatives were ranked in the top 15 algorithms of the 2019 Kidney and Kidney Tumor Segmentation Challenge (KiTS) (Isensee et al., 2021), the MICCAI challenge with the most participants in the year 2019.

Before: The method further became the new state-of-the-art method and was used in several segmentation challenges by other researchers.

- **overstated** (0.85)
  - Claim (finding): Eight nnU-Net derivatives were ranked in the top 15 algorithms of the 2019 Kidney and Kidney Tumor Segmentation Challenge (KiTS)
  - Attributed: Eight nnU-Net derivatives were ranked in the top 15 algorithms of the 2019 Kidney and Kidney Tumor Segmentation Challenge (KiTS)
  - Explanation: The cited work states that the top-15 methods in the KiTS 2019 challenge were offspring of the U-Net architecture, but it does not specify that eight of them were nnU-Net derivatives or give the exact number eight.
  - Quote (verified): First, the top-15 methods were offspring of the (3D) U-Net architecture from 2016, confirming its impact on the field of biomedical image segmentation
  - Passage (0.81, Kidney and Kidney Tumor Segmentation (KiTS) (D17)): Challenge summary. The Kidney and Kidney Tumor Segmentation challenge SMheller2019kits19 was the largest competition (in terms of number of participants) at MICCAI 2019. The target structures are the kidneys and kidney tumors. 210 training and 90 test cases are provided by the challenge organizers. The organizers provide the data both in their original geometry (with voxel spacing varying between cases) as well as interpolated to a common voxel spacing. Evaluation of the test set predictions is done on the online platform. We participated in the original KiTS 2019 MICCAI challenge with a manua
  - Passage (0.79, Introduction): The strong results underline the significance of nnU-Net for users who require algorithms for semantic segmentation on their custom datasets: as an open source tool, nnU-Net can simply be downloaded and trained out-of-the box to generate state of the art segmentations without requiring manual adaptation or expert knowledge. We further demonstrate shortcomings in the design process of current biomedical segmentation methods. Specifically, we take an in-depth look at the 2019 Kidney and Kidney Tumor Segmentation (KiTS) semantic image segmentation challenge and demonstrate how important task-spec
  - Passage (0.79, Results): Despite its generic nature, nnU-Net outperforms most existing semantic segmentation solutions, even though the latter were specifically optimized towards the respective task. Overall, nnU-Net sets a new state of the art in 29 out of 49 target structures and otherwise shows performances on par with or close to the top leaderboard entries.

Figure: nnU-Net outperforms most specialized deep learning pipelines. Quantitative results from all international challenges that nnU-Net competed in. For each segmentation task, results achieved by nnU-Net are highlighted in red, competing teams are shown in

## s37 → Sirinukunwattana et al., 2017: Gland segmentation in colon histology images: The glas challenge contest
*Reference text: full* · `.cache/fulltext/f75495b70d5705735d20f037e7ecf2f2.txt`

> Most of the observed task-specific performances reflect the natural difficulty and expected inter-rater variability of the tasks: Liver and spleen are large organs that are easy to detect and outline (Campadelli et al., 2009), whereas pancreas and colon cancers are much harder to segment as annotation experts themselves often do not agree on the correct outlines (Sirinukunwattana et al., 2017; Re et al., 2011).

Before: Similarly to the development phase, a much smaller drop in performance (0.56 and 0.96 for colon and spleen respectively) was observed in the top ranking method.

- **overstated** (0.42) · escalated · scope_inflation
  - Claim (finding): Pancreas and colon cancers are much harder to segment as annotation experts themselves often do not agree on the correct outlines
  - Attributed: Colon cancers are much harder to segment because annotation experts often do not agree on the correct outlines.
  - Explanation: The cited work states that inter-observer variability was not accounted for because ground truth came from a single expert, and that malignant gland boundaries are not well-defined. It does not report that experts frequently disagree on outlines; the claim generalizes beyond the source's description of dataset limitations.
  - Quote: First of all, inter-observer variability was not taken into account as the ground truth was generated by a single expert. This is because the intricate and arduous nature of the problem makes it difficult to find several volunteer experts to perform manual segmentation. Considerable experience is required in order to delineate boundaries of malignant glands, which are not so well-defined as those of the benign ones.
  - Note: scope_inflation
  - Passage (0.72, 1. Introduction 1): Participants developed gland 45 segmentation algorithms, which were applied to benign tissue and to colonic 46 carcinomas. A training dataset was provided, together with ground truth 47 annotations by an expert pathologist. The participants developed and op- 48 timized their algorithms on this dataset. The results were judged on the 49 performance of the algorithms on test datasets. Success was measured by 50 how closely the automated segmentation matched the pathologist’s. 51
  - Passage (0.71, Method): 801 Automatic fetal ultrasound standard plane detection using knowledge 802 transferred recurrent neural networks, in: Medical Image Computing and 803 Computer-Assisted Intervention–MICCAI 2015. Springer, pp. 507–514. 804

Chen, H., Qi, X., Yu, L., Heng, P.A., 2016. DCAN: Deep contour-aware net- 805 works for accurate gland segmentation. arXiv preprint arXiv:1604.02677 806 . 807

Chen, L.C., Papandreou, G., Kokkinos, I., Murphy, K., Yuille, A.L., 2014. 808 Semantic image segmentation with deep convolutional nets and fully con- 809 nected CRFs. arXiv preprint arXiv:1412.7062 . 810

Cohen, A., R
  - Passage (0.70, Method): Some may have 667 argued that there is no need to perform segmentation, but instead, to fol- 668 low conventional pattern recognition approaches by extracting mathematical 669 features which normally capture local and/or global tissue architecture and 670 then identifying features that are most suited to the objective of the study. 671 It is true that there are a number of successful works that follow such an 672 approach (Jafari-Khouzani and Soltanian-Zadeh, 2003; Tabesh et al., 2007; 673 Altunbay et al., 2010; Basavanhally et al., 2010; Ozdemir and Gunduz-Demir, 674 2013; Gultekin et al., 20

