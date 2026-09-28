# Assignment 2.1: Programming Task 1
 
**Course:** 2650_COMP_5004_2 — 2026 Semester 2
**Due:** Monday, 26 Oct 2026, 9:00 am (Monday after Mid-Semester Break)
**Available from:** 3 Aug 2026, 9:00 am
**Weighting:** 30%
**Type:** Group work (team of 3–4)
**Points:** 30 (ungraded / unlimited attempts allowed)
 
## NumCompute: A High-Performance NumPy Toolkit (Week 1–6)
 
In this assignment, you will work in a team of 3–4 to build a **modular, production-grade scientific computing toolkit** using **plain Python + NumPy** (Weeks 1–6 scope).
 
Your toolkit should provide reusable, vectorised components for:
 
- Data I/O and preprocessing
- Sorting / search / top-k and ranking
- Descriptive statistics
- Evaluation metrics
- Finite-difference gradients (and optional Jacobians)
- A lightweight pipeline abstraction for chaining transformations
- Benchmark utilities to compare vectorised implementations against Python loops
Apply the programming skills from Weeks 1–2 and 4–6, together with the AIML concepts from Weeks 3–6, to complete this assignment as a team. The emphasis is on **deep algorithmic understanding**, **numerical stability** (e.g. stable softmax, NaN handling, tie handling), and **clean software engineering**: clear APIs, consistent semantics and shapes, comprehensive unit tests, docstrings, and reproducible benchmarks.
 
> ⚠️ **External ML/DL or data libraries are not permitted** (no scikit-learn, pandas, PyTorch, etc.) — plain Python and NumPy only.
 
## Course Learning Outcomes
 
Completing this assignment will help you develop the following:
 
- **CLO1:** Design and implement algorithmic solutions in code that meet specified AI Machine Learning (AIML) functionality requirements.
- **CLO2:** Generate modularised code structures to support the implementation of AI Machine Learning (AIML) workflows.
- **CLO3:** Ability to systematically debug and test code to ensure functionality and reliability.
- **CLO4:** Illustrate best practices in programming through coding style standards and course code documentation.
## Instructions
 
You will implement a **modular, end-to-end machine learning framework from scratch**, simulating the core functionality of libraries like scikit-learn. It must include components for data handling, model training, evaluation, and pipeline abstraction. The focus is on deep algorithmic understanding, numerical computation, and clean software engineering. No external ML/DL libraries are allowed — plain Python and NumPy only.
 
### Required package structure
 
```
NumCompute/
├── numcompute/
│   ├── __init__.py
│   ├── io.py            # CSV reader (streaming/chunking), dtype handling
│   ├── preprocessing.py # StandardScaler, MinMaxScaler, Imputer, OneHot
│   ├── sort_search.py   # Sorts, argpartition/Top-k, quickselect, binary_search
│   ├── rank.py          # Ranking with ties; percentiles
│   ├── stats.py         # Streaming stats (Welford), histogram, quantiles
│   ├── metrics.py       # Accuracy/Precision/Recall/F1, MSE, Confusion, ROC/AUC*
│   ├── optim.py         # Finite-diff gradients (∇, Jacobian), line search (optional)
│   ├── pipeline.py      # Transformer/Estimator API, Compose, FeatureUnion
│   ├── utils.py         # Distances, activations, logsumexp, top-k helpers, batching
│   └── benchmarking.py  # Micro-bench harness, performance comparisons
├── tests/                # ≥ 20 unit tests; edge cases covered
├── demo/
│   └── quickstart.ipynb  # End-to-end demo
├── README.md             # Installation, examples, API overview, performance table
└── pyproject.toml / setup.py  # Packaging configuration
```
 
The core module specifications are defined below in [Module Specifications](#module-specifications).
 
### Requirements
 
- **Vectorisation:** No Python loops in core computations unless necessary.
- **Numerical stability:** Handle overflow/underflow, ties, and NaNs; use stable forms (e.g. max-shifted softmax, `logsumexp`).
- **API consistency:** Clear `axis` semantics, documented input/output shapes, informative error messages for type/shape mismatches.
- **Documentation:** Docstrings must specify parameters, return values, shapes, exceptions raised, and time/space complexity.
- **Testing:** ≥ 20 unit tests covering edge cases (empty arrays, all-equal values, duplicates/ties, extreme `k`, NaNs, non-contiguous strides).
- **Benchmarking:** Provide reproducible benchmarks (script or notebook) with timing comparisons and environment notes.
### Demo (`demo/quickstart.ipynb`) should
 
- Read a CSV using `io.py` (including rows with missing values).
- Preprocess with `preprocessing.py` (e.g. `StandardScaler`, `Imputer`, `OneHotEncoder`).
- Show `sort_search` (top-k, quickselect, binary search) and `rank` features (tie handling, percentiles).
- Compute stats with `stats.py` (mean/variance, histogram/quantiles).
- Estimate gradients/Jacobian using `optim.py` (finite differences) on scalar/vector functions.
- Benchmark vectorised implementations vs. Python loops; include a short performance table.
## Module Specifications
 
### `io.py`
 
- Load CSV files using `numpy.loadtxt` or `numpy.genfromtxt`
- Support custom delimiters (e.g. comma, tab)
- Handle missing values (skip or fill with a placeholder value)
- Return data as NumPy arrays
### `preprocessing.py`
 
- Implement:
  - `StandardScaler` (z-score standardization)
  - `MinMaxScaler` (scale to a given range)
  - `OneHotEncoder` (for categorical variables)
  - *(Optional)* Simple imputer to replace NaN with a constant
- API:
  - `fit(X) -> self`
  - `transform(X) -> X_out`
  - `fit_transform(X) -> X_out`
- Fully vectorized using NumPy; avoid Python loops
### `sort_search.py`
 
- **Sorting:**
  - Stable sort wrapper (`np.sort(kind='stable')`)
  - Multi-key sort (sort by multiple columns)
- **Top-k / partial sort:**
  - `topk(values, k, largest=True, return_indices=True)` using `np.argpartition`
  - Implement **quickselect** for educational purposes
- **Searching:**
  - `binary_search(sorted_array, x)` returning insertion index and existence boolean
### `rank.py`
 
- `rank(data, method='average'|'dense'|'ordinal')` — handle ties
- `percentile(data, q, interpolation='linear'|'lower'|'higher'|'midpoint')`
### `stats.py`
 
- Basic descriptive statistics: mean, median, standard deviation, min, max
- Histogram
- Quantiles (with NaN handling)
- Axis-wise stats with clear dimension/shape behaviour
### `metrics.py`
 
- **Classification:** `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`
- **Regression:** `mse(y_true, y_pred)`
- **Bonus:** `roc_curve` and `auc` for binary classification
### `optim.py`
 
- Finite-difference gradient estimation:
  - `grad(f, x, h=1e-5, method='central'|'forward')`
  - `jacobian(F, x, ...)`
### `pipeline.py`
 
- Minimal `Transformer` / `Estimator` protocol:
  - Preprocessors: `fit`, `transform`
  - Models: `fit`, `predict` (no ML model implementation required)
- `Pipeline` chaining, e.g.:
```python
  pipe = Pipeline([
      ('scale', StandardScaler()),
      ('encode', OneHotEncoder())
  ])
  X_tr = pipe.fit_transform(X)
```
 
## What to Submit (Each Group)
 
1. **Git repo** (link), including:
   - `numcompute/` package with all modules
   - `tests/` folder (≥ 20 unit tests)
   - `demo/quickstart.ipynb`
   - `benchmark/` (loop vs. vectorised)
   - `README.md` with usage instructions
2. **Demo video** (max 5 min), a screen recording showing:
   - Setting up and running the demo
   - Using your pipeline
   - Reading outputs (metrics, ranks, gradients)
   - Submit as `.mp4` or a shareable link
3. **Report** (PDF, max 4 pages), a brief technical summary including:
   - Design choices (vectorisation, API)
   - Testing & edge cases
   - Benchmark results
   - Reflections on teamwork
## Academic Integrity and Use of Generative AI
 
Generative AI is permitted for this assignment at the following level:
 
**Allowed level of AI use: AI-assisted ideas.** See the [AI Assessment Guidelines](https://learn.adelaide.edu.au/courses/370/files/1717942?wrap=1).
 
| Level | Type | What this means |
|---|---|---|
| 1 | **No AI** | The entire assessment must be completed without AI. AI must not be used at any point, including for translation, editing, or generating content or ideas. |
| 2 | **AI-assisted editing** | AI can be used as an editor to improve clarity/quality (spelling/grammar checking, style rewriting, feedback on language/structure, data cleaning). You remain accountable for the quality and appropriateness of the writing and all other aspects of the assignment. AI should not be used to generate new ideas, answers, or content (that's Level 3). |
| 3 | **AI-assisted ideas** *(this assignment's level)* | AI can be used for brainstorming, creating structures, and generating ideas, including feedback on content and methods. You must clearly acknowledge how AI was used. AI can be used as a source of information, but you are accountable for validating and verifying it. AI-generated content should be cited as a source (quoted or sufficiently paraphrased, with source cited). |
| 4 | **AI-assisted creation, human evaluation** | AI completes certain elements of the task, and you provide discussion/commentary on the AI-generated content, with explicit critical engagement and evaluation of its output. AI-generated content must be cited, and you must fully understand and be able to discuss every aspect of it. |
 
**Confidentiality and copyright:** Don't upload or input into AI anything you don't own or that may be subject to copyright, and never input confidential information (e.g. patient case notes).
 
**Academic integrity:** If you have questions about appropriate AI use, check with your tutor or course coordinator first. Inappropriate use may lead to an academic integrity report. You must be able to fully discuss every aspect of your assessment, and may be asked to discuss the assignment with a tutor before your mark is finalised if AI use is a concern.
 
## Grading Rubric
 
| Criteria | Exceeds | Mastery | Near | Below | No Evidence |
|---|---|---|---|---|---|
| **Algorithm Design** (/20) | Designs optimal algorithmic solutions that exceed all specified functionality requirements; accurately identifies all computational components. (16–20 pts) | Designs solutions that meet all specified requirements; identifies all computational components. (14–16 pts) | Designs solutions that meet specified requirements; identifies all computational components. (12–14 pts) | Designs solutions that meet specified requirements; identifies most computational components. (9–12 pts) | Not yet able to design solutions meeting requirements or identify computational components. (0–9 pts) |
| **Functionality Implementation** (/20) | Implements the algorithm, optimally generating workable and accurate functions meeting all requirements. (16–20 pts) | Generates workable and appropriate functions meeting all algorithmic requirements. (14–16 pts) | Generates workable and appropriate functions meeting the minimum algorithmic requirements. (12–14 pts) | Generates workable functions meeting the minimum requirements. (9–12 pts) | Not yet able to implement workable functions meeting minimum requirements. (0–9 pts) |
| **Code Reusability** (/15) | Functions are well-structured, meaningfully named, show strong abstraction. (15 pts) | Functions are accurate and well-structured. (13 pts) | Functions are accurate and organised in a reasonable structure. (11 pts) | Functions are workable and organised in a reasonable structure. (9 pts) | Not yet able to generate workable code in a reasonable structure. (7 pts) |
| **Code Maintainability** (/15) | Systematic debugging via tests resolves all bugs. (13–15 pts) | Systematic debugging via tests resolves all bugs; tests cover all functionality, no edge cases missed. (11–13 pts) | Reasonable/appropriate debugging resolves most bugs; tests cover key functionality, some edge cases missed. (9–11 pts) | Appropriate debugging resolves obvious bugs; tests cover key functionality with some edge cases missed. (7–9 pts) | Not yet able to use appropriate debugging or test key functionality. (0–7 pts) |
| **Testing & Debugging Evidence** (/10) | Systematic debugging via tests resolves all bugs; tests effectively cover all functionality, no edge cases missed. (8–10 pts) | Systematic debugging via tests resolves all bugs; tests cover all functionality, no edge cases missed. (6–8 pts) | Reasonable/appropriate debugging resolves most bugs; tests cover key functionality, some edge cases missed. (4–6 pts) | Appropriate debugging resolves obvious bugs; tests cover key/limited functionality with some edge cases missed. (2–4 pts) | Not yet able to use appropriate debugging or test key functionality. (0–2 pts) |
| **Teamwork** (/10) | Coding outputs fully consistent across members; each member's contribution fully identified in a concise format. (8–10 pts) | Coding outputs fully consistent; each member's contribution fully identified. (6–8 pts) | Coding outputs fully consistent; each member's contribution identified. (4–6 pts) | Coding outputs consistent; each member's contribution identified. (2–4 pts) | Not yet able to provide consistent outputs across members; contributions unclear. (0–2 pts) |
| **Coding Style, Documentation & Comments** (/10) | Code consistently follows best practices; clearly and fully documented. (8–10 pts) | Coding style consistently follows standards; code well-organised, commented, formatted. (6–8 pts) | Coding style mostly consistent with few deviations; code generally readable. (4–6 pts) | Code adherent to conventions with inconsistent formatting. (2–4 pts) | Not yet able to adhere to coding style standards. (0–2 pts) |
 
**Total: 100 pts** (scaled to 30 points possible for this assignment)
 
## Submission
 
- Submission types accepted: **Web URL**, **Upload**, or **Canvas Studio**.
- Requires agreeing to the tool's End-User License Agreement and confirming the submission is your own, original work.
---
 
## Missing / Not Captured from the Original Page
 
A few things worth noting:
 
1. **Rubric point boundaries are approximate in places** — the source table used overlapping ranges (e.g. "16 to >14 pts" next to "14 to >12 pts"), which I've kept as close to the original as possible, but it's worth double-checking the exact cutoffs directly on Canvas if precision matters for your team.
2. **No images or diagrams** were present in the saved page — it was text/table content only, so nothing was omitted on that front.
3. **Page chrome** (Canvas navigation menu, "Immersive Reader" links, breadcrumbs like "Previous: Programming Exam 3 | Week 9") was intentionally left out as it's not part of the assignment content itself.
