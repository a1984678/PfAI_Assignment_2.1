# Assignment 2.2: Programming Task 2

**Course:** 2650_COMP_5004_2 – Programming for Artificial Intelligence (2026 Semester 2)

| | |
|---|---|
| **Due** | Mon, 16 Nov 2026, 9:00 (Monday after SWOT Week) |
| **Points** | 25 |
| **Weighting** | 25% |
| **Attempts** | Unlimited |
| **Available from** | 12 Oct 2026, 9:00 |
| **Type** | Individual work |

---

## NumCompute-Stream: A Modularised Decision Tree-based Streaming Machine Learning Framework (Week 1-10)

In this assignment, you will extend your existing NumCompute package by building a decision tree–based machine learning framework that supports both:

- **Streaming Learning:** All components (models, preprocessing, metrics, statistics) must support incremental updates as new data arrives in chunks, simulating a real-world online learning scenario.
- **Buffered Decision Tree:** Implement a single decision tree model from scratch that handles streaming data by periodic retraining (see Requirements below).
- **Built-in Visualisation Module:** Add a lightweight `visualise.py` module that provides built-in plotting functions using matplotlib.

You should apply the learned knowledge over 10 topics in this course to build a decision tree–based, stream-compatible learning framework. This assignment emphasises deep algorithmic understanding, decision tree learning, and real-time adaptation to incoming data. Focus on numerical stability (e.g., NaN handling, tie resolution, incremental statistics), modular design, and clean engineering—well-documented APIs, shape-consistent computation, unit tests, and real-time metric visualisation.

You may use **only NumPy and matplotlib**; other external machine learning or data processing libraries (e.g., scikit-learn, pandas) are not allowed.

## Course Learning Outcomes

This assignment will help you develop your skills and knowledge, so that on successful completion of this course, you will be able to:

- **CLO1:** Design and implement algorithmic solutions in code that meets specified AI Machine Learning (AIML) functionality requirements.
- **CLO2:** Generate modularised code structures to support the implementation of AI Machine Learning (AIML) workflows.
- **CLO3:** Ability to systematically debug and test code to ensure the functionality and reliability.
- **CLO4:** Illustrate best practices in programming through coding style standards and course code documentation.

## Instructions

In this assignment, you will work individually to extend your NumCompute package by building a streaming, decision tree–based machine learning framework, centred on a single decision tree, from scratch. Your system will simulate key functionalities of online learning libraries, where incoming chunks are stored in a buffer and the decision tree is retrained once the buffer reaches a user-defined size (`buffer_size`). It must include components for incremental data handling, progressive model training, evaluation, and real-time visualisation. The focus is on deep algorithmic understanding, numerical stability, and clean software engineering. External machine learning or data libraries (e.g., scikit-learn, pandas, PyTorch) are not permitted—use only plain Python, NumPy, and matplotlib.

The updated core module specifications are defined in the file: **Assignment2.2-specs.md** (download from the Canvas assignment page).

### Requirements

- **Vectorisation:** Core logic must use NumPy vectorised operations. Avoid Python loops unless absolutely necessary.
- **Streaming Compatibility:** All modules must support chunk-wise updates through `.partial_fit()` or `.update()` methods.
- **Buffered Retraining:** Each call to `.partial_fit()` adds the new chunk to an internal data buffer; once the buffer holds at least `buffer_size` samples, the tree is retrained on the buffered data and the buffer is cleared. `buffer_size` is a hyperparameter set by the user (e.g., `DecisionTree(max_depth=5, buffer_size=500)`). Between retrains, `.predict()` uses the most recently trained tree.
- **Numerical Stability:** Handle edge cases such as NaNs, ties, zero-variance chunks, and unstable operations safely.
- **API Consistency:** Clear axis semantics, documented input/output shapes, informative error messages for type/shape mismatches.
- **Testing:** Provide at least 30 unit tests, covering both standard functionality and edge cases (especially under streaming conditions).
- **Benchmarking:** Include performance comparisons between (a) loop-based and vectorised implementations, and (b) different `buffer_size` values (e.g., how accuracy and retraining time change with the buffer size), under streaming scenarios.

### Demo (`demo/stream_demo.ipynb`) should

- Load a dataset from a CSV file using your custom I/O pipeline (`io.py`)
- Split the dataset into multiple chunks to simulate a streaming data setting
- Train the pipeline incrementally using `.partial_fit()` on each chunk
- Log and visualise key metrics over time (e.g., accuracy, error, model comparison) using your `visualise.py` module

Please refer to **classification-tree.html** (download from the Canvas assignment page) for more details of how to build a decision tree.

## What to Submit (Each Student)

### 1. Git Repo (using Fork + Branch)

- Fork from the group repository
- `main` branch: original group code (**do not modify**)
- Individual branch: your code for this assignment

Include:

- `numcompute_stream/` package with all modules
- `tests/` folder (≥ 30 unit tests)
- `demo/stream_demo.ipynb`
- `benchmark/` (loop vs. vectorised)
- `README.md` with usage instructions

### 2. Demo Video (max 5 min)

A screen-recording showing:

- Setup and running the demo
- Visualisations & predictions
- Pipeline usage and logs

Submit as `.mp4` or a shareable link.

### 3. Report (PDF, max 4 pages)

Brief technical summary including:

- Design decisions
- Testing & edge cases
- Benchmark results
- Reflections

## Rubric (25 pts)

| Criteria | Exceeds | Mastery | Near | Below | No Evidence | Points |
|---|---|---|---|---|---|---|
| **Algorithm Design** | Design optimal algorithmic solutions that exceed all specified functionality requirements. Accurately identifies all computational components in the algorithmic solution. (25 to >21) | Designs algorithmic solutions that meet all specified functionality requirements. Identifies all computational components. (21 to >18) | Designs algorithmic solutions that meet specified functionality requirements. Identifies all computational components. (18 to >16) | Designs algorithmic solutions that meet specified functionality requirements. Identifies most computational components. (16 to >12) | Not yet able to design algorithmic solutions that meet specified requirements and identify computational components. (12 to >0) | /25 |
| **Functionality Implementation** | Implements the algorithm in the required coding environment by optimally generating workable and accurate functions that meet all algorithmic requirements. (25) | Implements the algorithm by generating workable and appropriate functions that meet all algorithmic requirements. (21) | Generates workable and appropriate functions that meet the minimum algorithmic requirements. (18) | Generates workable functions that meet the minimum requirements of the algorithm. (16) | Not yet able to implement the algorithm by generating workable functions that meet the minimum algorithmic requirements. (12) | /25 |
| **Code Reusability** | Code structure is completely modularised and shows excellent scalability across all related pipelines. (15 to >13) | Completely modularised; scalability across most related pipelines. (13 to >11) | Completely modularised; scalability across related pipelines. (11 to >9) | Modularised; shows scalability across pipelines. (9 to >7) | Not yet able to show a modularised code structure. (7 to >0) | /15 |
| **Code Maintainability** | Functions are well-structured, named meaningfully, and show strong abstraction. (15 to >13) | Functions are accurate and well-structured. (13 to >11) | Functions are accurate and organised in a reasonable structure. (11 to >9) | Functions are workable and organised in a reasonable structure. (9 to >7) | Not yet able to generate workable code in a reasonable structure. (7 to >0) | /15 |
| **Testing & Debugging Evidence** | Systematic debugging process through tests to resolve all bugs. Tests effectively cover all functionalities with no edge cases missed. (10) | Systematic debugging process; tests cover all functionalities with no edge cases missed. (9) | Reasonable debugging process resolves most bugs. Tests cover key functionality with some edge cases missed. (8) | Appropriate debugging process resolves obvious bugs. Tests cover key functionality with some edge cases missed and/or limited functionality. (4) | Not yet able to use an appropriate debugging process to resolve obvious bugs and/or test key functionality. (2) | /10 |
| **Coding style, documentation and comments** | Code consistently follows best practices and is clearly and fully documented. (10 to >8) | Coding style consistently follows standards. Code is well-organised, commented, and formatted. (8 to >6) | Coding style mostly consistent with few deviations from standards. Code is generally readable. (6 to >4) | Code adheres to conventions with inconsistent formatting. (4 to >2) | Not yet able to adhere to coding style standards. (2 to >0) | /10 |
