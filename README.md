# ML Algorithm Implementations

**Author:** Bradley Swalberg

A collection of notebooks exploring core machine learning algorithms with scikit-learn — training, evaluating, and tuning hyperparameters across multiple real-world datasets.

## Notebooks

### [`decision-trees.ipynb`](./decision-trees.ipynb)
`DecisionTreeClassifier` and `DecisionTreeRegressor` across the Iris, Congressional Voting Records, and Cars Evaluation datasets, plus a self-selected body fat dataset for regression.
- Baseline fitting and the effect of `max_depth` on accuracy
- Handling missing categorical data via a dedicated "unknown" feature value
- N-fold cross-validation
- Interpreting induced trees and feature importances
- Reducing overfitting: `min_samples_leaf`, `min_samples_split`, `min_impurity_decrease`, and cost-complexity pruning (`ccp_alpha`)
- Regression on a real-valued target (body fat %) with pruning to control overfitting

### [`clustering.ipynb`](./clustering.ipynb)
K-means and Hierarchical Agglomerative Clustering (HAC) on the Abalone and Iris datasets.
- Baseline K-means: cluster centers, inertia, and silhouette scores
- HAC across linkage types (single, complete, average, ward)
- Effect of `n_init` and K-means++ initialization on cluster quality
- Choosing *k* via silhouette analysis across a range of values
- Head-to-head comparison of K-means vs. HAC

### [`backpropagation.ipynb`](./backpropagation.ipynb)
`MLPClassifier` and `MLPRegressor` (multi-layer perceptron) on the Iris and Vowel datasets, plus a large-scale (700K+ row) regression task.
- Diagnosing overfitting and mitigating it with early stopping and L2 regularization
- Hyperparameter sweeps: learning rate, hidden layer width, momentum
- Automated tuning via `GridSearchCV` and `RandomizedSearchCV`, compared for speed and accuracy
- Applying `MLPRegressor` to a large real-world time-series regression problem

## Tools

Python, scikit-learn, pandas, NumPy, Matplotlib
