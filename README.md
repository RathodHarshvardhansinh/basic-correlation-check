# Task 35 – Basic Correlation Check

**Track:** Data Analytics · **Level 1 · Day 35**
**Tools:** Python, Pandas (plus Matplotlib for plots)
**Dataset:** Iris (150 rows, 4 numeric variables)

## Objective

Check relationships between selected numeric variables and introduce correlation carefully — inspecting scatterplots before trusting a number, and not inferring causation.

## Approach

1. Loaded the Iris dataset and kept the four numeric columns: `sepal_length`, `sepal_width`, `petal_length`, `petal_width`.
2. Computed the Pearson correlation matrix with `DataFrame.corr()` and ranked all variable pairs by absolute correlation.
3. Plotted a heatmap and scatterplots of a strong-positive pair, another strong pair, and a near-zero pair.
4. Ran one extra check: recomputed `sepal_length` vs `sepal_width` within each species.

## Correlation table (Pearson r)

|               | sepal_length | sepal_width | petal_length | petal_width |
|---------------|:-----------:|:-----------:|:------------:|:-----------:|
| sepal_length  | 1.000 | -0.118 | 0.872 | 0.818 |
| sepal_width   | -0.118 | 1.000 | -0.428 | -0.366 |
| petal_length  | 0.872 | -0.428 | 1.000 | 0.963 |
| petal_width   | 0.818 | -0.366 | 0.963 | 1.000 |

![Heatmap](outputs/correlation_heatmap.png)
![Scatterplots](outputs/scatterplots.png)

## Observations

1. **Petal length and petal width are almost linearly related (r = 0.96).** The scatterplot shows a clear upward trend, but also two separate clusters (small-petal setosa vs. the other species), so part of this strength comes from group differences.
2. **Sepal length vs sepal width is near zero overall (r = -0.12), but that hides structure.** The scatterplot is a diffuse cloud, yet within each species the correlation is positive (setosa 0.74, versicolor 0.53, virginica 0.46). Mixing groups cancelled the relationship out — a reminder to plot and segment before reading a single r.
3. **Sepal width is moderately negatively correlated with the petal measurements (r ≈ -0.37 to -0.43).** This is also influenced by species mixing, and should be treated as a pattern in this sample, not a rule.

> Correlation is not causation: none of these results say that one measurement drives another. They only describe how the variables move together in this dataset.

## Interview questions

**Correlation vs causation?**
Correlation measures how two variables move together; causation means a change in one produces a change in the other. A third factor (here, species) or coincidence can create correlation without any causal link, so causal claims need experiments or careful causal design.

**What does correlation near zero mean?**
There is no *linear* relationship between the variables. It does not mean they are unrelated — a curved relationship or mixed subgroups (as with sepal length/width above) can also give r ≈ 0, which is why scatterplots matter.

## Notes

- The suggested Housing dataset was not used: it needs a download and this analysis was built offline. Iris is bundled with scikit-learn, so the project runs without internet.

## How to run

```bash
pip install -r requirements.txt
python correlation_check.py
```

Outputs are written to `outputs/`: `correlation_table.csv`, `ranked_pairs.csv`, `sepal_corr_by_species.csv`, `correlation_heatmap.png`, `scatterplots.png`.
