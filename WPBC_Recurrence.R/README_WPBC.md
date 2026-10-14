# Predicting Breast Cancer Recurrence: A Statistical Learning Analysis of the Wisconsin Prognostic Breast Cancer (WPBC) Data

Comparison of seven classification methods for predicting breast cancer recurrence, with a focus on interpreting results honestly on a small, imbalanced data set. All code is in R.

Term project for *Statistical Inference* (Master's level), Paris Lodron University of Salzburg, Summer Semester 2026.

## Summary

After breast cancer surgery, a fraction of patients experience a recurrence. This project asks whether features available at the time of surgery can identify which patients are at higher risk. It is a **prognosis** problem (predicting a future event), unlike the more common diagnosis version of this data set (benign vs malignant).

- **Predictive power is modest.** On the held-out test set, AUCs range from 0.43 to 0.57 across methods, close to chance. Only 46 of 194 patients recurred.
- **Accuracy is misleading.** A rule that always predicts "no recurrence" scores 77.6% on the test set. Most models matched that number by flagging almost no recurrences (sensitivity near 0).
- **Nodal status is the most consistent signal.** The number of positive lymph nodes (`pnodes`) ranks first in both random forest and boosting variable importance.
- **Nuclear morphology adds information as a block.** A likelihood-ratio test shows the four "worst" nuclear features jointly improve a model that already contains tumour size and nodal status (p = 0.037), even though only one of them is individually significant.
- **Simple beat complex.** The reduced logistic regression gave the best-ranked risk scores, and the tree ensembles did not clearly outperform it.

## Data

The Wisconsin Prognostic Breast Cancer data come from the UCI Machine Learning Repository (Wolberg, Street and Mangasarian) and are also bundled in the CRAN package `TH.data`. This is a public, anonymised research data set with no patient identifiers.

- **198 patients**, each described by an outcome (`status`: recurred `R` or not `N`), follow-up time, 30 features computed from a digitised image of a fine needle aspirate (the mean, standard error and "worst" value of ten nuclear characteristics such as radius, texture and concavity), tumour diameter (`tsize`) and the number of positive lymph nodes (`pnodes`).
- **Four patients with missing `pnodes` were dropped** (under 2.1%), leaving **194 patients**: 148 without recurrence and 46 with (23.7%).
- **The follow-up time column is not used.** Using it as a predictor would leak information only known in hindsight, and modelling it properly needs survival analysis (see Limitations).

The cleaned file `data/wpbc_clean.csv` is derived from the public dataset from the UCI

## Method

1. **Split.** A single 70/30 train/test split with `set.seed(1)`: 136 training patients (33 recurrences) and 58 test patients (13 recurrences). The test set is used once, for the final comparison.
2. **Tuning.** The lasso penalty and the tree's cost-complexity parameter are chosen by 10-fold cross-validation on the training set only, and the number of boosting trees by 5-fold cross-validation.
3. **Models.**
   - **Reduced logistic regression:** one representative feature per nuclear characteristic (the "worst" value of radius, texture, concave points and symmetry) plus `tsize` and `pnodes`, because the 30 features are highly collinear. Variance inflation factors are 1.01 to 1.69.
   - **Likelihood-ratio test:** the four nuclear features as a block against a model with only `tsize` and `pnodes`.
   - **Lasso and ridge logistic regression** (`glmnet`), using the most regularised penalty within one standard error of the minimum (`lambda.1se`).
   - **Decision tree:** CART with cost-complexity pruning (`rpart`).
   - **Bagging and random forest:** 500 trees each, with `mtry` equal to all predictors and to the square root of the number of predictors respectively.
   - **Class-balanced random forest:** equal numbers of recurrence and non-recurrence cases drawn into each bootstrap sample.
   - **Boosting:** Bernoulli deviance, interaction depth 2, shrinkage 0.01, up to 1,000 trees (`gbm`).
4. **Evaluation.** Accuracy, sensitivity (recall for recurrence), specificity and AUC on the test set, at a 0.5 threshold, plus ROC curves.

## Results

Test-set performance (threshold 0.5; sensitivity is the share of recurrences correctly flagged):

| Model | Accuracy | Sensitivity | Specificity | AUC |
|---|---|---|---|---|
| Logistic (reduced) | 0.776 | 0.077 | 0.978 | 0.573 |
| Lasso logistic | 0.776 | 0.000 | 1.000 | 0.500 |
| Pruned tree | 0.776 | 0.000 | 1.000 | 0.500 |
| Bagging | 0.707 | 0.000 | 0.911 | 0.436 |
| Random forest | 0.776 | 0.000 | 1.000 | 0.430 |
| Random forest (balanced) | 0.672 | 0.154 | 0.822 | 0.517 |
| Boosting | 0.776 | 0.000 | 1.000 | 0.544 |

The naive "always no recurrence" baseline has accuracy 0.776. Ridge was fitted for comparison but is not in the test evaluation.

Other findings:

- **Logistic regression:** only `worst_radius` is individually significant (odds ratio 1.19, 95% interval 1.06 to 1.35, p = 0.006), meaning about 19% higher odds of recurrence per unit increase, holding the other features fixed. `tsize` and `pnodes` are positive but not individually significant (p = 0.40 and 0.17).
- **Likelihood-ratio test:** the four nuclear features as a block improve the fit (deviance 10.2 on 4 degrees of freedom, p = 0.037).
- **Lasso:** at `lambda.1se` the lasso shrinks every coefficient to zero (the printed output shows only an intercept), which matches its chance-level AUC of exactly 0.500.
- **Pruned tree:** cross-validated error is lowest at the root, so the pruned tree is a single node predicting "no recurrence" for everyone. With 33 training recurrences, splits that reduce training error do not generalise.
- **Variable importance:** `pnodes` ranks first in both random forest (mean decrease in Gini) and boosting. `tsize` ranks sixth in the random forest and third in boosting.
- **Class imbalance:** the balanced random forest trades accuracy (0.672) for the highest sensitivity among the ensembles (0.154).
- **Operating threshold:** Youden's index on the logistic model's ROC suggests a cutoff near 0.26 instead of 0.5, which flags 6 of 13 recurrences (46% sensitivity) at the cost of 10 false alarms among 45 non-recurrent patients (78% specificity). This threshold was chosen on the test set itself, so these figures are optimistic.

## Limitations

- **Censoring.** Treating `status` as a plain binary outcome counts non-recurrent patients as truly disease-free, although some may simply have short follow-up. A Kaplan-Meier or Cox proportional-hazards analysis would use the follow-up time properly. This is the most important methodological caveat.
- **Small sample.** With 194 patients, 46 recurrences and only 13 recurrences in the test set, all estimates carry large uncertainty. A different random split could change the ranking of methods.
- **Single split and optimistic threshold.** One train/test split is used, and the Youden threshold was selected on the test set. A cleaner design would choose it by cross-validation on the training data.
- **Multiplicity.** Over 30 correlated candidate predictors were considered, with no multiple-testing correction, so individual coefficients and importance rankings should be read as suggestive.
- **Collinearity.** Several "worst" nuclear features proxy for the same tumour-severity signal, so which one a method selects is somewhat arbitrary.
- **Not clinical advice.** This is a methods exercise on a public benchmark data set, not a validated clinical tool.

## Folder structure

```
WPBC_Recurrence.R/
  README.md
  WPBCR_Code.R               # standalone script: data preparation, models, evaluation
```

### Requirements

R (the report was produced with R 4.3.3) and the packages `rpart`, `randomForest`, `gbm`, `glmnet`, `pROC` and `knitr`. The report used `rpart` 4.1.23, `randomForest` 4.7-1.1, `gbm` 2.1.8.1, `glmnet` 4.1-8 and `pROC` 1.18.5.

### Running

1. Rebuild wbpc_clean.csv manually with software of choice (R or SQL recommended) using 'TH.data' or download from UCI site
2. Set the working directory to the repository root and run `WPBCR_Code.R` from top to bottom.
3. Results depend on the random seed. The train/test split uses `set.seed(1)`, and the forests and boosting are random, so small differences between runs are expected unless seeds are set for those steps too.

## Authorship and tools

[Claude (Anthropic) Assisted with code descriptions in WBBCR_Code.R, Class balanced Random Forest and Boosting Specifications]

## References

- James, G., Witten, D., Hastie, T. and Tibshirani, R. (2023). *An Introduction to Statistical Learning with Applications in R*, 2nd edition.
- Wolberg, W.H., Street, W.N. and Mangasarian, O.L. *Breast Cancer Wisconsin (Prognostic) Data Set.* UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/16
- Data also available through the CRAN package `TH.data` (`wpbc`). R packages: `rpart`, `randomForest`, `gbm`, `glmnet`, `pROC`.
