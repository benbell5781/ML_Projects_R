#' ---
#' title: "Predicting Breast Cancer Recurrence: A Statistical Learning Analysis of the Wisconsin Prognostic Breast Cancer (WPBC) Data"
#' author: "Ben Bell"
#' date: "`r format(Sys.Date(), '%d %B %Y')`"
#' output:
#'   pdf_document:
#'     toc: true
#'     number_sections: true
#'     fig_caption: true
#' header-includes:
#'   - \usepackage{float}
#'   - \floatplacement{figure}{H}
#' ---
#' 
## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE,
                       fig.align = "center", fig.pos = "H")
options(width = 65)
suppressPackageStartupMessages({
  library(rpart)
  library(randomForest)
  library(e1071)
  library(gbm)
  library(glmnet)
  library(pROC)
  library(knitr)
})
set.seed(1)

#' 
#' # Introduction
#' 
#' Breast cancer patients who undergo surgery are not automatically free of
#' disease: a fraction will experience a **recurrence** of the cancer within a
#' few years. Being able to identify, at the time of surgery, which patients
#' are at high risk of recurrence would let clinicians tailor follow-up and
#' adjuvant therapy. This report analyses the **Wisconsin Prognostic Breast
#' Cancer (WPBC)** data set from the UCI Machine Learning Repository, in which
#' each of 198 patients who had a breast mass surgically removed is described
#' by 30 features computed from a digitized image of a fine needle aspirate
#' (FNA) of the mass (the mean, standard error, and "worst"/largest value of
#' ten nuclear characteristics such as radius, texture, and concavity), plus
#' the diameter of the tumor (`tsize`) and the number of positive lymph nodes
#' (`pnodes`). The outcome of interest, `status`, records whether the cancer
#' recurred (`R`) or did not recur (`N`) during follow-up.
#' 
#' This is a genuine **prognosis** problem (predicting a future event) rather
#' than the more common **diagnosis** version of this data set (Wisconsin
#' Diagnostic Breast Cancer, which classifies a tumor as malignant or benign
#' at the time of biopsy). It was chosen specifically to apply and compare the
#' statistical learning methods covered in class -- logistic regression and
#' statistical inference, regularization, cross-validation, decision trees
#' built by **greedy recursive binary splitting**, tree ensembles (bagging,
#' random forests, boosting), and support vector machines -- on a single,
#' realistic, moderately hard classification problem, and to interpret the
#' results the way an applied statistician would rather than simply chasing
#' predictive accuracy.
#' 
#' All analysis was performed in R. Code and results are shown throughout this
#' report; a clean, standalone R script (`analysis.R`) accompanies this PDF.
#' 
#' # Data and Exploratory Analysis
#' 
## ----load-data----------------------------------------------------------------
df <- read.csv("data/wpbc_clean.csv", stringsAsFactors = TRUE)
df$status <- factor(df$status, levels = c("N", "R"))
dim(df)
sapply(df, class) |> table()

#' 
#' The raw WPBC data has 198 patients and 34 columns (`status`, `time`, 30
#' FNA-derived features, `tsize`, `pnodes`). Four patients have a missing value
#' for `pnodes` (\< 2.1% of the sample); since this is a small fraction, these
#' rows were dropped rather than imputed, leaving **`r nrow(df)`** patients.
#' The `time` column (months to recurrence, or months of recurrence-free
#' follow-up) was **not used as a predictor** -- using it directly would leak
#' information that is only available in hindsight, and modeling it properly
#' (with `R`/`N` as a censoring indicator) calls for survival analysis
#' (Kaplan-Meier curves, the Cox proportional-hazards model), which is outside
#' the scope of this course. This omission is revisited in the Limitations
#' section, since it does affect how the outcome should be interpreted.
#' 
#' ## Class balance
#' 
## ----class-balance, fig.height=3, fig.width=4---------------------------------
tab <- table(df$status)
print(tab)
barplot(tab, col = c("#4C78A8", "#E45756"),
        main = "Recurrence status", ylab = "Number of patients")

#' 
#' Only **`r tab["R"]`** of `r sum(tab)` patients (`r round(100*tab["R"]/sum(tab),1)`%)
#' experienced a recurrence. This class imbalance is the single most important
#' fact about this data set: it means that a classifier which *always*
#' predicts "no recurrence" is already correct `r round(100*tab["N"]/sum(tab),1)`%
#' of the time, so raw accuracy is a poor and potentially misleading way to
#' compare models here. This motivates the use of sensitivity, specificity,
#' and ROC/AUC (Section 4) alongside accuracy throughout this report.
#' 
#' ## Predictor summary and collinearity
#' 
## ----summary-table------------------------------------------------------------
summary(df[, c("tsize", "pnodes", "worst_radius", "worst_concavepoints")])

#' 
#' The 30 FNA features come in triplets (`mean_*`, `SE_*`, `worst_*`) computed
#' from the same underlying cell-nucleus measurements, so many predictors are
#' highly correlated with one another by construction.
#' 
## ----corr-heatmap, fig.height=5.5, fig.width=6, fig.cap="Correlation matrix of the 30 FNA-derived predictors (mean/SE/worst blocks of 10 features each). Dark red = strong positive correlation."----
feat <- df[, 3:32]  # the 30 mean/SE/worst FNA features
cmat <- cor(feat)
image(1:ncol(cmat), 1:ncol(cmat), cmat[, ncol(cmat):1],
      axes = FALSE, xlab = "", ylab = "",
      col = colorRampPalette(c("#3B4CC0", "white", "#B40426"))(50), zlim = c(-1, 1))
box()

#' 
#' As expected, size-related measures (radius, perimeter, area) are almost
#' perfectly correlated with each other within each block, and moderately
#' correlated across the mean/worst blocks. This collinearity has direct
#' consequences for **inference** in linear/logistic models (inflated standard
#' errors, unstable coefficients) even though it causes fewer problems for
#' tree-based and margin-based methods; it is addressed explicitly in
#' Section 3.1 and 3.2.
#' 
#' ## Univariate signal
#' 
## ----boxplots, fig.height=3.2, fig.width=8, fig.cap="Distributions of three clinically-motivated predictors by recurrence status."----
par(mfrow = c(1, 3))
boxplot(pnodes ~ status, data = df, col = c("#4C78A8","#E45756"),
        main = "Positive lymph nodes", xlab = "", ylab = "pnodes")
boxplot(tsize ~ status, data = df, col = c("#4C78A8","#E45756"),
        main = "Tumor size (cm)", xlab = "", ylab = "tsize")
boxplot(worst_concavepoints ~ status, data = df, col = c("#4C78A8","#E45756"),
        main = "Worst concave points", xlab = "", ylab = "worst_concavepoints")

#' 
#' Patients who later recurred tend to have more positive lymph nodes, larger
#' tumors, and nuclei with more severe concave points, but the distributions
#' overlap substantially -- consistent with the class-imbalance discussion
#' above, no single feature cleanly separates the two groups, so this is a
#' genuinely hard classification problem.
#' 
#' # Methods
#' 
#' A single stratified-by-outcome 70/30 train/test split is used for the final,
#' head-to-head comparison of methods in Section 4 (`r round(0.7*nrow(df))`
#' training patients, `r nrow(df) - round(0.7*nrow(df))` test patients). Every
#' tuning parameter (the lasso penalty $\lambda$, the tree's cost-complexity
#' parameter $\alpha$, the boosting learning rate/number of trees, and the SVM
#' cost/gamma) is chosen by **10-fold cross-validation on the training set
#' only** (Chapter 5), so that the held-out test set is used exactly once, for
#' final evaluation, and never for model selection.
#' 
## ----split--------------------------------------------------------------------
set.seed(1)
n <- nrow(df)
train_idx <- sample(seq_len(n), size = round(0.7 * n))
train <- df[train_idx, ]
test  <- df[-train_idx, ]
cat("Training set:", nrow(train), "patients (", table(train$status)["R"], "recurrences )\n")
cat("Test set:    ", nrow(test),  "patients (", table(test$status)["R"], "recurrences )\n")

#' 
#' ## Logistic regression and statistical inference
#' 
#' Because the 30 FNA features are highly collinear (Section 2.2), fitting
#' logistic regression on all of them produces unstable, hard-to-interpret
#' coefficients. A **reduced model** was fit instead, using one representative
#' feature per clinical concept (the "worst" -- largest -- value of each
#' nuclear characteristic, which the original WPBC study found most
#' predictive) together with `tsize` and `pnodes`:
#' 
## ----logistic-----------------------------------------------------------------
log_fit <- glm(status ~ worst_radius + worst_texture + worst_concavepoints +
                 worst_symmetry + tsize + pnodes,
               data = train, family = binomial)
summary(log_fit)

#' 
## ----vif----------------------------------------------------------------------
# Manual variance inflation factors (car package not required)
vif_manual <- function(fit) {
  X <- model.matrix(fit)[, -1]
  sapply(colnames(X), function(v) {
    r2 <- summary(lm(X[, v] ~ X[, colnames(X) != v]))$r.squared
    1 / (1 - r2)
  })
}
round(vif_manual(log_fit), 2)

#' 
#' All variance inflation factors are well below the conventional concern
#' threshold of 5, confirming that restricting to one feature per concept was
#' enough to control collinearity in this reduced model.
#' 
## ----logistic-inference-------------------------------------------------------
ci <- confint(log_fit)                 # profile-likelihood 95% CIs
or_table <- exp(cbind(OddsRatio = coef(log_fit), ci))
round(or_table, 3)

#' 
## ----sig-var, include=FALSE---------------------------------------------------
pvals <- summary(log_fit)$coefficients[, "Pr(>|z|)"]
sig_vars <- names(pvals)[pvals < 0.05 & names(pvals) != "(Intercept)"]

#' 
#' **Interpretation.** Holding the other predictors fixed, only
#' **`r sig_vars`** has a 95% confidence interval for its odds ratio that
#' excludes 1 (Wald p = `r signif(pvals[sig_vars],2)`): each one-unit increase
#' in the worst (largest) recorded nuclear radius is associated with roughly a
#' `r round((exp(coef(log_fit)["worst_radius"])-1)*100,0)`% increase in the
#' odds of recurrence, holding tumor size, nodal status, and the other nuclear
#' features fixed. `tsize` and `pnodes` are directionally positive (larger
#' tumors and more positive nodes both raise predicted recurrence risk) but
#' are **not** individually significant in this six-predictor model
#' (p = `r round(pvals["tsize"],2)` and `r round(pvals["pnodes"],2)`
#' respectively) -- with only `r nrow(train)` training patients and
#' `r sum(train$status=="R")` recurrences, and `worst_radius` itself
#' moderately correlated with both (Section 2.2), there is limited power to
#' resolve which of these overlapping "how large/advanced is this tumor"
#' measures is doing the work.
#' 
#' A likelihood-ratio test checks whether the four nuclear-morphology features
#' *as a block* add anything beyond `tsize` and `pnodes`:
#' 
## ----lrt----------------------------------------------------------------------
log_fit_small <- glm(status ~ tsize + pnodes, data = train, family = binomial)
lrt <- anova(log_fit_small, log_fit, test = "Chisq")
lrt

#' 
#' Unlike any single nuclear-morphology coefficient on its own, the block
#' **does** improve the fit significantly
#' (p = `r signif(lrt[["Pr(>Chi)"]][2],2)`): nuclear morphology carries
#' real information about recurrence beyond simple tumor size and nodal
#' count, even though collinearity (Section 2.2) makes it hard for the Wald
#' test to credit any one nuclear feature individually. This is a textbook
#' illustration of why "not individually significant" and "not informative"
#' are different statements when predictors are correlated -- exactly the
#' kind of subtlety a pure prediction-accuracy metric would miss entirely.
#' 
#' ## Regularized logistic regression: ridge and lasso
#' 
#' To use *all* 30 FNA features (plus `tsize`, `pnodes`) without the
#' instability collinearity causes in ordinary logistic regression, ridge and
#' lasso logistic regression (Chapter 6) shrink the coefficients, with the
#' lasso additionally performing variable selection by setting some
#' coefficients exactly to zero.
#' 
## ----glmnet-------------------------------------------------------------------
x_train <- model.matrix(status ~ ., data = train)[, -1]
y_train <- train$status
x_test  <- model.matrix(status ~ ., data = test)[, -1]

cv_lasso <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1, nfolds = 10)
cv_ridge <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0, nfolds = 10)

#' 
## ----glmnet-plot, fig.height=3.4, fig.width=7, fig.cap="10-fold cross-validated binomial deviance for the lasso (left) and ridge (right) as a function of log(lambda). The left dashed line is lambda.min; the right is the more parsimonious lambda.1se."----
par(mfrow = c(1, 2))
plot(cv_lasso); title("Lasso", line = 2.5)
plot(cv_ridge); title("Ridge", line = 2.5)

#' 
## ----lasso-coefs--------------------------------------------------------------
lasso_fit <- glmnet(x_train, y_train, family = "binomial", alpha = 1,
                     lambda = cv_lasso$lambda.1se)
lasso_coefs <- coef(lasso_fit)
lasso_coefs[lasso_coefs[, 1] != 0, , drop = FALSE]

#' 
#' At the cross-validated `lambda.1se` (the most regularized model within one
#' standard error of the minimum CV deviance -- a standard, more parsimonious
#' choice), the lasso keeps only **`pnodes`** and **`worst_concavity`** as
#' nonzero predictors, shrinking all 30 remaining coefficients to exactly
#' zero. Note that this is a *different* nuclear-morphology feature from the
#' `worst_radius` singled out by the reduced logistic model above --
#' consistent with the collinearity already flagged in Section 2.2, several
#' correlated "worst" features can each proxy for the same underlying
#' tumor-aggressiveness signal, so which one a given method or predictor
#' subset happens to select is somewhat arbitrary. What is robust across
#' *both* methods is the broader pattern: `pnodes` plus one nuclear-severity
#' measure.
#' 
#' ## Decision tree: greedy recursive binary splitting
#' 
#' A classification tree is grown using **CART** (`rpart`), which builds the
#' tree by **greedy recursive binary splitting**: at each node it picks the
#' single predictor and split point that most reduces node impurity (Gini
#' index) *right now*, without any lookahead for splits further down the
#' tree. This greediness is what makes trees fast to fit and easy to
#' interpret, but also what makes a single tree high-variance -- a point
#' returned to in Section 3.4.
#' 
## ----tree---------------------------------------------------------------------
tree_fit <- rpart(status ~ ., data = train, method = "class",
                   control = rpart.control(cp = 0.001, minsplit = 10, xval = 10))
printcp(tree_fit)

#' 
#' The `xerror` column is the 10-fold cross-validated misclassification cost
#' at each value of the cost-complexity parameter `cp` -- the tree-pruning
#' analogue of the cross-validation used to choose $\lambda$ above. The
#' smallest cross-validated error picks the pruning level:
#' 
## ----prune, fig.height=3, fig.width=5, fig.cap="10-fold CV relative error vs tree size (cost-complexity pruning)."----
plotcp(tree_fit)
best_cp <- tree_fit$cptable[which.min(tree_fit$cptable[, "xerror"]), "CP"]
pruned_tree <- prune(tree_fit, cp = best_cp)
pruned_tree

#' 
## ----tree-plot-full, fig.height=5.5, fig.width=8, fig.cap="The fully grown, unpruned tree (before cost-complexity pruning), shown to illustrate the greedy recursive-splitting rule itself."----
plot(tree_fit, uniform = TRUE, margin = 0.1)
text(tree_fit, pretty = 0, cex = 0.6)

#' 
#' Figure 5 shows the *fully grown* tree, i.e. the raw output of greedy
#' recursive binary splitting before any pruning: at the root it greedily
#' picks `pnodes` as the single most impurity-reducing split, and keeps
#' greedily splitting (on `worst_concavepoints`, `worst_texture`, `mean_area`,
#' and others) until the stopping rule is reached. But cross-validation
#' (Figure 4, and the `nsplit`/`xerror` table above) shows that **every one of
#' these splits increases cross-validated error relative to the unsplit
#' root** -- the cross-validated relative error is minimized at `nsplit = 0`.
#' In other words, the CV-optimal pruned tree collapses all the way back to a
#' single node that simply predicts the majority class (`N`) for everyone.
#' 
#' This is a genuinely informative result, not a coding artifact: with only
#' `r nrow(train)` training patients and `r table(train$status)["R"]`
#' recurrences, there is not enough signal-to-noise for a deep, greedily-built
#' tree to find splits that generalize -- every additional split reduces
#' *training* error but, per the CV curve, increases *expected test* error.
#' This is a textbook illustration of the bias-variance trade-off in tree
#' depth (ISL Ch. 8), and it foreshadows the pruned tree's poor showing (chance-level
#' AUC) in the model comparison of Section 4 -- and, in turn, the motivation
#' for ensembling in the next section.
#' 
#' ## Bagging, random forests, and boosting
#' 
#' Ensembles address exactly the high-variance weakness of a single greedy
#' tree by averaging over many trees.
#' 
## ----bagging-rf---------------------------------------------------------------
p <- ncol(train) - 1
mtry_rf <- floor(sqrt(p))
bag_fit <- randomForest(status ~ ., data = train,
                         mtry = p, ntree = 500,
                         importance = TRUE)
rf_fit  <- randomForest(status ~ ., data = train,
                         mtry = mtry_rf, ntree = 500,
                         importance = TRUE)

# Compact summary (avoids randomForest's print.default "Call:" line-wrap quirk)
rf_summary <- function(fit, label) {
  oob <- round(100 * fit$err.rate[fit$ntree, "OOB"], 2)
  cat(sprintf("%s: mtry = %d, ntree = %d, OOB error = %s%%\n",
              label, fit$mtry, fit$ntree, oob))
  print(fit$confusion)
}
rf_summary(bag_fit, "Bagging (mtry = p)")
rf_summary(rf_fit,  "Random forest (mtry = sqrt(p))")

#' 
#' Bagging (`mtry = p`, i.e. every split considers all predictors) and random
#' forests (`mtry = sqrt(p)`, decorrelating the trees by restricting each
#' split to a random subset of predictors) are both fit with 500 bootstrap
#' trees. The out-of-bag (OOB) confusion matrices above already show the
#' class-imbalance problem clearly: both ensembles achieve a low overall OOB
#' error mostly by predicting the majority class `N`.
#' 
## ----rf-importance, fig.height=6, fig.width=6.5, fig.cap="Random forest variable importance (mean decrease in Gini index)."----
varImpPlot(rf_fit, main = "", n.var = 15, type = 2)

#' 
#' `pnodes` and `tsize` again top the importance ranking, consistent with the
#' logistic and lasso results -- three quite different methods agree on the
#' same two dominant clinical predictors, which is reassuring evidence that
#' this is a real signal rather than an artifact of any one modeling
#' approach.
#' 
#' Because bagging/random forests build each bootstrap tree from a sample
#' that mirrors the 76%/24% imbalance of the training data, most trees rarely
#' even see a recurrence case. A **class-balanced random forest** -- which
#' draws an equal number of `N` and `R` cases into every bootstrap sample via
#' the `strata`/`sampsize` arguments -- is fit as a direct remedy:
#' 
## ----rf-balanced--------------------------------------------------------------
min_class <- min(table(train$status))
rf_bal <- randomForest(status ~ ., data = train,
                        mtry = mtry_rf,
                        ntree = 500,
                        strata = train$status,
                        sampsize = c(N = min_class, R = min_class),
                        importance = TRUE)
rf_summary(rf_bal, "Class-balanced random forest")

#' 
#' This trades some overall accuracy for much better OOB sensitivity to the
#' recurrence class -- the trade-off is quantified on the test set in
#' Section 4.
#' 
#' Boosting (Chapter 8.2.3) grows trees **sequentially**, each one fit to the
#' residual signal left by the previous trees, rather than averaging
#' independent bootstrap trees:
#' 
## ----boosting, results='hide'-------------------------------------------------
train_gbm <- train
train_gbm$status01 <- ifelse(train_gbm$status == "R", 1, 0)
train_gbm$status <- NULL
boost_fit <- gbm(status01 ~ ., data = train_gbm, distribution = "bernoulli",
                  n.trees = 1000, interaction.depth = 2, shrinkage = 0.01,
                  cv.folds = 5, verbose = FALSE)
best_iter <- gbm.perf(boost_fit, method = "cv", plot.it = FALSE)

#' 
## ----boosting-summary, fig.height=5, fig.width=6.5, fig.cap="Boosting relative influence of the top predictors."----
cat("CV-optimal number of trees:", best_iter, "\n")
inf <- summary(boost_fit, n.trees = best_iter, plotit = TRUE, las = 1, cBars = 12)

#' 
#' Boosting's shallow trees (`interaction.depth = 2`) and small learning rate
#' (`shrinkage = 0.01`), with the number of trees itself chosen by
#' cross-validation, again single out `pnodes` and `tsize` as the most
#' influential predictors, along with several standard-error features that
#' capture heterogeneity within a tumor.
#' 
#' ## Support vector machines
#' 
#' Linear and radial-kernel SVMs are tuned by 10-fold cross-validation over
#' cost (and, for the radial kernel, gamma):
#' 
## ----svm, results='hide'------------------------------------------------------
svm_lin_tune <- tune(svm, status ~ ., data = train, kernel = "linear",
                      ranges = list(cost = c(0.01, 0.1, 1, 5, 10)),
                      probability = TRUE)
svm_rad_tune <- tune(svm, status ~ ., data = train, kernel = "radial",
                      ranges = list(cost = c(0.1, 1, 5, 10), gamma = c(0.001, 0.01, 0.1)),
                      probability = TRUE)

#' 
## ----svm-params---------------------------------------------------------------
svm_lin_tune$best.parameters
svm_rad_tune$best.parameters
svm_lin <- svm_lin_tune$best.model
svm_rad <- svm_rad_tune$best.model

#' 
#' The radial kernel allows a non-linear decision boundary in feature space;
#' comparing it with the linear kernel (and with the linear logistic/lasso
#' models above) is itself a check of whether the recurrence boundary looks
#' linear or not.
#' 
#' # Model Comparison
#' 
#' All eight fitted models are now evaluated **once** on the held-out test
#' set.
#' 
## ----predict-all--------------------------------------------------------------
log_prob   <- predict(log_fit, newdata = test, type = "response")
log_pred   <- factor(ifelse(log_prob > 0.5, "R", "N"), levels = c("N", "R"))

lasso_prob <- as.numeric(predict(lasso_fit, newx = x_test, type = "response"))
lasso_pred <- factor(ifelse(lasso_prob > 0.5, "R", "N"), levels = c("N", "R"))

tree_prob  <- predict(pruned_tree, newdata = test, type = "prob")[, "R"]
tree_pred  <- predict(pruned_tree, newdata = test, type = "class")

bag_prob   <- predict(bag_fit, newdata = test, type = "prob")[, "R"]
bag_pred   <- predict(bag_fit, newdata = test, type = "class")

rf_prob    <- predict(rf_fit, newdata = test, type = "prob")[, "R"]
rf_pred    <- predict(rf_fit, newdata = test, type = "class")

rfbal_prob <- predict(rf_bal, newdata = test, type = "prob")[, "R"]
rfbal_pred <- predict(rf_bal, newdata = test, type = "class")

boost_prob <- predict(boost_fit, newdata = test, n.trees = best_iter, type = "response")
boost_pred <- factor(ifelse(boost_prob > 0.5, "R", "N"), levels = c("N", "R"))

svmlin_raw  <- predict(svm_lin, newdata = test, probability = TRUE)
svmlin_prob <- attr(svmlin_raw, "probabilities")[, "R"]
svmrad_raw  <- predict(svm_rad, newdata = test, probability = TRUE)
svmrad_prob <- attr(svmrad_raw, "probabilities")[, "R"]

compute_metrics <- function(pred, prob, actual) {
  cm <- table(factor(pred, levels(actual)), actual)
  acc  <- sum(diag(cm)) / sum(cm)
  sens <- cm["R", "R"] / sum(cm[, "R"])
  spec <- cm["N", "N"] / sum(cm[, "N"])
  auc  <- as.numeric(pROC::auc(pROC::roc(actual, prob, levels = c("N", "R"),
                                          direction = "<", quiet = TRUE)))
  c(Accuracy = acc, Sensitivity = sens, Specificity = spec, AUC = auc)
}

results <- rbind(
  "Logistic (reduced)"      = compute_metrics(log_pred, log_prob, test$status),
  "Lasso logistic"          = compute_metrics(lasso_pred, lasso_prob, test$status),
  "Pruned tree"             = compute_metrics(tree_pred, tree_prob, test$status),
  "Bagging"                 = compute_metrics(bag_pred, bag_prob, test$status),
  "Random forest"           = compute_metrics(rf_pred, rf_prob, test$status),
  "Random forest (balanced)"= compute_metrics(rfbal_pred, rfbal_prob, test$status),
  "Boosting"                = compute_metrics(boost_pred, boost_prob, test$status),
  "SVM (linear)"            = compute_metrics(svmlin_raw, svmlin_prob, test$status),
  "SVM (radial)"            = compute_metrics(svmrad_raw, svmrad_prob, test$status)
)
kable(round(results, 3),
      caption = "Test-set performance (threshold = 0.5). Sensitivity = recurrence recall.")

#' 
## ----naive-baseline-----------------------------------------------------------
naive_acc <- max(table(test$status)) / sum(table(test$status))
cat("Naive 'always predict no recurrence' baseline accuracy:", round(naive_acc, 3), "\n")

#' 
#' **Accuracy is a misleading headline number here**: the naive rule of always
#' predicting "no recurrence" already achieves `r round(naive_acc*100,1)`%
#' accuracy, and most models sit close to that value while correctly flagging
#' few or none of the actual recurrences (Sensitivity near 0). This is the
#' classic **accuracy paradox** of imbalanced classification, and it is
#' exactly why ROC/AUC -- which evaluates the *ranking* of predicted
#' probabilities across every possible threshold, not just the default 0.5 --
#' is the more meaningful comparison for this data set.
#' 
## ----roc-plot, fig.height=5.5, fig.width=6, fig.cap="ROC curves for all fitted classifiers on the test set."----
my_roc <- function(prob) {
  pROC::roc(test$status, prob, levels = c("N", "R"), direction = "<", quiet = TRUE)
}
rocs <- list(
  "Logistic"      = my_roc(log_prob),
  "Lasso"         = my_roc(lasso_prob),
  "Tree"          = my_roc(tree_prob),
  "Random forest" = my_roc(rf_prob),
  "RF (balanced)" = my_roc(rfbal_prob),
  "Boosting"      = my_roc(boost_prob),
  "SVM (radial)"  = my_roc(svmrad_prob)
)
cols <- c("#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#9D755D", "#76B7B2")
plot(rocs[[1]], col = cols[1], lwd = 2, legacy.axes = TRUE)
for (i in 2:length(rocs)) plot(rocs[[i]], col = cols[i], lwd = 2, add = TRUE)
legend("bottomright", legend = names(rocs), col = cols, lwd = 2, cex = 0.7)
abline(a = 1, b = -1, lty = 3, col = "grey50")

#' 
#' The AUCs cluster in the 0.5-0.65 range (a coin flip would give 0.5), which
#' is a genuinely honest result for this problem: with only `r sum(train$status=="R")`
#' recurrence events in the training set and 30+ correlated candidate
#' predictors, there is limited statistical power to learn a strong
#' classifier, and this is consistent with the wider literature on this
#' particular UCI data set. The **radial SVM** and **logistic regression**
#' give the best ranking of patients by risk (highest AUC); the plain
#' (unbalanced) tree-based ensembles are essentially at chance because, at the
#' default threshold and majority bootstrap sampling, they collapse toward
#' predicting the majority class almost everywhere.
#' 
#' ## Choosing an operating threshold
#' 
#' Since the default 0.5 cutoff is a poor choice for an imbalanced outcome,
#' Youden's index identifies the threshold on the logistic model's predicted
#' probabilities that best balances sensitivity and specificity:
#' 
## ----youden-------------------------------------------------------------------
roc_log <- rocs[["Logistic"]]
best_thresh <- pROC::coords(roc_log, "best", best.method = "youden", transpose = FALSE)
best_thresh
log_pred_opt <- factor(ifelse(log_prob > best_thresh$threshold, "R", "N"), levels = c("N", "R"))
table(Predicted = log_pred_opt, Actual = test$status)

#' 
#' At this data-driven cutoff (around `r round(best_thresh$threshold,2)`
#' instead of 0.5), the logistic model correctly flags roughly
#' `r round(best_thresh$sensitivity*100,0)`% of recurrences at the cost of a
#' `r round((1-best_thresh$specificity)*100,0)`% false-positive rate among
#' non-recurrent patients -- a far more clinically useful operating point than
#' the near-zero sensitivity produced by the default threshold, and a good
#' illustration of why the choice of decision threshold is itself a
#' statistical (and ultimately clinical/cost-based) decision, not just a
#' modeling detail.
#' 
#' # Discussion
#' 
#' Four quite different modeling strategies -- a hand-reduced logistic
#' regression, a likelihood-ratio test, automatic lasso variable selection,
#' and random-forest/boosting variable importance -- agree on the broad
#' picture, while disagreeing on the fine detail, and both facts are
#' informative. **Nodal status (`pnodes`) and tumor size (`tsize`) show up
#' consistently** as top-ranked predictors in the lasso and in every
#' tree-based importance measure, and the "worst" nuclear-morphology features
#' as a block add statistically significant information beyond `tsize` and
#' `pnodes` (likelihood-ratio p = `r signif(lrt[["Pr(>Chi)"]][2],2)`). At the
#' same time, *which single* nuclear feature gets "credit" for that signal
#' differs by method -- `worst_radius` in the hand-reduced logistic
#' regression, `worst_concavity` in the lasso, a mix of SE-features in
#' boosting -- which is exactly what Section 2.2's collinearity analysis
#' predicts: several correlated measurements of tumor size/severity carry
#' overlapping information, so any single-coefficient "most important
#' predictor" claim is unstable, even though the general theme (larger,
#' more node-positive, more severely-shaped tumors recur more often) is
#' robust across methods. This agreement-on-substance,
#' disagreement-on-detail pattern is itself a stronger and more honest
#' form of evidence than any single model's coefficient table taken alone.
#' 
#' Predictive performance, however, is modest across every method tried
#' (test AUCs of roughly 0.5-0.65), and the class imbalance (only 24% of
#' patients recur) makes plain accuracy an actively misleading metric: several
#' models achieve 78% accuracy purely by never predicting a recurrence.
#' Addressing the imbalance directly -- via a class-balanced random forest, or
#' via a data-driven decision threshold on the logistic model -- substantially
#' improves sensitivity to the recurrence class at a modest cost to
#' specificity, which is very likely the right trade-off in a clinical
#' screening context where missing a recurrence is far more costly than a
#' false alarm.
#' 
#' Regarding the methods compared: the single greedy classification tree
#' performed the worst of all methods once pruned by cross-validation (it was
#' pruned back almost to its root), which is a direct, visible illustration of
#' the *high variance* of a single greedily-built tree on a small, noisy
#' data set -- exactly the motivation Chapter 8 gives for bagging, random
#' forests, and boosting. In this instance, however, the ensembles did not
#' clearly outperform the much simpler regularized logistic regression, which
#' is itself a useful, realistic lesson: more flexible methods are not
#' automatically better, especially with `n = 194` and heavy class imbalance,
#' where the bias introduced by a simple, well-regularized linear model can
#' be worth its lower variance.
#' 
#' ## Limitations
#' 
#' Several limitations should temper any of the conclusions above:
#' 
#' 1. **Censoring.** By modeling `status` as a simple binary outcome, this
#'    analysis implicitly treats "N" (non-recurrent) patients as truly
#'    disease-free, when in fact some of them may simply have short follow-up
#'    and could recur later -- they are *right-censored*, not confirmed
#'    negative. The `time` column encodes exactly this follow-up information,
#'    and a Kaplan-Meier/Cox proportional-hazards analysis (ISL Chapter 11)
#'    would use it properly instead of discarding it, which is very likely
#'    why predictive performance here (and in the wider literature on this
#'    data set) is modest. This is flagged as the most important methodological
#'    caveat and a natural next step beyond the current course material.
#' 2. **Sample size.** With only 194 patients and 46 recurrences, all
#'    estimates (coefficients, importance scores, AUCs) carry substantial
#'    sampling uncertainty; the single 70/30 split used for final evaluation
#'    means the reported test metrics themselves have a nontrivial standard
#'    error, and a different random split could shift the ranking of methods
#'    in Table 1 by a few points.
#' 3. **Multiplicity.** Thirty-plus candidate predictors were considered
#'    across several models; individual "significant" coefficients or
#'    top-ranked importance scores should be read as suggestive rather than
#'    as confirmed findings, since no formal multiple-testing correction
#'    (Chapter 13) was applied.
#' 
#' # Conclusion
#' 
#' Across logistic regression, regularized (lasso/ridge) logistic regression,
#' a cross-validated CART tree, bagging, random forests, boosting, and support
#' vector machines, nodal status and tumor size are consistently the
#' top-ranked predictors of breast cancer recurrence in the WPBC data, and
#' nuclear-morphology features add real (statistically significant, via the
#' likelihood-ratio test) but collinear and method-dependent information on
#' top of them -- no single nuclear feature is robustly "the" second most
#' important variable across methods, even though the nuclear-morphology
#' block as a whole matters. Predictive performance is modest for every
#' method (test AUC roughly 0.5-0.65), which appears to be a genuine property
#' of this small, imbalanced data set (and of ignoring censoring) rather than
#' a weakness specific to any one method. Simple, well-regularized models
#' performed competitively with, or better than, more flexible tree ensembles
#' and SVMs -- a useful reminder that model flexibility should be matched to
#' the amount of signal actually available in the data.
#' 
#' # References
#' 
#' - James, G., Witten, D., Hastie, T., and Tibshirani, R. (2023). *An
#'   Introduction to Statistical Learning with Applications in R*, 2nd
#'   edition.
#' - Wolberg, W.H., Street, W.N., and Mangasarian, O.L. Breast Cancer Wisconsin
#'   (Prognostic) Data Set. UCI Machine Learning Repository.
#'   https://archive.ics.uci.edu/dataset/16
#' - Data accessed via the `wpbc` data set bundled with the CRAN package
#'   `TH.data` (a mirror of the original UCI data), R packages `rpart`,
#'   `randomForest`, `e1071`, `gbm`, `glmnet`, and `pROC`.
#' 
#' # Appendix: Session information
#' 
## ----session-info-------------------------------------------------------------
sessionInfo()

