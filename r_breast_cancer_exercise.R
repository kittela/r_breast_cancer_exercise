options(digits = 3)
library(matrixStats)
library(tidyverse)
library(caret)
library(dslabs)
library(reshape2)
data(brca)

# How many samples are in the dataset?
nrow(brca$x)

# How many predictors are in the matrix?
ncol(brca$x)

# What proportion of the samples are malignant?
mean(brca$y == "M")

# Which column number has the highest mean?
which.max(colMeans(brca$x))

# Which column number has the lowest standard deviation?
which.min(apply(brca$x, 2, sd))

# Use sweep() two times to scale each column: subtract the column means of brca$x, 
# then divide by the column standard deviations of brca$x.

# After scaling, what is the standard deviation of the first column?
scaled_brca <- sweep(brca$x, MARGIN = 2, STATS = colMeans(brca$x)) %>%
  sweep(brca$x, MARGIN = 2, STATS = colSds(brca$x), FUN = "/")
sd(scaled_brca[,1])
median(scaled_brca[,1])

# Calculate the distance between all samples using the scaled matrix.
d <- dist(scaled_brca)
rownames(d) <- rownames(brca$x)
colnames(d) <- rownames(brca$x)
# What is the average distance between the first sample, which is benign, and other benign samples?
d_all_benign <- as.matrix(d)[1,which(brca$y == "B")]
d_first_to_all_b <- abs(d_all_benign[1] - d_all_benign[-1])
mean(d_first_to_all_b)
# What is the average distance between the first sample and malignant samples?
d_all_malignent <- as.matrix(d)[1,which(brca$y == "M")]
d_first_to_all_b <- abs(d_all_benign[1] - d_all_malignent)
mean(d_first_to_all_b)

# Make a heatmap of the relationship between features using the scaled matrix.
d_features <- dist(t(scaled_brca))
heatmap(as.matrix(d_features), labRow = NA, labCol = NA)

# Perform hierarchical clustering on the 30 features. Cut the tree into 5 groups.
h <- hclust(d_features)
groups <- cutree(h, k = 5)
split(names(groups), groups)

# Perform a principal component analysis of the scaled matrix.
# 
# What proportion of variance is explained by the first principal component?
# How many principal components are required to explain at least 90% of the variance?
pca <- prcomp(scaled_brca)
summary(pca)

# Plot the first two principal components with color representing tumor type (benign/malignant).
data.frame(pca$x[,1:2], type = brca$y) %>%
  ggplot(aes(PC1, PC2, color = type)) +
  geom_point()

# Make a boxplot of the first 10 PCs grouped by tumor type.
# 
# Which PCs are significantly different enough by tumor type that there is no overlap
# in the interquartile ranges (IQRs) for benign and malignant samples?
first_10 <- data.frame(type = brca$y, pc1 = pca$x[,1], pc2 = pca$x[,2], pc3 = pca$x[,3],
                       pc4 = pca$x[,4], pc5 = pca$x[,5], pc6 = pca$x[,6], pc7 = pca$x[,7], 
                       pc8 = pca$x[,8], pc9 = pca$x[,9], pc10 = pca$x[,10]) %>%
  melt()

first_10 %>% ggplot(aes(x = variable, y = value, fill = type)) +
  geom_boxplot()

# Set the seed to 1, then create a data partition splitting brca$y and the scaled 
# version of the brca$x matrix into a 20% test set and 80% train using the following 
# code:

set.seed(1, sample.kind = "Rounding")    # if using R 3.6 or later
test_index <- createDataPartition(brca$y, times = 1, p = 0.2, list = FALSE)
test_x <- scaled_brca[test_index,]
test_y <- brca$y[test_index]
train_x <- scaled_brca[-test_index,]
train_y <- brca$y[-test_index]

# Check that the training and test sets have similar proportions of benign and malignant tumors.
#
# What proportion of the training set is benign?
mean(train_y == "B")

# What proportion of the test set is benign?
mean(test_y == "B")

# Set the seed to 3. Perform k-means clustering on the training set with 2 centers 
# and assign the output to k. Then use the predict_kmeans() function to make predictions 
# on the test set.

set.seed(3, sample.kind = "Rounding")
predict_kmeans <- function(x, k) {
  centers <- k$centers    # extract cluster centers
  # calculate distance to cluster centers
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances))  # select cluster with min distance to center
}

# What is the overall accuracy?
k <- kmeans(train_x, 2)
y_hat_kmeans <- predict_kmeans(test_x, k)
accuracy_kmeans <- confusionMatrix(factor(y_hat_kmeans), factor(as.numeric(test_y)))$overall["Accuracy"]
accuracy_kmeans

# What proportion of benign tumors are correctly identified?
sum(y_hat == 1) / sum(test_y == "B")

# What proportion of malignant tumors are correctly identified?
sum(y_hat == 2) / sum(test_y == "M")

# Fit a logistic regression model on the training set with caret::train() using all 
# predictors. Ignore warnings about the algorithm not converging. Make predictions 
# on the test set.

# What is the accuracy of the logistic regression model on the test set?
fit_glm <- train(train_y ~ ., method = "glm", data.frame(train_y, train_x))
y_hat_glm <- predict(fit_glm, test_x)
accuracy_glm <- confusionMatrix(y_hat_glm, test_y)$overall["Accuracy"]
accuracy_glm

# Train an LDA model and a QDA model on the training set. Make predictions on the 
# test set using each model.

# What is the accuracy of the LDA model on the test set?
fit_lda <- train(train_y ~ ., method = "lda", data.frame(train_y, train_x))
y_hat_lda <- predict(fit_lda, test_x)
accuracy_lda <- confusionMatrix(y_hat_lda, test_y)$overall["Accuracy"]
accuracy_lda

# What is the accuracy of the QDA model on the test set?
fit_qda <- train(train_y ~ ., method = "qda", data.frame(train_y, train_x))
y_hat_qda <- predict(fit_qda, test_x)
accuracy_qda <- confusionMatrix(y_hat_qda, test_y)$overall["Accuracy"]
accuracy_qda

# Set the seed to 5, then fit a loess model on the training set with the caret package.
# You will need to install the gam package if you have not yet done so. Use the default 
# tuning grid. This may take several minutes; ignore warnings. Generate predictions 
# on the test set.

set.seed(5, sample.kind = "Rounding")

# What is the accuracy of the loess model on the test set?
fit_loess <- train(train_y ~ ., method = "gamLoess", data.frame(train_y, train_x))
y_hat_loess <- predict(fit_loess, test_x)
accuracy_loess <- confusionMatrix(y_hat_loess, test_y)$overall["Accuracy"]
accuracy_loess

# Set the seed to 7, then train a k-nearest neighbors model on the training set using 
# the caret package. Try odd values of  k  from 3 to 21. Use the final model to generate 
# predictions on the test set.

set.seed(7, sample.kind = "Rounding")

# What is the final value of  k  used in the model?
fit_knn <- train(train_y ~ ., 
                 method = "knn", 
                 data = data.frame(train_y, train_x),
                 tuneGrid = data.frame(k = seq(3, 21, 2)))
plot(fit_knn)
fit_knn$bestTune
y_hat_knn <- predict(fit_knn, test_x)
accuracy_knn <- confusionMatrix(y_hat_knn, test_y)$overall["Accuracy"]
accuracy_knn

# Set the seed to 9, then train a random forest model on the training set using 
# the caret package. Test mtry values of c(3, 5, 7, 9). Use the argument importance = TRUE 
# so that feature importance can be extracted. Generate predictions on the test set.
# Note: please use c(3, 5, 7, 9) instead of seq(3, 9, 2) in tuneGrid.

set.seed(9, sample.kind = "Rounding")

fit_rf <- train(train_y ~ ., 
                 method = "rf", 
                 data = data.frame(train_y, train_x),
                importance = TRUE,
                 tuneGrid = data.frame(mtry = c(3, 5, 7, 9)))
plot(fit_rf)
y_hat_rf <- predict(fit_rf, test_x)

# What value of mtry gives the highest accuracy?
fit_rf$bestTune

# What is the accuracy of the random forest model on the test set?
accuracy_rf <- confusionMatrix(y_hat_rf, test_y)$overall["Accuracy"]
accuracy_rf

# What is the most important variable in the random forest model?
varImp(fit_rf)

# Create an ensemble using the predictions from the 7 models created in the previous 
# exercises: k-means, logistic regression, LDA, QDA, loess, k-nearest neighbors, and 
# random forest. Use the ensemble to generate a majority prediction of the tumor type 
# (if most models suggest the tumor is malignant, predict malignant).

ensemble <- data.frame(kmeans = as.numeric(y_hat_kmeans),
                       glm = as.numeric(y_hat_glm), 
                       lda = as.numeric(y_hat_lda), 
                       qda = as.numeric(y_hat_qda), 
                       loess = as.numeric(y_hat_loess), 
                       knn = as.numeric(y_hat_knn), 
                       rf = as.numeric(y_hat_rf))

ensemble <- ensemble %>% mutate(y_hat = ifelse(rowMeans(.) > 1.5, "M", "B"))

# What is the accuracy of the ensemble prediction?
accuracy_ensemble <- confusionMatrix(factor(ensemble$y_hat), test_y)$overall["Accuracy"]

# Make a table of the accuracies of the 7 models and the accuracy of the ensemble model.

all_accuracies <- data.frame(kmeans = accuracy_kmeans,
                             glm = accuracy_glm,
                             lda = accuracy_lda,
                             qda = accuracy_qda,
                             loess = accuracy_loess,
                             knn = accuracy_knn,
                             rf = accuracy_rf,
                             ensemble = accuracy_ensemble
                             )

# Which of these models has the highest accuracy?
all_accuracies[which.max(all_accuracies)]