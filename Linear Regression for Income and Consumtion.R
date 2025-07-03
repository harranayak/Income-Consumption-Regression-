# Step 1 Click on Session Menu -> and from drop down 
# select set Working Directory->
# and select Choose Directory and finally select a folder where Income_Consumption.csv file is saved
# Step 2 type following command and ensure same folder is displayed on the screen
getwd()
# Step 3 Import Data from the working Directory and store in data memory variable
data <- read.csv("Income_Consumption.csv")
# Step 4 Display Data on the Screen
data  
# Step 5 Assumption 1 Plot the data into scatter plot for testing assumption of Linearity
plot(data$Income, data$Consumption, 
     main = "Scatter Plot in Base R", # Title
     xlab = "Income",           # Label for X-axis
     ylab = "Consumption",           # Label for Y-axis
     col = "red",                    # Point color
     pch = 19)                        # Point style (19 = solid circle)

# Step 6 Assumption 2 Plot the (Dependent variable) data for testing assumption of Normal Distribution
qqnorm(data$Consumption, main = "Q-Q Plot")
qqline(data$Consumption, col = "red")

# testing assumption of Normal Distribution through test
shapiro.test(data$Consumption)
ks.test(data$Consumption, "pnorm", mean = mean(data$Consumption, na.rm = TRUE), 
        sd = sd(data$Consumption, na.rm = TRUE))
install.packages("nortest")
library(nortest)
ad.test(data$Consumption)

hist(data$Consumption, breaks = 20, prob = TRUE, main = "Histogram with Normal Curve")
lines(density(data$Consumption), col = "blue")
curve(dnorm(x, mean = mean(data$Consumption), sd = sd(data$Consumption)), col = "red", add = TRUE)


# Fit the linear model
model <- lm(formula=Consumption ~ Income, data = data)

# Display the model summary
summary(model)

# Assumption  3 Scatter plot with regression line for linearity qualitatively
plot(data$Income, data$Consumption, main = "Scatterplot of x vs. y with Regression Line")
abline(model, col = "red", lwd = 2)


# Make predictions
predictions <- predict(model, data = data)
predictions

# Create a scatter plot with the regression line
install.packages("ggplot")
update.packages("ggplot")
library(ggplot2)

ggplot(data, aes(x = Income, y = Consumption)) +
  geom_point(color = "blue", size = 2) +
  geom_smooth(method = "lm", col = "red") +
  labs(title = "Income vs. Consumption",
       x = "Income",
       y = "Consumption") +  theme_minimal()

# Save the plot (optional)
ggsave("Income_vs_Consumption.png")

# Predicted values and residuals
data$Predicted <- predict(model, newdata = data)
data$Residuals <- residuals(model)

# Check the predicted values and residuals
head(data)

# Assumption 4 Residuals vs Fitted plot Homoscedasticity Assumption

plot(model$fitted.values, resid(model), 
     main = "Residuals vs Fitted",
     xlab = "Fitted values",
     ylab = "Residuals")
abline(h = 0, col = "red", lty = 2)

# Breusch-Pagan test for Homoscedasticity
library(lmtest)
bptest(model)

# Assumption 5 Histogram of residuals (Normal Distribution of residuals)
hist(resid(model), main = "Histogram of Residuals", xlab = "Residuals")

# Q-Q plot
qqnorm(resid(model))
qqline(resid(model), col = "red")

# Shapiro-Wilk test
shapiro.test(resid(model))

# Assumption  5 Durbin-Watson test The Durbin-Watson test can check for autocorrelation in residuals:
install.packages("lmtest")
library(lmtest)
dwtest(model)

# Save the updated data set with predicted values and residuals
write.csv(data, "Income_Consumption_with_Predictions.csv", row.names = FALSE)


# Step 4: Load ML libraries
install.packages("caret")
install.packages("randomForest")
install.packages("e1071")  # caret depends on this
library(caret)
library(randomForest)
library(e1071)

# Step 5: Data splitting into Training and Testing sets (80/20)
set.seed(123)  # for reproducibility
trainIndex <- createDataPartition(data$Consumption, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Step 6: Train a Linear Regression Model (ML way)
model_lm <- train(Consumption ~ Income, data = trainData, method = "lm")
summary(model_lm)

# Step 7: Train a Random Forest Model
model_rf <- train(Consumption ~ Income, data = trainData, method = "rf", importance = TRUE)
print(model_rf)

# Step 8: Make Predictions on Test Set
pred_lm <- predict(model_lm, newdata = testData)
pred_rf <- predict(model_rf, newdata = testData)

# Step 9: Model Evaluation (R-squared, RMSE, MAE)
postResample(pred = pred_lm, obs = testData$Consumption)
postResample(pred = pred_rf, obs = testData$Consumption)

# Step 10: Visualize Predictions vs Actual
library(ggplot2)

# Linear Regression
ggplot(testData, aes(x = Consumption, y = pred_lm)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, col = "red", linetype = "dashed") +
  labs(title = "Linear Regression: Actual vs Predicted",
       x = "Actual Consumption", y = "Predicted Consumption") +
  theme_minimal()

# Random Forest
ggplot(testData, aes(x = Consumption, y = pred_rf)) +
  geom_point(color = "darkgreen") +
  geom_abline(slope = 1, intercept = 0, col = "red", linetype = "dashed") +
  labs(title = "Random Forest: Actual vs Predicted",
       x = "Actual Consumption", y = "Predicted Consumption") +
  theme_minimal()

# Step 11: Save predictions (optional)
testData$Pred_LM <- pred_lm
testData$Pred_RF <- pred_rf
write.csv(testData, "Income_Consumption_Predictions_ML.csv", row.names = FALSE)
