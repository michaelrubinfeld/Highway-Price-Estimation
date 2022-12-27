### IMPORTS ###
pacman::p_load(tidyverse,
               lubridate,
               dplyr,
               outliers,
               cluster,
               ClusterR,
               fastDummies,
               factoextra,
               rpart,
               ROCR,
               FNN,
               xgboost
               )


### Part 1 - The Product Analyst ###
## Q1. Load Data ##
df <- read.csv('price_data_20220604.csv')
colnames(df) <- c('datetime', 'price')
df$datetime <- as.POSIXct(df$datetime, format="%Y-%m-%dT%H:%M:%S")


## Q2. Arrange and clean ##
# check for missing values
any(is.na(df))  # no missing values found

# check for outliers
Q1 <- quantile(df$price, .25)
Q3 <- quantile(df$price, .75)
IQR <- IQR(df$price)
upper_bound <- Q3 + 1.5*IQR
lower_bound <- Q1 - 1.5*IQR
# lower bound = upper bound. We will check for outliers in an alternative way

# Grubbs' test
grubbs_test <- grubbs.test(df$price)
# according to Grubbs' test, we don't reject the null hypothesis (p_value=0.1033). 
# So, no outliers in the data.



## Q3+Q4. EDA + Addition of Temporal Features ##
# create basic temporal features
df['date'] <- date(df$datetime)
df['month'] <- month(df$datetime)
df['day'] <- day(df$datetime)
df['hour'] <- hour(df$datetime)
df['minute'] <- minute(df$datetime)


# first look at the data
ggplot(data=df, aes(x=datetime, y=price)) +
  geom_line(color = "darkgreen") +
  xlab('Temporal Data Point') + ylab('Price') + 
  ggtitle("Price Over Time") +
  theme(plot.title = element_text(hjust = 0.5))

# check daily average for seasonality
df_daily_average <- 
  aggregate(df$price, list(df$date), FUN=mean)
colnames(df_daily_average) <- c('date', 'mean_price')
ggplot(data=df_daily_average, aes(x=date, y=mean_price)) +
  geom_line(color = "lightcyan4", lwd=1.5) +
  geom_point(shape=21, color="black", fill="lightcyan", size=4) +
  xlab('Temporal Data Point') + ylab('Price') + 
  ggtitle("Average Price per day") +
  theme(plot.title = element_text(hjust = 0.5))


# check hourly average to see if there is an hourly seasonality
df_aggregated_by_hour <- aggregate(df$price, list(df$hour), FUN=mean)
colnames(df_aggregated_by_hour) <- c('hour', 'mean_price')

ggplot(data=df_aggregated_by_hour, aes(x=hour, y=mean_price)) +
  geom_line(color = "darkgray", lwd=1.5) +
  geom_point(shape=21, color="black", fill="lightgreen", size=4) +
  xlab('Hour') + ylab('Mean Price') + 
  ggtitle("Average Price Per Hour") +
  theme(plot.title = element_text(hjust = 0.5))

# CONCLUSION - RUSH HOUR WHICH BUMPS UP THE PRICE IS ROUGHLY AT 6 - 10
df['is_rush_hour'] <- ifelse(df$hour >= 6 & df$hour <= 10, 1, 0)


# check weekdays to see if there is a weekly seasonality
df['weekday'] <- weekdays(as.Date(df$date))

df_aggregated_by_weekday <- aggregate(df$price, list(df$weekday), FUN=mean)
colnames(df_aggregated_by_weekday) <- c('weekday', 'mean_price')
days_order <- c('Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday')

ggplot(df_aggregated_by_weekday, aes(x = factor(weekday, level=days_order), y = mean_price, fill=weekday)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = signif(mean_price, digits = 3)), nudge_y = 1) +
  xlab('Day of Week') + ylab('Mean Price') + 
  ggtitle("Day of Week Average") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = 'none')

# CONCLUSION - WE CAN SEE THAT THERE IS A DIFFERENCE IN THE WEEKEND
df['is_weekend'] <- ifelse(df$weekday %in% c('Friday', 'Saturday'), 1, 0) 

# turn days to dummy variables. We also remove the Sunday indicator because
# if we don't there will be a feature which is a linear combination of other
# features in the data.
df <- dummy_cols(df, select_columns = 'weekday')
df <- df[, ! names(df) %in% c("weekday", "weekday_Sunday")]


# temporal features from other sources #

# holidays 
holidays_df <- read.csv('fcal_Israel_2021.csv')
holidays_df <- sapply(holidays_df, substring, 0, 10)
holidays_df <- as.POSIXct(holidays_df, format="%d.%m.%Y")
holidays <- holidays_df[year(holidays_df) == 2022]

df['is_holiday'] <- ifelse(df$date %in% holidays, 1, 0) 


# temperature
temps_df <- read.csv('Tel_Aviv_temperatures.csv')
temps_df <- temps_df[c('datetime', 'temp')]
colnames(temps_df) <- c('date', 'temp')
temps_df$date <- as.POSIXct(temps_df$date, format="%Y-%m-%d")
temps_df['date'] <- date(temps_df$date)
df <- left_join(df, temps_df, by='date')

# turn datetime to index
df <- df %>% column_to_rownames(., var = 'datetime')





### Part 2 - The Economist ###
data_for_clustering <- df[, ! names(df) %in% c("date")]

# scale numeric features
numeric_features <- c(1:5, 15)
data_for_clustering[numeric_features] <- 
  as.data.frame(scale(data_for_clustering[numeric_features]))

# elbow method for optimal K
k.max <- 10
wss <- sapply(1:k.max, 
              function(k){
                kmeans(data_for_clustering, k, 
                       nstart=50, iter.max = 15)$tot.withinss
                }
              )

plot(1:k.max, wss,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")
title(main='Elbow Method For Optimal K')




### Part 3 - The Business Analyst ###
relevant_hours <- c(7, 8, 9, 10 ,11)
relevant_hours_df <- 
  df_aggregated_by_hour[df_aggregated_by_hour$hour %in% relevant_hours,]

salary_per_minute <- 100 / 60
relevant_hours_df['minimum_time_difference'] <- 
  relevant_hours_df['mean_price'] / salary_per_minute




### Part 4 - The Data Scientist ###
final_df <- df[, ! names(df) %in% c("date")]

# scale numeric features
numeric_features <- c(2:5, 15)
final_df[numeric_features] <- as.data.frame(scale(final_df[numeric_features]))

# split data
split_ratio <- 0.8
amount_train <- as.integer(nrow(df)*split_ratio)

X <- final_df[c(2:length(final_df))]
y <- final_df['price']

X_train <- X[c(1:amount_train-1),]
X_val <- X[c(amount_train:nrow(X)),] 

y_train <- y[c(1:amount_train-1),]
y_val <- y[c(amount_train:nrow(X)),] 

train <- final_df[c(1:amount_train-1),]
val <- final_df[c(amount_train:nrow(X)),]


# evaluate a model's MSE on data and label
get_mse <- function(model, val_data, y_gt) {
  y_hat <- predict(model, val_data)
  mse <- mean((y_hat - y_gt)**2)
  return(mse)
}



## basic baseline prediction - guess the mean ##
y_hat_baseline <- rep(mean(y_train), each=length(y_val))
basic_baseline_mse <- mean((y_hat_baseline - y_val)**2)

## advanced baseline prediction - for each sample guess its hourly average ##
train_hourly_averages <- aggregate(df$price, list(df$hour), FUN=mean)
colnames(train_hourly_averages) <- c('hour', 'mean_price')

val_for_baseline_pred <- val
val_for_baseline_pred <- merge(x=val_for_baseline_pred, y=train_hourly_averages, 
                               by="hour",all.x=FALSE, all.y=FALSE) 
baseline_mse <- 
  mean((val_for_baseline_pred$price - val_for_baseline_pred$mean_price)**2)

# OUR MSE BASELINE IS: 157.2602. This means that learning is feasible only if a 
# model can have a better MSE than our baseline MSE.


## Linear Regression ##
#model is linear, hence closed solution, hence no hyperparameters to tune.

liner_reg <- lm(price ~., data=train)
lin_reg_mse <- get_mse(liner_reg, X_val, y_val)
# our baseline MSE is better. This means that the function which takes us from X to y
# is probably not linear. the assumption of a linear connection between the 
# features and the label is so false, we're better off just guessing 
# the hourly average!


## KNN ##
knn_regressor <- knn.reg(train = X_train, y = y_train, k=100, test = X_val)
knn_reg_mse <- mean((knn_regressor$pred - y_val)**2)
Ks <- c(50, 60, 70, 80, 90, 100, 110, 120, 130, 140)
MSEs <- vector()
i = 1

while (i < length(Ks)) {
  curr_knn_regressor <- knn.reg(train = X_train, y = y_train, k=Ks[i], test = X_val)
  curr_knn_reg_mse <- mean((knn_regressor$pred - y_val)**2)
  MSEs[i] <- curr_knn_reg_mse
  i = i+1
}  




## Decision Tree ##
tree <- rpart(price ~ ., method = "anova", data = train)
png(file = "decTreeGFG.png", width = 600, 
    height = 600)
# Plot
plot(tree, uniform = TRUE,
     main = "Single Descision Tree")
text(tree, use.n = TRUE, cex = .7)
dev.off()

single_tree_mse <- get_mse(tree, X_val, y_val)



## XGBoost ##
# USING NON - SCALED DATA (because XGBoost is insensitive to 
# monotonic transformations) 
final_df_ns <- df[, ! names(df) %in% c("date")]

X <- final_df_ns[c(2:length(final_df_ns))]
y <- final_df_ns['price']

X_train <- X[c(1:amount_train-1),]
X_val <- X[c(amount_train:nrow(X)),] 

y_train <- y[c(1:amount_train-1),]
y_val <- y[c(amount_train:nrow(X)),] 

xgb_train = xgb.DMatrix(data = data.matrix(X_train), label = y_train)
xgb_val = xgb.DMatrix(data = data.matrix(X_val), label = y_val)


# hyperparameter tuning using self implemented pseudo-RandomSearch (without CV)
params <- expand.grid(min_child_weight = c(10, 13, 16, 19, 21, 24),
                      gamma = c(0.5, 1, 1.5, 2, 3, 5),
                      subsample = c(0.5, 0.8, 1.0),
                      colsample_bytree = c(0.3, 0.5, 0.6, 0.8, 1.0),
                      max_depth = c(2, 3, 4, 5, 6, 7, 8),
                      learning_rate = c(0.01, 0.05, 0.001, 0.0001),
                      eta = seq(.2, .35, .01)
                      ) 

curr_round = 1
while (curr_round < 10) {
  curr_min_child_weight <- sample(params$min_child_weight, 1) 
  curr_gamma <- sample(params$gamma, 1)
  curr_subsample <- sample(params$subsample, 1)
  curr_colsample_bytree <- sample(params$colsample_bytree, 1)
  curr_max_depth <- sample(params$max_depth, 1)
  curr_learning_rate <- sample(params$learning_rate, 1)
  curr_eta <- sample(params$eta, 1)
  
  
  xgb_untuned <- xgboost(
    data = xgb_train, 
    nrounds = 100, 
    verbose = 1,
    min_child_weight = curr_min_child_weight,
    gamma = curr_gamma,
    subsample = curr_subsample,
    colsample_bytree = curr_colsample_bytree,
    max_depth = curr_max_depth,
    learning_rate = curr_learning_rate,
    eta = curr_eta
  )
  curr_round = curr_round + 1
  print(curr_min_child_weight)
  print(curr_gamma)
  print(curr_subsample)
  print(curr_colsample_bytree)
  print(curr_max_depth)
  print(curr_learning_rate)
  print(curr_eta)
  print(get_mse(xgb_untuned, xgb_val, y_val))
  
}


# nround ~ epochs
# nrounds has a major impact on the model's performance (we decided to think of it
# as epochs). Epochs are not tuned if there is an "early stopping" mechanism. But,
# we only trained on the train set so we will plot the training graph over epochs
# to choose the optimal epochs that yields a good validation loss before the model
# learns the train data "too well" (risk of overfitting)
epochs <- seq(10, 220, 5)
train_MSEs <- vector()
val_MSEs <- vector()
i = 1

while (i <= length(epochs)) {
  tuned_xgb <- xgboost(
    data = xgb_train, 
    nrounds = epochs[i], 
    verbose = 1,
    min_child_weight = 16,
    gamma = 3,
    subsample = 0.8,
    colsample_bytree = 0.5,
    max_depth = 8,
    learning_rate = 0.05,
    eta = 0.27
  )
  train_MSEs[i] <- get_mse(tuned_xgb, xgb_train, y_train)
  val_MSEs[i] <- get_mse(tuned_xgb, xgb_val, y_val)
  i <- i+1
}

# plot model performance over epochs
ggplot() + 
  geom_line(aes(x = epochs, y = train_MSEs, color = "Train MSE")) + 
  geom_point(aes(x = epochs, y = train_MSEs),  fill='azure4', 
             shape=21, size=1.5) + 
  geom_line(aes(x = epochs, y = val_MSEs, color = "Validation MSE")) +
  geom_point(aes(x = epochs, y = val_MSEs), fill='navy', 
             shape=21, size=1.5) +
  xlab('Epochs (nrounds)') + ylab('MSE') + 
  ggtitle("Training Graph Over Epochs") +
  theme(plot.title = element_text(hjust = 0.5), legend.position = 'bottom',
        legend.title=element_blank())



### Final Prediction ###
test_df <- read.csv('EinBDW22_A3_predict_Ori_Twili_Michael_Rubinfeld.csv')
colnames(test_df) <- c('datetime', 'price')
test_df$datetime <- as.POSIXct(test_df$datetime, format="%Y-%m-%dT%H:%M:%S")

# preprocessing of the test set
test_df['date'] <- date(test_df$datetime)
test_df['month'] <- month(test_df$datetime)
test_df['day'] <- day(test_df$datetime)
test_df['hour'] <- hour(test_df$datetime)
test_df['minute'] <- minute(test_df$datetime)
test_df['is_rush_hour'] <- ifelse(test_df$hour >= 6 & test_df$hour <= 10, 1, 0)

# preprocessing of daily data
test_df['weekday'] <- weekdays(as.Date(test_df$date))
test_df['is_weekend'] <- ifelse(test_df$weekday %in% c('Friday', 'Saturday'), 1, 0) 
test_df <- dummy_cols(test_df, select_columns = 'weekday')
test_df <- test_df[, ! names(test_df) %in% c("weekday", "weekday_Sunday")]
test_df['is_holiday'] <- ifelse(test_df$date %in% holidays, 1, 0) 

# for temperature we will use data from last year!!
temps_df_test <- read.csv('Tel_Aviv_temperatures_2021-09-01_to_2021-09-14.csv')
temps_df_test <- temps_df_test[c('datetime', 'temp')]
colnames(temps_df_test) <- c('date', 'temp')
temps_df_test$date <- as.POSIXct(temps_df_test$date, format="%Y-%m-%d")
temps_df_test['date'] <- date(temps_df_test$date)
temps_df_test['date'] <- as.Date(temps_df_test$date) %m+% years(1)
test_df <- left_join(test_df, temps_df_test, by='date')

# final preparations for modeling
test_df <- test_df[, ! names(test_df) %in% c("date")]
test_df <- test_df %>% column_to_rownames(., var = 'datetime')
test_df <- test_df[, ! names(test_df) %in% c("price")]

# finally, after making sure the model isn't overfit we train the tuned model on all 
# of the data we have...
xgb_data = xgb.DMatrix(data = data.matrix(X), label = y$price)


# final predictor
final_xgb <- xgboost(
  data = xgb_data, 
  nrounds = 110, 
  verbose = 1,
  min_child_weight = 16,
  gamma = 3,
  subsample = 0.8,
  colsample_bytree = 0.5,
  max_depth = 8,
  learning_rate = 0.05,
  eta = 0.27
)


final_prediction <- predict(final_xgb, data.matrix(test_df))

# before submitting we're going to cap the lower bound to be 7, since we know for sure that the value
# can never be under 7, so why should our prediction be any different?
final_prediction <- replace(final_prediction, final_prediction < 7, 7)

# outputting the prediction
prediction_df <- read.csv('EinBDW22_A3_predict_Ori_Twili_Michael_Rubinfeld.csv')
prediction_df$mean_price <- final_prediction
write.csv(prediction_df, 
          'final_prediction/EinBDW22_A3_predict_Ori_Twili_Michael_Rubinfeld.csv',
          row.names=FALSE)









