# Tasheena 

# load package
library(tidyverse)
library(caret)

# Package for easy timing in R
library(tictoc)


# Demo of timer function --------------------------------------------------
# Run the next 5 lines at once
tic()
Sys.sleep(3)
timer_info <- toc()
runtime <- timer_info$toc - timer_info$tic
runtime



# Get data ----------------------------------------------------------------
# Accelerometer Biometric Competition Kaggle competition data
# https://www.kaggle.com/c/accelerometer-biometric-competition/data

train <- read_csv("./Data/train.csv")

# YOOGE!
dim(train)



# knn modeling ------------------------------------------------------------
model_formula <- as.formula(Device ~ X + Y + Z)

# Values to use:

n_values <- c(200000, 400000, 600000, 800000, 1000000,
              1200000, 1400000, 1600000, 1800000, 2000000,
              2200000, 2400000, 2600000, 2800000, 3000000,
              3200000, 3400000, 3600000, 3800000, 4000000,
              4200000, 4400000, 4600000, 4800000, 5000000
)

k_values <- c(30,60,90,
              120,150,180,
              210,240,270,300
              )

n_val <- rep(0,length(n_values)*length(k_values)) 
k_val <- rep(0,length(n_values)*length(k_values))
time_val <- rep(0,length(n_values)*length(k_values))

count = 1
set.seed(495)

# Time knn here -----------------------------------------------------------
for (i in n_values){
  train_sample <- sample_n(train, i)
  for (j in k_values){
    tic()
    model_knn <- knn3(model_formula, train_sample, k = j)
    timer_info <- toc()
    time <- timer_info$toc - timer_info$tic
    
    n_val[count] <- i
    k_val[count] <- j
    time_val[count] <- time
    
    count <- count + 1
  }
}

# save v,k,time vectors in a data frame
df <- data.frame(n = n_val,k = k_val,runtime = time_val)

# Plot your results ---------------------------------------------------------
# Think of creative ways to improve this barebones plot. Note: you don't have to
# necessarily use geom_point
runtime_plot <- ggplot(df, aes(x=n, y=runtime, col=k, group=k)) +
  geom_line() +
  labs(x = "Training sample size",
       y = "Time taken in sec",
       title ="Analyzing runtime complexity for knn on different sample size and cluster size")

runtime_plot
ggsave(filename="Tasheena_Narraidoo.png", width=16, height = 9)




# Runtime complexity ------------------------------------------------------
# Can you write out the rough Big-O runtime algorithmic complexity as a function
# of:
# -n: number of points in training set
# -k: number of neighbors to consider
# -p: number of predictors used? In this case p is fixed at 3

# Answer:
# From the graph, we see that the run time is bounded below and above and there seems to be a 
# linear relationship. 
# Big-O runtime algorithmic complexity will be O(pnk) or as explained below, O(nk). 
# The knn function would calculate the distance between each point in the training
# set with k neighbors. And for each node, we have p predictors.
# Here, since we know the value of p (p fixed at 3), we get O(3nk) = O(nk).


# Reference:
# http://ggplot2.tidyverse.org/reference/aes_group_order.html
# https://github.com/topepo/caret/blob/master/pkg/caret/R/knn3.R

