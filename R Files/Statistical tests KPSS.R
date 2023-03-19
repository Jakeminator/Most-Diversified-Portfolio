
#Load packages
library(readxl)
library(urca)
library(tseries)
library(fpp2)

#Copy into the console of Rstudio
#install.packages("readxl")
#install.packages("urca")
#install.packages("tseries")
#install.packages("fpp2")

#import the data
Test_EQ <- read_excel("iCloud Drive (arkiv) - 1/Documents/CBS/Master/3. Semester/Thesis/Code/R files/Test of EQ.xlsx")

Test_MDP <- read_excel("iCloud Drive (arkiv) - 1/Documents/CBS/Master/3. Semester/Thesis/Code/R files/Test of MDP.xlsx")

Test_MVAR <- read_excel("iCloud Drive (arkiv) - 1/Documents/CBS/Master/3. Semester/Thesis/Code/R files/Test of MVAR.xlsx")

#View the timeseries
View(Test_EQ)
View(Test_MDP)
View(Test_MVAR)

#Check the class of the timeseries (if it is a  "tbl" standing for table we have to convert it to a timeseries)
class(Test_EQ)
class(Test_MDP)
class(Test_MVAR)

#as the series is not a timeseries we need to convert it to a (ts)
Test_EQ_ts <- ts(Test_EQ)
Test_MDP_ts <- ts(Test_MDP)
Test_MVAR_ts <- ts(Test_MVAR)

#Check the classes
class(Test_EQ_ts)
class(Test_MDP_ts)
class(Test_MVAR_ts)

#################   ACF   ################

ggAcf(Test_EQ_ts)
ggAcf(Test_MDP_ts)
ggAcf(Test_MVAR_ts)

#only the spikes are considered to be significant under the critical threshhold.

################# PACF. ################

ggPacf(Test_EQ_ts)
ggPacf(Test_MDP_ts)
ggPacf(Test_MVAR_ts)

##################.  Augmented Dickey Fuller test.  #############

adf.test(Test_EQ_ts)
adf.test(Test_MDP_ts)
adf.test(Test_MVAR_ts)

#A large p-Value means we fail to reject the H0 hypothesis
#A Low p-value means we reject the H0 hypothesis indicating that there is stationarity

#In all cases we can conclude that there is stationarity in the variables

####################### Box Jungberg test ########################

Box.test(Test_EQ_ts, lag = 10, type = "Ljung-Box")
Box.test(Test_MDP_ts, lag = 10, type = "Ljung-Box")
Box.test(Test_MVAR_ts, lag = 10, type = "Ljung-Box")

#########################    KPSS Test     ############################

# Perform the KPSS test on the time series data
kpss.test.EQ <- ur.kpss(Test_EQ_ts, type = "tau", use.lag = NULL)
kpss.test.MDP <- ur.kpss(Test_MDP_ts, type = "tau", use.lag = NULL)
kpss.test.MVAR <- ur.kpss(Test_MVAR_ts, type = "tau", use.lag = NULL)

# Print the test results
summary(kpss.test.EQ)
summary(kpss.test.MDP)
summary(kpss.test.MVAR)

#The H0 is that there is stationarity in the dataset if the p-value is lower than 0,05 it suggests that the data is none stationary.

# Interpretations

#If the t_statistic is lower than the critical values it means that we fail to reject the H0. Meaning the data is stationary

#If the t_statistic is larger than the critical values it means we can reject the H0 and conclude that the data is non-stationary and have a unit root.

###### Creating data table #####

# Create an empty data frame to store the test results
results <- data.frame(matrix(ncol = 10, nrow = 0))

# Add the test results for each time series
results <- rbind(results, c("EQ", kpss.test.EQ@teststat, kpss.test.EQ@cval[1], kpss.test.EQ@cval[2], kpss.test.EQ@cval[3], kpss.test.EQ@cval[4], ifelse(kpss.test.EQ@teststat < kpss.test.EQ@cval[1], "Stationary", "Non-stationary"), ifelse(kpss.test.EQ@teststat < kpss.test.EQ@cval[2], "Stationary", "Non-stationary"), ifelse(kpss.test.EQ@teststat < kpss.test.EQ@cval[3], "Stationary", "Non-stationary"), ifelse(kpss.test.EQ@teststat < kpss.test.EQ@cval[4], "Stationary", "Non-stationary")))

results <- rbind(results, c("MDP", kpss.test.MDP@teststat, kpss.test.MDP@cval[1], kpss.test.MDP@cval[2], kpss.test.MDP@cval[3], kpss.test.MDP@cval[4], ifelse(kpss.test.MDP@teststat < kpss.test.MDP@cval[1], "Stationary", "Non-stationary"), ifelse(kpss.test.MDP@teststat < kpss.test.MDP@cval[2], "Stationary", "Non-stationary"), ifelse(kpss.test.MDP@teststat < kpss.test.MDP@cval[3], "Stationary", "Non-stationary"), ifelse(kpss.test.MDP@teststat < kpss.test.MDP@cval[4], "Stationary", "Non-stationary")))

results <- rbind(results, c("MVAR", kpss.test.MVAR@teststat, kpss.test.MVAR@cval[1], kpss.test.MVAR@cval[2], kpss.test.MVAR@cval[3], kpss.test.MVAR@cval[4], ifelse(kpss.test.MVAR@teststat < kpss.test.MVAR@cval[1], "Stationary", "Non-stationary"), ifelse(kpss.test.MVAR@teststat < kpss.test.MVAR@cval[2], "Stationary", "Non-stationary"), ifelse(kpss.test.MVAR@teststat < kpss.test.MVAR@cval[3], "Stationary", "Non-stationary"), ifelse(kpss.test.MVAR@teststat < kpss.test.MVAR@cval[4], "Stationary", "Non-stationary")))


colnames(results) <- c("Variable", "Test statistic", "Critical value (10%)","Critical value (5%)", "Critical value (2,5%)", "Critical value (1%)", "Stationarity at 10%", "Stationarity at 5%", "Stationarity at 2,5%", "Stationarity at 1%")

# Print the results
View(results)

#Assuming we still do not know if the data is stationary we can use a ndiffs to find out how much differencing we need to make the data stationary - however if it is already staionary it will tell us that zero number of diffs are needed and we do not need to apply differencing methods.

ndiffs(Test_EQ_ts) 
ndiffs(Test_MDP_ts)
ndiffs(Test_MVAR_ts)

diff_Test_MDP_ts <- diff(Test_MDP_ts)
  ggAcf(diff_Test_MDP_ts)
  
diff_Test_MVAR_ts <- diff(Test_MVAR_ts)
  ggAcf(diff_Test_MVAR_ts)