library(fpp2)
library(tseries)

#Set working directory

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

##### ADF ###### test

adf.test(Test_EQ_ts)
adf.test(Test_MDP_ts)
adf.test(Test_MVAR_ts)



##### Box Junberg test €€€€
Box.test(Test_EQ_ts,
         lag = 10, type = "Ljung-Box")
