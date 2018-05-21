import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
#from sklearn import linear_model as lm
import statsmodels.api as sm
from scipy.stats import t
import pickle
from prettytable import PrettyTable

class PolynomailTail:
    def __init__(self, initial_cap, *files):
        self.ic = initial_cap  # Initial Capital
        self.files = files
        self.train_list = []
        self.test_list = []
        self.train_date = []
        self.test_date = []
        self.price_list = []
        self.train_log_return = []
        self.test_log_return = []
        self.log_return = []
        self.train_price = []
        self.test_price = []
        self.V_t = []
        self.date_list = []
        self.end_date = []
        self.shares = []
        self.index = []
        self.lambda_dict = []
        self.mu = []
        self.cov_matrix = []
        self.weeklyReturns = []


    def get_price_list(self):
        # Get whole price list and date list
        date_list = []
        price_list = []
        for file in self.files:
            csv = pd.read_csv(file)
            csv.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
            list_0 = []
            list_1 = []
            for item in csv.as_matrix():
                list_0.append(item[0])
                list_1.append(item[1])
            date_list.append(list_0)
            price_list.append(list_1)
        self.price_list = price_list
        date_list = np.asarray(date_list)
        price_list = np.asarray(price_list)

        # Get train list and test list
        index = []
        for i in range(len(date_list)):
            for j in range(len(date_list[i])):
                date_list[i][j] = datetime.datetime.strptime(date_list[i][j], '%Y-%m-%d').strftime('%m/%d/%Y')
            index.append(np.nonzero(date_list[i] == '02/27/2018')[0][0])
        self.date_list = date_list
        self.train_date = [sub[:index[0]+1] for sub in date_list]
        self.test_date = [sub[index[0]+1:] for sub in date_list]
        train_price = [sub[:index[0]+1] for sub in price_list]
        test_price = [sub[index[0]+1:] for sub in price_list]
        self.train_price = train_price
        self.test_price = test_price

        for i in train_price:
            returns = []
            for j in range(len(i) - 1):
                returns.append((i[j + 1] - i[j]) / i[j])
            self.train_log_return.append(np.log(1 + np.asarray(returns)))
        for i in test_price:
            returns = []
            for j in range(len(i) - 1):
                returns.append((i[j + 1] - i[j]) / i[j])
            self.test_log_return.append(np.log(1 + np.asarray(returns)))
        for i in price_list:
            returns = []
            for j in range(len(i) - 1):
                returns.append((i[j + 1] - i[j]) / i[j])
            self.log_return.append(np.log(1 + np.asarray(returns)))
			
			
    def weeklyLosses(self, week_number):
        weekly_end_date = ['21-02-2017','24-02-2017','03-03-2017','10-03-2017','17-03-2017','24-03-2017','31-03-2017','07-04-2017', '13-04-2017','21-04-2017','28-04-2017','05-05-2017','12-05-2017','19-05-2017','26-05-2017','02-06-2017','09-06-2017','16-06-2017','23-06-2017','30-06-2017','07-07-2017','14-07-2017','21-07-2017','28-07-2017','04-08-2017','11-08-2017','18-08-2017','25-08-2017','01-09-2017','08-09-2017','15-09-2017','22-09-2017','29-09-2017','06-10-2017','13-10-2017','20-10-2017','27-10-2017','03-11-2017','10-11-2017','17-11-2017','24-11-2017','01-12-2017','08-12-2017','15-12-2017','22-12-2017','29-12-2017', '05-01-2018', '12-01-2018', '19-01-2018', '26-01-2018', '02-02-2018', '09-02-2018',
                           '16-02-2018', '23-02-2018', '02-03-2018', '09-03-2018', '16-03-2018', '23-03-2018', '29-03-2018', '06-04-2018', '13-04-2018', '20-04-2018'
                           ]

        length = len(weekly_end_date)
        for i in range(length):
            weekly_end_date[i] = datetime.datetime.strptime(weekly_end_date[i], '%d-%m-%Y').strftime('%m/%d/%Y')
            
        index = []
        for i in range(length):
            index.append(np.nonzero(self.date_list[0] == weekly_end_date[i])[0][0])
        self.index = index

        L_act = [-(self.V_t[self.index[i+1]] - self.V_t[self.index[i]]) for i in range(week_number-313,week_number)]
#        R_act = [-(self.V_t[self.index[i+1]]/self.V_t[self.index[i]]) for i in range(month_num)]  
        
#        
#        L_act = [-(self.V_t[self.index[i+1]] - self.V_t[self.index[i]]) for i in range(len(index) - 1)]
        return L_act
#        
#        returns_weekly = [-(self.V_t[self.index[i+1]]/self.V_t[self.index[i]]) for i in range(len(index)-1)]
#        self.weeklyReturns = returns_weekly
#
#        sorted_L_act = np.sort(L_act)
#        return sorted_L_act

        
    def estimate_a_regression_estimator(self, week_num):
        L_act = self.weeklyLosses(week_num)
        sorted_losses = np.sort(L_act)
        n = len(sorted_losses)        
        m = .2 * n
        m = int(m)
        k = n-m
        Y = []
        intercept = np.ones((m-1,1))
        X_values = []        
        for i in range(n-1,k,-1):
            Y.append(np.log(sorted_losses[i]))
            X_values.append(np.log((n-i)/n))
        
        X_values = np.asmatrix(X_values)
        X = np.concatenate((intercept, X_values.T), axis = 1) 
        Y = np.asmatrix(Y)
        betas = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T, Y.T))        
        a = -(1/betas[1][0])
        if(week_num == 106):
            plt.figure(1)
            X_values = np.asarray(X_values).reshape(-1)
            Y = np.asarray(Y).reshape(-1)
            plt.title("Week 106 Regression")
                        
            m_x = np.mean(X_values)
            m_y = np.mean(Y)
            n = np.size(X_values)
            SS_xy = np.sum(Y*X_values - n*m_x*m_y)            
            SS_xx = np.sum(Y*X_values - n*m_x*m_x)
            b_1 = SS_xy / SS_xx
            b_0 = m_y - b_1 * m_x
            y_pred = b_0 + b_1*X_values
            plt.scatter(X_values, Y, color = "m",marker = "o", s = 30)
            # plotting the regression line
            plt.figure(1)

            plt.plot(X_values, y_pred, color = "g")            
            plt.legend()
            plt.title('OLS predicted values')
            plt.ylabel("Log(L_k)")
            plt.xlabel("log(n-k/n)")
            plt.show()

        return a   
        
        
    def project2_VaR_ES_regression_estimator(self):
        a_parameter = []
        L_VaR_95 = []
        L_VaR_99 = []
        ES_95 = []
        ES_99 = []
        for i in range(106, 313):
            lossess = self.weeklyLosses(i)
            sorted_losses = np.sort(lossess)
            a = self.estimate_a_regression_estimator(i)
            a_parameter.append(a)
            alpha0 = 90
            alpha1_1 = 95
            alpha1_2 = 99
            VaR_alpha0 = np.percentile(sorted_losses,alpha0)
            power = (1/a)
            x = power[0][0]
            var_multiple_1 = np.power(((100-alpha0)/(100-alpha1_1)),x)
            var_multiple_2 = np.power(((100-alpha0)/(100-alpha1_2)),x)            
            L_95 = VaR_alpha0*var_multiple_1
            L_99 = VaR_alpha0*var_multiple_2
            L_VaR_95.append(L_95)
            L_VaR_99.append(L_99)                
            ES_multiple = (a/(a-1))
            ES_95.append(L_95*ES_multiple)
            ES_99.append(L_99*ES_multiple)
                
        a_parameter = np.asarray(a_parameter).reshape(-1) 
        plt.figure(3)        
        x= np.linspace(0,207,len(a_parameter))
        plt.title("a-Estimation via Regression Method")        
        plt.xlabel("Weeks")
        plt.ylabel("a-estimate")        
        plt.plot(x, a_parameter, color = "m",marker = "o")
        plt.show()
        
        L_VaR_95 = np.asarray(L_VaR_95).reshape(-1)
        L_VaR_99 = np.asarray(L_VaR_99).reshape(-1)
        ES_95 = np.asarray(ES_95).reshape(-1)
        ES_99 = np.asarray(ES_99).reshape(-1)
        
        plt.figure(2)        
        plt.plot(range(0, 207), L_VaR_95, label='VaR_.95 Loss')
        plt.legend()
        plt.plot(range(0, 207), L_VaR_99, label='VaR_.99 Loss')
        plt.legend()
        plt.plot(range(0, 207), ES_95, label='CVAR 95')
        plt.legend()
        plt.plot(range(0, 207), ES_99, label='CVAR 99')
        plt.legend()
        plt.show()

    def estimate_a_hill_estimator(self, week_num):
        L_act = self.weeklyLosses(week_num)
        sorted_losses = np.sort(L_act)
        length = len(sorted_losses)        
        c = []
        n_c = 0
#        c_value = np.percentile(sorted_losses, 94.9)
        for i in sorted_losses:
            if(i > 0.0):
                c.append(i)
            else:
                pass            
        n_c = len(c)
        c = np.sort(c)  
        a_estimate = []
        
        
        for i in range(n_c - 1):
            first_loss = c[i]
            a = 0
            for j in range((i+1),n_c):
                a += np.log(c[j]/first_loss)
            temp = n_c/a
            a_estimate.append(temp)
            
        if(week_num%4 == 0):
            plt.figure(week_num)
            x = np.linspace(0, n_c, n_c-1)
            plt.plot(x, a_estimate, color = "g")            
            plt.legend()
    #            plt.title('Week Hill Plot')
            plt.ylabel("a-estimate")
            plt.xlabel("n_c")
            plt.show()
        
        x = np.mean(a)    
        return x
            
    def project2_VaR_ES_hillEstimator(self):
        a_parameter = []
        L_VaR_95 = []
        L_VaR_99 = []
        ES_95 = []
        ES_99 = []        
        for i in range(106,313):
            lossess = self.weeklyLosses(i)
            sorted_losses = np.sort(lossess)
            a = self.estimate_a_hill_estimator(i)
#            if(i%4 == 0):
            a_parameter.append(a)
            alpha0 = 90
            alpha1_1 = 95
            alpha1_2 = 99
            VaR_alpha0 = np.percentile(sorted_losses,alpha0)
            power = (1/a)
            x = power
            var_multiple_1 = np.power(((100-alpha0)/(100-alpha1_1)),x)
            var_multiple_2 = np.power(((100-alpha0)/(100-alpha1_2)),x)            
            L_95 = VaR_alpha0*var_multiple_1            
            L_99 = VaR_alpha0*var_multiple_2
            L_VaR_95.append(L_95)
            L_VaR_99.append(L_99)     
            ES_multiple = (a/(a-1))
            ES_95.append(L_95*ES_multiple)
            ES_99.append(L_99*ES_multiple)
#       
        plt.figure(7)
        xaxis = np.linspace(0,207,207)
        plt.plot(xaxis,a_parameter)
        plt.show()
        
        L_VaR_95 = np.asarray(L_VaR_95).reshape(-1)
        L_VaR_99 = np.asarray(L_VaR_99).reshape(-1)
        ES_95 = np.asarray(ES_95).reshape(-1)
        ES_99 = np.asarray(ES_99).reshape(-1)
        
        plt.figure(5)        
        plt.plot(range(0, 207), L_VaR_95, label='VaR_.95 Loss')
        plt.legend()
        plt.plot(range(0, 207), L_VaR_99, label='VaR_.99 Loss')
        plt.legend()
        plt.plot(range(0, 207), ES_95, label='CVaR_95')
        plt.legend()
        plt.plot(range(0, 207), ES_99, label='CVaR_99')
        plt.legend()
        plt.show()

solver = PolynomailTail(450000, "AAL.csv", "AAPL.csv", "AMZN.csv", "C.csv",
                        "Hsbc.csv",  "Hmc.csv", "Googl.csv", "JPM.csv", "MS.csv",
                        "Msft.csv", "Ge.csv", "RY.csv", "KO.csv", "V.csv", "WFc.csv")
						
solver.estimate_a_regression_estimator()
solver.project2_VaR_ES_regression_estimator()
solver.estimate_a_hill_estimator()
solver.project2_VaR_ES_hillEstimator()