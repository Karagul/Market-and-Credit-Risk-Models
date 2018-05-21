import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
#from sklearn import linear_model as lm
import statsmodels.api as sm
from scipy.stats import t
from prettytable import PrettyTable

class Portfolio_Risk:
    def __init__(self, capital, *files):
        self.ic = capital  # Initial Capital
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
        self.yield_list = []

    def get_yield(self,file):
        yield_list = []
        csv = pd.read_csv(file)
        list_1 = []
        for item in csv.as_matrix():
            list_1.append(item[1])
            yield_list.append(list_1)
        
        yield_list = np.asarray(yield_list)

    
    def get_price_list(self,instrument):
        # Get yield list
        date_list = []
        price_list = []
        yield_list = []
        for file in self.files:
            csv = pd.read_csv(file)
            list_0 = []
            list_1 = []
            list_2 = []
            if instrument == "stocks":
                csv.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1, inplace=True)
                for item in csv.as_matrix():
                    list_0.append(item[0])
                    list_1.append(item[1])
            elif instrument == "bonds":
                for item in csv.as_matrix():
                    list_0.append(item[0])
                    list_1.append(item[2])
                    list_2.append(item[1]*0.01)
            date_list.append(list_0)
            price_list.append(list_1)
            yield_list.append(list_2)
        self.price_list = price_list
        self.yield_list = yield_list
        date_list = np.asarray(date_list)
        price_list = np.asarray(price_list)
        yield_list = np.asarray(yield_list)
        # Get train list and test list
        index = []
        for i in range(len(date_list)):
            for j in range(len(date_list[i])):
                date_list[i][j] = datetime.datetime.strptime(date_list[i][j], '%Y-%m-%d').strftime('%m/%d/%Y')
        index.append(np.nonzero(date_list[i] == '02/26/2018')[0][0])
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


    ## Calculate protfolio value

    def cal_V_t(self):
        end_date = ['27-02-2017','03-03-2017','10-03-2017','17-03-2017','24-03-2017','31-03-2017','07-04-2017', '13-04-2017','21-04-2017','28-04-2017','05-05-2017','12-05-2017','19-05-2017','26-05-2017','02-06-2017','09-06-2017','16-06-2017','23-06-2017','30-06-2017','07-07-2017','14-07-2017','21-07-2017','28-07-2017','04-08-2017','11-08-2017','18-08-2017','25-08-2017','01-09-2017','08-09-2017','15-09-2017','22-09-2017','29-09-2017','06-10-2017','13-10-2017','20-10-2017','27-10-2017','03-11-2017','10-11-2017','17-11-2017','24-11-2017','01-12-2017','08-12-2017','15-12-2017','22-12-2017','29-12-2017', '05-01-2018', '12-01-2018', '19-01-2018', '26-01-2018', '02-02-2018', '09-02-2018',
                    '16-02-2018', '23-02-2018', '26-02-2018', '02-03-2018', '09-03-2018', '16-03-2018', '23-03-2018', '29-03-2018', '06-04-2018', '13-04-2018', '20-04-2018']
                    
        portfolio_end_date = ['26-02-2018', '02-03-2018', '09-03-2018', '16-03-2018', '23-03-2018', '29-03-2018', '06-04-2018', '13-04-2018', '20-04-2018']
                    
        length = len(end_date)
        for i in range(length):
            end_date[i] = datetime.datetime.strptime(end_date[i], '%d-%m-%Y').strftime('%m/%d/%Y')
            
        length = len(portfolio_end_date)
        for i in range(length):
            portfolio_end_date[i] = datetime.datetime.strptime(portfolio_end_date[i], '%d-%m-%Y').strftime('%m/%d/%Y')

        self.end_date = end_date
        # Find index of first 1 years
        index = []
        for i in range(len(end_date)):
            index.append(np.nonzero(self.date_list[0] == end_date[i])[0][0])
        self.index = index
        lambda_dict = {}
        # Get lambda list
        
        weight = 1 / 15  # N = 15
        for i in range(len(end_date)):
            asset_list = [item[index[i]] for item in self.price_list]
            if i == 0:
                lambda_dict[end_date[i]] = self.ic * weight / np.asarray(asset_list)
            else:
                lambda_dict[end_date[i]] = sum(lambda_dict[end_date[i - 1]] * np.asarray(asset_list)) * weight / np.asarray(asset_list)                
        
        for i in range(len(portfolio_end_date)):
            lambda_dict[portfolio_end_date[i]] = np.asarray([546,168,20, 387, 588, 818, 26, 254,527,316,2054, 371, 682,241 ,501]);
        
        self.lambda_dict = lambda_dict
        # Get portfolio value list
        V_t = [sum(np.asarray([item[0] for item in self.price_list]) * lambda_dict[end_date[0]])]
        print(V_t)
        for i in range(len(index) - 1):
            for j in range(index[i] + 1, index[i + 1] + 1):
                V_t.append(sum(np.asarray([item[j] for item in self.price_list]) * lambda_dict[end_date[i]]))
        self.V_t = V_t
        self.V_t[251] =450000


    def cal_V_t_bonds(self):
        end_date = ['27-02-2017','03-03-2017','10-03-2017','17-03-2017','24-03-2017','31-03-2017','07-04-2017', '13-04-2017','21-04-2017','28-04-2017','05-05-2017','12-05-2017','19-05-2017','26-05-2017','02-06-2017','09-06-2017','16-06-2017','23-06-2017','30-06-2017','07-07-2017','14-07-2017','21-07-2017','28-07-2017','04-08-2017','11-08-2017','18-08-2017','25-08-2017','01-09-2017','08-09-2017','15-09-2017','22-09-2017','29-09-2017','06-10-2017','13-10-2017','20-10-2017','27-10-2017','03-11-2017','10-11-2017','17-11-2017','24-11-2017','01-12-2017','08-12-2017','15-12-2017','22-12-2017','29-12-2017', '05-01-2018', '12-01-2018', '19-01-2018', '26-01-2018', '02-02-2018', '09-02-2018',
                    '16-02-2018', '23-02-2018', '26-02-2018', '02-03-2018', '09-03-2018', '16-03-2018', '23-03-2018', '29-03-2018', '06-04-2018', '13-04-2018', '20-04-2018']
                    
        portfolio_end_date = ['26-02-2018', '02-03-2018', '09-03-2018', '16-03-2018', '23-03-2018', '29-03-2018', '06-04-2018', '13-04-2018', '20-04-2018']
                    
        length = len(end_date)
        for i in range(length):
            end_date[i] = datetime.datetime.strptime(end_date[i], '%d-%m-%Y').strftime('%m/%d/%Y')
            
        length = len(portfolio_end_date)
        for i in range(length):
            portfolio_end_date[i] = datetime.datetime.strptime(portfolio_end_date[i], '%d-%m-%Y').strftime('%m/%d/%Y')

        self.end_date = end_date
        # Find index of first 1 years
        index = []
        for i in range(len(end_date)):
            index.append(np.nonzero(self.date_list[0] == end_date[i])[0][0])
        self.index = index
        lambda_dict = {}
        # Get lambda list
        
        weight = 1 / 2  # N = 15
        for i in range(len(end_date)):
            asset_list = [item[index[i]] for item in self.price_list]
            if i == 0:
                lambda_dict[end_date[i]] = self.ic * weight / np.asarray(asset_list)
            else:
                lambda_dict[end_date[i]] = sum(lambda_dict[end_date[i - 1]] * np.asarray(asset_list)) * weight / np.asarray(asset_list)                
        
        for i in range(len(portfolio_end_date)):
            lambda_dict[portfolio_end_date[i]] = np.asarray([4506.283, 3423.54]);
        self.lambda_dict = lambda_dict
        
        V_t = [sum(np.asarray([item[0] for item in self.price_list]) * lambda_dict[end_date[0]])]
        print(V_t)
        for i in range(len(index) - 1):
            for j in range(index[i] + 1, index[i + 1] + 1):
                V_t.append(sum(np.asarray([item[j] for item in self.price_list]) * lambda_dict[end_date[i]]))
        self.V_t = V_t
        self.V_t[251] =668000


    ## Plot empirical and fited PDF & CDF

    def calibrate(self, week_num): # 61 >= week_num >= 53
        mu = [np.mean(item) for item in np.asarray(self.log_return)[:, self.index[week_num-53]:self.index[week_num]]]
        self.mu = mu
        cov_matrix = np.cov(np.asarray(self.log_return)[:, self.index[week_num-53]:self.index[week_num]])
        self.cov_matrix = cov_matrix
        ## Fitted by normal
        X_normal = np.random.multivariate_normal(mu, cov_matrix, 1000)
        L = np.array(
            [-sum(
            self.lambda_dict[self.end_date[week_num - 1]]
            * np.asarray(self.price_list)[:, self.index[week_num]]
            * (np.exp(np.asarray(X_normal[i])) - 1)
            )
            for i in range(len(X_normal[:, 0]))
            ]
        )
        weight = 1 / 15
        L_delta = np.array([sum(-weight * self.V_t[week_num] * np.asarray(X_normal[i]))
                            for i in range(len(X_normal[:, 0]))]
                           )
        VaR = np.percentile(L_delta, 0.95);
        ## Fitted by t-student
        L_act = [-(self.V_t[self.index[i+1]] - self.V_t[self.index[i]]) for i in range(week_num-53, week_num)]
        parameters = t.fit(L_act)
        L_t = t.rvs(parameters[0], parameters[1], parameters[2], 1000)
        return [L, L_delta, L_t]

    ## Plot empirical and fited PDF & CDF

    def calibrate_bonds(self, week_num): # 60 >= week_num >= 53
        difference_0 = [self.yield_list[0][self.index[i + 1]] - self.yield_list[0][self.index[(i)]]
                        for i in range(week_num-53,week_num)]
        difference_1 = [
            self.yield_list[1][self.index[i + 1]] - self.yield_list[1][self.index[i]]
            for i in range(week_num-53, week_num)]

    
        mu = [np.mean(difference_0), np.mean(difference_1)]
        self.mu = mu
        temp = np.array([difference_0, difference_1])
 
        cov_matrix = np.cov(temp)
        self.cov_matrix = cov_matrix
        X_normal = np.random.multivariate_normal(mu, cov_matrix, 1000)
#        delta_list = [] 
#        for item in range(len(self.date_list[0])):
#            delta_list.append(1/(item+1))
        ## Fitted by normal
        
        t =  week_num
        lambda_pos = self.lambda_dict[self.end_date[week_num]]
        prices = np.asarray(self.price_list)[:, self.index[week_num]]
        yield_value = ((np.asarray(self.yield_list)[:, self.index[week_num]]))                       
        
        if week_num == 53:                 
            T_tau = np.array([(datetime.datetime.strptime('5/15/19', "%m/%d/%y") - datetime.datetime.strptime('2/26/18', "%m/%d/%y")).days / (50),
                                  (datetime.datetime.strptime('5/15/28', "%m/%d/%y") - datetime.datetime.strptime(
                                      '2/26/18', "%m/%d/%y")).days / (50)])
            L_delta = np.array(
                [-sum(
                lambda_pos *prices
                * (((yield_value/50) - X_normal[i] * T_tau))
                )
                for i in range(len(X_normal[:, 0]))
                ])
        else:
            T_tau = np.array([(datetime.datetime.strptime('5/15/19', "%m/%d/%y") - datetime.datetime.strptime('3/29/18', "%m/%d/%y")).days / (50),
                                  (datetime.datetime.strptime('5/15/28', "%m/%d/%y") - datetime.datetime.strptime(
                                      '03/29/18', "%m/%d/%y")).days / (50)])
            L_delta = np.array(
                [-sum(
                lambda_pos *prices
                * (((yield_value/50) - X_normal[i] * T_tau))
                )
                for i in range(len(X_normal[:, 0]))
                ])
                
        VaR = np.percentile(L_delta, 0.95)
        ## Fitted by t-student
        L_act = [-(self.V_t[self.index[i+1]] - self.V_t[self.index[i]]) for i in range(week_num-53, week_num)]
        L_act = np.array(L_act)
#        parameters = t.fit(L_act)
#        L_t = t.rvs(parameters[0], parameters[1], parameters[2], 1000)
        return [L_delta]
        
    ## Compare among normal L, normal L_delta and t-student L
    def compare(self):
        for i in range(54, 61):
            L, L_delta, L_t = self.calibrate(i)
            plt.figure(100 + i)
            # PDF
 #           plt.subplot(211)
            kde = stats.gaussian_kde(L)
            xaxis = np.linspace(L.min(), L.max(), 5000)
            plt.plot(xaxis, kde(xaxis), label="L normal")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()

            kde = stats.gaussian_kde(L_delta)
            xaxis_delta = np.linspace(L_delta.min(), L_delta.max(), 5000)
            plt.plot(xaxis_delta, kde(xaxis_delta), label="L-delta normal")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()

            kde = stats.gaussian_kde(L_t)
            xaxis_t = np.linspace(L_t.min(), L_t.max(), 5000)
            plt.plot(xaxis_t, kde(xaxis_t), label="L t-student")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()
            plt.grid(True)
            plt.show()
    ## Back test

    def back_test(self):
        L_act = [-(self.V_t[self.index[i+1]] - self.V_t[self.index[i]]) for i in range(53, 61)]
        # Plot a histogram of actual loss
        plt.hist(L_act, normed=True)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.grid()
        plt.title('Histogram of Loss')
        plt.xlabel("loss(unit=$1)")
        plt.show()
        # plot realized losses
        plt.plot(range(53, 61), L_act)
        plt.title("Loss time series")
        plt.ylabel("loss(unit=$1)")
        plt.xlabel("time(space = 1month)")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.show()
        # Table for showing actual monthly losses

        # rolling window
        for i in range(53, 61):
            L, L_delta, L_t = self.calibrate(i)
            ## Fitted by normal
            # Loss distribution
            plt.figure(100+i)
            # PDF
            plt.subplot(211)
            plt.title("L fitted by normal", y=1.08)
            plt.hist(L, normed=True)
            kde = stats.gaussian_kde(L)
            xaxis = np.linspace(L.min(), L.max(), 1000)
            plt.plot(xaxis, kde(xaxis), label="PDF")
            plt.axvline(x=L_act[54 - i], color="red")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()
            plt.grid(True)
            # CDF
            plt.subplot(212)
            p = np.arange(len(L)) / (len(L) - 1)
            plt.plot(sorted(L), p, label="CDF")
            plt.axvline(x=L_act[53 - i], color="red")
            plt.xlabel("loss(unit = $1)")
            plt.legend()
            plt.grid(True)
#            plt.savefig('48-' + str(i) + '-1' + '.png', bbox_inches='tight')
            plt.show()
            # Linearized Loss distribution
            plt.figure(200+i)
            # PDF
            plt.subplot(211)
            plt.title("Linearized L fitted by normal", y=1.08)
            plt.hist(L_delta, normed=True)
            kde = stats.gaussian_kde(L_delta)
            xaxis_delta = np.linspace(L_delta.min(), L_delta.max(), 1000)
            plt.plot(xaxis_delta, kde(xaxis_delta), label="PDF")
            plt.axvline(x=L_act[53 - i], color="red")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()
            plt.grid(True)
            # CDF
            p = np.arange(len(L_delta)) / (len(L_delta) - 1)
            plt.subplot(212)
            plt.plot(sorted(L_delta), p, label="CDF")
            plt.axvline(x=L_act[53 - i], color="red")
            plt.xlabel("loss(unit = $1)")
            plt.legend()
            plt.grid(True)
#            plt.savefig('48-' + str(i) + '-2' + '.png', bbox_inches='tight')
            plt.show()

            ## Fitted by t-distribution
            plt.figure(300+i)
            # PDF
            plt.subplot(211)
            plt.title("L fitted by student-t", y=1.08)
            plt.hist(L_t, normed=True)
            kde = stats.gaussian_kde(L_t)
            xaxis_t = np.linspace(L_t.min(), L_t.max(), 1000)
            plt.plot(xaxis_t, kde(xaxis_t), label="PDF")
            plt.axvline(x=L_act[53 - i], color="red")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()
            plt.grid(True)
            # CDF
            p = np.arange(len(L_t)) / (len(L_t) - 1)
            plt.subplot(212)
            plt.plot(sorted(L_t), p, label="CDF")
            plt.axvline(x=L_act[53 - i], color="red")
            plt.xlabel("loss(unit = $1)")
            plt.legend()
            plt.grid(True)
#            plt.savefig('48-' + str(i) + '-3' + '.png', bbox_inches='tight')
            plt.show()
        a = 1

    ## Risk

    def risk(self):
        # Loss L
        L_VaR_95 = []
        L_CVaR_95 = []
        # Linearized loss L_delta
        L_delta_VaR_95 = []
        L_delta_CVaR_95 = []
        # T-studen loss L_t
        L_t_VaR_95 = []
        L_t_CVaR_95 = []
        for i in range(53, 61):
            print("It's time " + str(i))
            L, L_delta, L_t = self.calibrate(i)
            # Normal case
            # For Loss L
            L_fit_mu, L_fit_sigma = norm.fit(L)
            # Calculate VaR and CVaR
            L_VaR_95.append(norm.ppf(0.95, L_fit_mu, L_fit_sigma))
            L_CVaR_95.append(L_fit_mu + L_fit_sigma * (norm.pdf(norm.ppf(0.95)) / (1 - 0.95)))

            # For linearized Loss L_delta
            L_delta_fit_mu, L_delta_fit_sigma = norm.fit(L_delta)
            # Calculate VaR and CVaR
            L_delta_VaR_95.append(norm.ppf(0.95, L_delta_fit_mu, L_delta_fit_sigma))
            L_delta_CVaR_95.append(L_delta_fit_mu + L_delta_fit_sigma * (norm.pdf(norm.ppf(0.95)) / (1 - 0.95)))

            # T-student case
            # For Loss L
            while True:
                L_t_fit_df, L_t_fit_mu, L_t_fit_sigma = t.fit(L_t)
                print(L_t_fit_df)
                if L_t_fit_df > 1:
                    break
                L, L_delta, L_t = self.calibrate(i)
            # Calculate VaR and CVaR
            L_t_VaR_95.append(t.ppf(0.95, L_t_fit_df, L_t_fit_mu, L_t_fit_sigma))
            L_t_CVaR_95.append(L_fit_mu + L_fit_sigma * ((t.pdf(t.ppf(0.95, L_t_fit_df), L_t_fit_df) / (1 - 0.95))) * ( ((L_t_fit_df + (t.ppf(0.95, L_t_fit_df))**2) / (L_t_fit_df - 1) )) )
        # Plot
        # For 0.95 case
        plt.plot(range(53, 61), L_VaR_95, label='VaR_.95 Loss')
        plt.legend()        
        plt.plot(range(53, 61), L_CVaR_95, label='CVaR_.95 Loss')
        plt.legend()
        plt.plot(range(53, 61), L_delta_VaR_95, label='VaR_.95 Linearized Loss')
        plt.legend()
        plt.plot(range(53, 61), L_delta_CVaR_95, label='CVaR_.95 Linearized Loss')
        plt.legend()
        plt.plot(range(53, 61), L_t_VaR_95, label='VaR_.95 Loss-t')
        plt.legend()
        plt.plot(range(53, 61), L_t_CVaR_95, label='CVaR_.95 Loss-t')
        plt.legend()
        plt.show()
        # Save these VaR and CVaR as .txt
        with open("test.txt", "wb") as fp:  # Pickling
            pickle.dump([[L_VaR_95, L_CVaR_95, L_delta_VaR_95, L_delta_CVaR_95, L_t_VaR_95, L_t_CVaR_95]
                         ], fp)

        ## Difference between actual monthly losses and VaR & CVaR values
        with open("test.txt", "rb") as fp:  # Unpickling
            VaR_CVaR_list = pickle.load(fp)
        xaxis = range(53, 61)
        L_act = np.asarray([-(solver.V_t[i + 1] - solver.V_t[i]) for i in range(53, 61)])
        plt.subplot(311)
        plt.plot(range(53, 61), (np.asarray(VaR_CVaR_list[0][0]) - L_act) / np.asarray(VaR_CVaR_list[0][0]),
                 label='VaR_.95 Loss')
        plt.legend()
        plt.subplot(312)
        plt.plot(range(53, 61), (np.asarray(VaR_CVaR_list[0][2]) - L_act) / np.asarray(VaR_CVaR_list[0][2]),
                 label='VaR_.95 Linearized Loss')
        plt.legend()
        plt.subplot(313)
        plt.plot(range(53, 61), (np.asarray(VaR_CVaR_list[0][4]) - L_act) / np.asarray(VaR_CVaR_list[0][4]),
                 label='VaR_.95 Loss-t')
        plt.legend()
        plt.xlabel("time(space = 1month)")
        plt.ylabel('percentage')
        plt.show()
        plt.subplot(311)
        plt.plot(range(53, 61), (np.asarray(VaR_CVaR_list[0][1]) - L_act) / np.asarray(VaR_CVaR_list[0][1]),
                 label='CVaR_.95 Loss')
        plt.legend()
        plt.subplot(312)
        plt.plot(range(53, 61), (np.asarray(VaR_CVaR_list[0][3]) - L_act) / np.asarray(VaR_CVaR_list[0][3]),
                 label='CVaR_.95 Linearized Loss')
        plt.legend()
        plt.subplot(313)
        plt.plot(range(53, 61), (np.asarray(VaR_CVaR_list[0][5]) - L_act) / np.asarray(VaR_CVaR_list[0][5]),
                 label='CVaR_.95 Loss-t')
        plt.legend()
        plt.xlabel("time(space = 1month)")
        plt.ylabel('percentage')
        plt.show()

    def back_test(self):
        L_act = [-(self.V_t[self.index[i+1]] - self.V_t[self.index[i]]) for i in range(53, 61)]
        # Plot a histogram of actual loss
        plt.hist(L_act, normed=True)
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        plt.grid()
        plt.title('Histogram of Loss')
        plt.xlabel("loss(unit=$1)")
        plt.show()
        # plot realized losses
        plt.plot(range(53, 61), L_act)
        plt.title("Loss time series")
        plt.ylabel("loss(unit=$1)")
        plt.xlabel("time(space = 1month)")
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.show()
        # Table for showing actual monthly losses

        # rolling window
        for i in range(53, 61):
            L, L_delta, L_t = self.calibrate(i)
            ## Fitted by normal
            # Loss distribution
            plt.figure(100+i)
            # PDF
            plt.subplot(211)
            plt.title("L fitted by normal", y=1.08)
            plt.hist(L, normed=True)
            kde = stats.gaussian_kde(L)
            xaxis = np.linspace(L.min(), L.max(), 1000)
            plt.plot(xaxis, kde(xaxis), label="PDF")
            plt.axvline(x=L_act[54 - i], color="red")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()
            plt.grid(True)
            # CDF
            plt.subplot(212)
            p = np.arange(len(L)) / (len(L) - 1)
            plt.plot(sorted(L), p, label="CDF")
            plt.axvline(x=L_act[53 - i], color="red")
            plt.xlabel("loss(unit = $1)")
            plt.legend()
            plt.grid(True)
#            plt.savefig('48-' + str(i) + '-1' + '.png', bbox_inches='tight')
            plt.show()
            # Linearized Loss distribution
            plt.figure(200+i)
            # PDF
            plt.subplot(211)
            plt.title("Linearized L fitted by normal", y=1.08)
            plt.hist(L_delta, normed=True)
            kde = stats.gaussian_kde(L_delta)
            xaxis_delta = np.linspace(L_delta.min(), L_delta.max(), 1000)
            plt.plot(xaxis_delta, kde(xaxis_delta), label="PDF")
            plt.axvline(x=L_act[53 - i], color="red")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()
            plt.grid(True)
            # CDF
            p = np.arange(len(L_delta)) / (len(L_delta) - 1)
            plt.subplot(212)
            plt.plot(sorted(L_delta), p, label="CDF")
            plt.axvline(x=L_act[53 - i], color="red")
            plt.xlabel("loss(unit = $1)")
            plt.legend()
            plt.grid(True)
#            plt.savefig('48-' + str(i) + '-2' + '.png', bbox_inches='tight')
            plt.show()

            ## Fitted by t-distribution
            plt.figure(300+i)
            # PDF
            plt.subplot(211)
            plt.title("L fitted by student-t", y=1.08)
            plt.hist(L_t, normed=True)
            kde = stats.gaussian_kde(L_t)
            xaxis_t = np.linspace(L_t.min(), L_t.max(), 1000)
            plt.plot(xaxis_t, kde(xaxis_t), label="PDF")
            plt.axvline(x=L_act[53 - i], color="red")
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            plt.legend()
            plt.grid(True)
            # CDF
            p = np.arange(len(L_t)) / (len(L_t) - 1)
            plt.subplot(212)
            plt.plot(sorted(L_t), p, label="CDF")
            plt.axvline(x=L_act[53 - i], color="red")
            plt.xlabel("loss(unit = $1)")
            plt.legend()
            plt.grid(True)
#            plt.savefig('48-' + str(i) + '-3' + '.png', bbox_inches='tight')
            plt.show()
        a = 1

    ## Risk

    def risk_bonds(self):
    
        # Linearized loss L_delta
        L_delta_VaR_95 = []
        L_delta_CVaR_95 = []
        # T-studen loss L_t
        L_t_VaR_95 = []
        L_t_CVaR_95 = []
        for i in range(52, 60):
            print("It's time " + str(i))
            L_delta, L_t = self.calibrate(i)
            
            # For linearized Loss L_delta
            L_delta_fit_mu, L_delta_fit_sigma = norm.fit(L_delta)
            # Calculate VaR and CVaR
            L_delta_VaR_95.append(norm.ppf(0.95, L_delta_fit_mu, L_delta_fit_sigma))
            L_delta_CVaR_95.append(L_delta_fit_mu + L_delta_fit_sigma * (norm.pdf(norm.ppf(0.95)) / (1 - 0.95)))

            # T-student case
            # For Loss L
            while True:
                L_t_fit_df, L_t_fit_mu, L_t_fit_sigma = t.fit(L_t)
                print(L_t_fit_df)
                if L_t_fit_df > 1:
                    break
                L_delta, L_t = self.calibrate(i)
            # Calculate VaR and CVaR
            L_t_VaR_95.append(t.ppf(0.95, L_t_fit_df, L_t_fit_mu, L_t_fit_sigma))
            L_t_CVaR_95.append(L_t_fit_mu + L_t_fit_sigma * ((t.pdf(t.ppf(0.95, L_t_fit_df), L_t_fit_df) / (1 - 0.95))) * ( ((L_t_fit_df + (t.ppf(0.95, L_t_fit_df))**2) / (L_t_fit_df - 1) )) )
        # Plot
        # For 0.95 case
        plt.plot(range(52, 60), L_delta_VaR_95, label='VaR_.95 Linearized Loss')
        plt.legend()
        plt.plot(range(52, 60), L_delta_CVaR_95, label='CVaR_.95 Linearized Loss')
        plt.legend()
        plt.plot(range(52, 60), L_t_VaR_95, label='VaR_.95 Loss-t')
        plt.legend()
        plt.plot(range(52, 60), L_t_CVaR_95, label='CVaR_.95 Loss-t')
        plt.legend()
        plt.show()
        # Save these VaR and CVaR as .txt
        with open("test.txt", "wb") as fp:  # Pickling
            pickle.dump([[L_delta_VaR_95, L_delta_CVaR_95, L_t_VaR_95, L_t_CVaR_95]
                         ], fp)

        ## Difference between actual monthly losses and VaR & CVaR values
        with open("test.txt", "rb") as fp:  # Unpickling
            VaR_CVaR_list = pickle.load(fp)
        xaxis = range(53, 61)
        L_act = np.asarray([-(solver.V_t[i + 1] - solver.V_t[i]) for i in range(53, 61)])
        plt.subplot(211)
        plt.plot(range(52, 60), (np.asarray(VaR_CVaR_list[0][0]) - L_act) / np.asarray(VaR_CVaR_list[0][0]),
                 label='VaR_.95 Linearized Loss')
        plt.legend()
        plt.subplot(212)
        plt.plot(range(52, 60), (np.asarray(VaR_CVaR_list[0][2]) - L_act) / np.asarray(VaR_CVaR_list[0][2]),
                 label='VaR_.95 Loss-t')
        plt.legend()
        plt.xlabel("time(space = 1month)")
        plt.ylabel('percentage')
        plt.show()
        
        plt.subplot(211)
        plt.plot(range(52, 60), (np.asarray(VaR_CVaR_list[0][1]) - L_act) / np.asarray(VaR_CVaR_list[0][1]),
                 label='CVaR_.95 Linearized Loss')
        plt.legend()
        plt.subplot(212)
        plt.plot(range(52, 60), (np.asarray(VaR_CVaR_list[0][3]) - L_act) / np.asarray(VaR_CVaR_list[0][3]),
                 label='CVaR_.95 Loss-t')
        plt.legend()
        plt.xlabel("time(space = 1month)")
        plt.ylabel('percentage')
        plt.show()


        def calibrate_options(self):
        L_delta = []
        L_quadratic = []
        L_t = []
        for i in [0, 2]:
            # Linearized normal
            if i == 0:
                option_ir_index = []
                for date in self.option_weekly_date[i]:
                    index = np.nonzero(self.option_ir_date[0] == date)[0][0]
                    option_ir_index.append(index)
                option_ir_index = np.asarray(option_ir_index)
                mu = np.array([np.mean(self.log_return[7][self.weekly_index[week_num:(113 + week_num)]]),
                               np.mean(np.asarray([self.option_ir[0][option_ir_index[item + 1]]-self.option_ir[0][option_ir_index[item]]
                                       for item in range(week_num, 23 + week_num - 1)])),
                               np.mean(
                                   np.asarray([self.option_iv[i][self.option_weekly_index[i][item + 1]] - self.option_iv[i][self.option_weekly_index[i][item]]
                                   for item in range(week_num, 23 + week_num - 1)]))
                               ])
                std = np.array([np.std(self.log_return[7][self.weekly_index[week_num:(113 + week_num)]]),
                               np.std(np.asarray([self.option_ir[0][option_ir_index[item + 1]]-self.option_ir[0][option_ir_index[item]]
                                       for item in range(week_num, 23 + week_num - 1)])),
                               np.std(
                                   np.asarray([self.option_iv[i][self.option_weekly_index[i][item + 1]] - self.option_iv[i][self.option_weekly_index[i][item]]
                                   for item in range(week_num, 23 + week_num - 1)]))
                               ])
            elif i == 1:
                option_ir_index = []
                for date in self.option_weekly_date[i]:
                    index = np.nonzero(self.option_ir_date[0] == date)[0][0]
                    option_ir_index.append(index)
                option_ir_index = np.asarray(option_ir_index)
                mu = np.array([np.mean(self.log_return[9][self.weekly_index[week_num:(113 + week_num)]]),
                               np.mean(np.asarray([self.option_ir[0][option_ir_index[item + 1]] - self.option_ir[0][
                                   option_ir_index[item]]
                                                   for item in range(week_num, 2 + week_num - 1)])),
                               np.mean(
                                   np.asarray([self.option_iv[i][self.option_weekly_index[i][item + 1]] -
                                               self.option_iv[i][self.option_weekly_index[i][item]]
                                               for item in range(week_num, 2 + week_num - 1)]))
                               ])
                std = np.array([np.std(self.log_return[9][self.weekly_index[week_num:(113 + week_num)]]),
                                np.std(np.asarray([self.option_ir[0][option_ir_index[item + 1]] - self.option_ir[0][
                                    option_ir_index[item]]
                                                   for item in range(week_num, 2 + week_num - 1)])),
                                np.std(
                                    np.asarray([self.option_iv[i][self.option_weekly_index[i][item + 1]] -
                                                self.option_iv[i][self.option_weekly_index[i][item]]
                                                for item in range(week_num, 2 + week_num - 1)]))
                                ])
            elif i == 2:
                option_ir_index = []
                for date in self.option_weekly_date[i]:
                    index = np.nonzero(self.option_ir_date[0] == date)[0][0]
                    option_ir_index.append(index)
                option_ir_index = np.asarray(option_ir_index)
                mu = np.array([np.mean(self.XEO_log_return[self.weekly_index[week_num:(113 + week_num)]]),
                               np.mean(np.asarray([self.option_ir[0][option_ir_index[item + 1]] - self.option_ir[0][
                                   option_ir_index[item]]
                                                   for item in range(week_num, 5 + week_num - 1)])),
                               np.mean(
                                   np.asarray([self.option_iv[i][self.option_weekly_index[i][item + 1]] -
                                               self.option_iv[i][self.option_weekly_index[i][item]]
                                               for item in range(week_num, 5 + week_num - 1)]))
                               ])
                std = np.array([np.std(self.XEO_log_return[self.weekly_index[week_num:(113 + week_num)]]),
                                np.std(np.asarray([self.option_ir[0][option_ir_index[item + 1]] - self.option_ir[0][
                                    option_ir_index[item]]
                                                   for item in range(week_num, 5 + week_num - 1)])),
                                np.std(
                                    np.asarray([self.option_iv[i][self.option_weekly_index[i][item + 1]] -
                                                self.option_iv[i][self.option_weekly_index[i][item]]
                                                for item in range(week_num, 5 + week_num - 1)]))
                                ])
            X_normal = np.array([np.random.normal(mu[0], std[0], 10000), np.random.normal(mu[1], std[1], 10000),
                                 np.random.normal(mu[2], std[2], 10000)])
            L_first = []
            L_second = []
            greeks = self.cal_option_greeks(week_num, i)
            for j in range(len(X_normal[0])):
                if week_num == 0:
                    index = np.nonzero(self.date_list[0] == '2/28/18')[0][0]
                    if i == 0:
                        # L_first.append(-(self.option_greeks[i][week_num][0]
                        #            + self.option_greeks[i][week_num][1] * X_normal[0][j] * self.price_list[7][index]
                        #            + self.option_greeks[i][week_num][2] * X_normal[1][j]
                        #            + self.option_greeks[i][week_num][3] * X_normal[2][j]))
                        L_first.append(-(greeks[0]/50
                                         + greeks[1] * X_normal[0][j] * self.price_list[7][index]
                                         + greeks[2] * X_normal[1][j]
                                         + greeks[3] * X_normal[2][j]))
                        L_second.append(
                            -(
                                (greeks[0]/50
                                 + greeks[1] * X_normal[0][j] * self.price_list[7][index]
                                 + greeks[2] * X_normal[1][j]
                                 + greeks[3] * X_normal[2][j])
                                +
                                0.5 * (
                                greeks[4] * (X_normal[0][j]**2) * (self.price_list[7][index]**2) +
                                2 * greeks[5] * self.price_list[7][index] * X_normal[0][j] * X_normal[2][j] +
                                greeks[6] * (X_normal[2][j]**2)
                                )
                            )
                        )
                    elif i == 1:
                        # L_first.append(-(self.option_greeks[i][week_num][0]
                        #                  + self.option_greeks[i][week_num][1] * X_normal[0][j] * self.price_list[9][
                        #                      index]
                        #                  + self.option_greeks[i][week_num][2] * X_normal[1][j]
                        #                  + self.option_greeks[i][week_num][3] * X_normal[2][j]))
                        L_first.append(-(greeks[0]/50
                                   + greeks[1] * X_normal[0][j] * self.price_list[9][index]
                                   + greeks[2] * X_normal[1][j]
                                   + greeks[3] * X_normal[2][j]))
                        L_second.append(
                                -(
                                        (greeks[0]/50
                                         + greeks[1] * X_normal[0][j] * self.price_list[9][index]
                                         + greeks[2] * X_normal[1][j]
                                         + greeks[3] * X_normal[2][j])
                                        +
                                        0.5 * (
                                                greeks[4] * (X_normal[0][j] ** 2) * (self.price_list[9][index] ** 2) +
                                                2 * greeks[5] * self.price_list[9][index] * X_normal[0][j] *
                                                X_normal[2][j] +
                                                greeks[6] * (X_normal[2][j] ** 2)
                                        )
                                )
                        )
                    elif i == 2:
                        # L_first.append(-(self.option_greeks[i][week_num][0]
                        #                  + self.option_greeks[i][week_num][1] * X_normal[0][j] * self.XEO_price[index]
                        #                  + self.option_greeks[i][week_num][2] * X_normal[1][j]
                        #                  + self.option_greeks[i][week_num][3] * X_normal[2][j]))
                        L_first.append(-(greeks[0]/50
                                   + greeks[1] * X_normal[0][j] * self.XEO_price[index]
                                   + greeks[2] * X_normal[1][j]
                                   + greeks[3] * X_normal[2][j]))
                        L_second.append(
                                -(
                                        (greeks[0]/50
                                         + greeks[1] * X_normal[0][j] * self.XEO_price[index]
                                         + greeks[2] * X_normal[1][j]
                                         + greeks[3] * X_normal[2][j])
                                        +
                                        0.5 * (
                                                greeks[4] * (X_normal[0][j] ** 2) * (self.XEO_price[index] ** 2) +
                                                2 * greeks[5] * self.XEO_price[index] * X_normal[0][j] *
                                                X_normal[2][j] +
                                                greeks[6] * (X_normal[2][j] ** 2)
                                        )
                                )
                        )
                else:
                    index = self.weekly_index[112 + week_num]
                    if i == 0:
                        # L_first.append(-(self.option_greeks[i][week_num][0]
                        #                  + self.option_greeks[i][week_num][1] * X_normal[0][j] * self.price_list[7][
                        #                      index]
                        #                  + self.option_greeks[i][week_num][2] * X_normal[1][j]
                        #                  + self.option_greeks[i][week_num][3] * X_normal[2][j]))
                        L_first.append(-(greeks[0]/50
                                   + greeks[1] * X_normal[0][j] * self.price_list[7][index]
                                   + greeks[2] * X_normal[1][j]
                                   + greeks[3] * X_normal[2][j]))
                        L_second.append(
                            -(
                                (greeks[0]/50
                                 + greeks[1] * X_normal[0][j] * self.price_list[7][index]
                                 + greeks[2] * X_normal[1][j]
                                 + greeks[3] * X_normal[2][j])
                                +
                                0.5 * (
                                greeks[4] * (X_normal[0][j]**2) * (self.price_list[7][index]**2) +
                                2 * greeks[5] * self.price_list[7][index] * X_normal[0][j] * X_normal[2][j] +
                                greeks[6] * (X_normal[2][j]**2)
                                )
                            )
                        )
                    elif i == 1:
                        # L_first.append(-(self.option_greeks[i][week_num][0]
                        #                  + self.option_greeks[i][week_num][1] * X_normal[0][j] * self.price_list[9][
                        #                      index]
                        #                  + self.option_greeks[i][week_num][2] * X_normal[1][j]
                        #                  + self.option_greeks[i][week_num][3] * X_normal[2][j]))
                        L_first.append(-(greeks[0]/50
                                   + greeks[1] * X_normal[0][j] * self.price_list[9][index]
                                   + greeks[2] * X_normal[1][j]
                                   + greeks[3] * X_normal[2][j]))
                        L_second.append(
                                -(
                                        (greeks[0]/50
                                         + greeks[1] * X_normal[0][j] * self.price_list[9][index]
                                         + greeks[2] * X_normal[1][j]
                                         + greeks[3] * X_normal[2][j])
                                        +
                                        0.5 * (
                                                greeks[4] * (X_normal[0][j] ** 2) * (self.price_list[9][index] ** 2) +
                                                2 * greeks[5] * self.price_list[9][index] * X_normal[0][j] *
                                                X_normal[2][j] +
                                                greeks[6] * (X_normal[2][j] ** 2)
                                        )
                                )
                        )
                    elif i == 2:
                        # L_first.append(-(self.option_greeks[i][week_num][0]
                        #                  + self.option_greeks[i][week_num][1] * X_normal[0][j] * self.XEO_price[index]
                        #                  + self.option_greeks[i][week_num][2] * X_normal[1][j]
                        #                  + self.option_greeks[i][week_num][3] * X_normal[2][j]))
                        L_first.append(-(greeks[0]/50
                                   + greeks[1] * X_normal[0][j] * self.XEO_price[index]
                                   + greeks[2] * X_normal[1][j]
                                   + greeks[3] * X_normal[2][j]))
                        L_second.append(
                                -(
                                        (greeks[0]/50
                                         + greeks[1] * X_normal[0][j] * self.XEO_price[index]
                                         + greeks[2] * X_normal[1][j]
                                         + greeks[3] * X_normal[2][j])
                                        +
                                        0.5 * (
                                                greeks[4] * (X_normal[0][j] ** 2) * (self.XEO_price[index] ** 2) +
                                                2 * greeks[5] * self.XEO_price[index] * X_normal[0][j] *
                                                X_normal[2][j] +
                                                greeks[6] * (X_normal[2][j] ** 2)
                                        )
                                )
                        )
            L_delta.append(np.asarray(L_first))
            L_quadratic.append(np.asarray(L_second))
        #L_delta = -6 * L_delta[0] + (-1) * L_delta[1] + L_delta[2]
        L_delta = -600 * np.asarray(L_delta[0]) + 100 * np.asarray(L_delta[1])
        #L_quadratic = -6 * L_quadratic[0] + (-1) * L_quadratic[1] + L_quadratic[2]
        L_quadratic = -600 * np.asarray(L_quadratic[0]) + 100 * np.asarray(L_quadratic[1])

        # t distribution
        index_0 = np.nonzero(self.option_date[0] == '1/26/18')[0][0]
        option_price_0 = self.option_price[0][index_0:]
        #index_1 = np.nonzero(self.option_weekly_date[1] == '1/26/18')[0][0]
        #option_price_1 = self.option_price[1]
        index_2 = np.nonzero(self.option_date[2] == '1/26/18')[0][0]
        option_price_2 = self.option_price[2][index_2:]
        V_t = - 600 * np.asarray(option_price_0) + 100 * np.asarray(option_price_2)
        L_act = [-(V_t[i + 1] - V_t[i]) for i in self.option_weekly_index[0][week_num:(4 + week_num)]]
        parameters = t.fit(L_act)
        L_t = t.rvs(parameters[0], parameters[1], parameters[2], 10000)
        return [np.asarray(L_delta), np.asarray(L_quadratic), np.asarray(L_t)]

solver = Portfolio_Risk(450000, "AAL.csv", "AAPL.csv", "AMZN.csv", "C.csv",
                        "Hsbc.csv",  "Hmc.csv", "Goog.csv", "JPM.csv", "MS.csv",
                        "Msft.csv", "Ge.csv", "RY.csv", "KO.csv", "V.csv", "WFc.csv")

solver.get_price_list("stocks")
solver.cal_V_t()
#solver.compare()
#solver.risk()
solver.calibrate(60)
#solver.back_test()

bond = Portfolio_Risk(668000,"AAL(1).csv","AAL(2).csv")
bond.get_price_list("bonds")
#bond.get_yield("yields.csv")
bond.cal_V_t_bonds()
bond.calibrate_bonds(60)

#bond.risk_bonds()
