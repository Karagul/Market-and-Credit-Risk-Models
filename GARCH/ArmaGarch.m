filename = "PortfolioValue.xlsx";
data = xlsread(filename)
roll=200;
n=length(data)
for i=1:n-roll+1                           
    train=data(i:i+roll-1);
    %normal Distribution
    Md1 = arima('ARLags',1,'MALags',1,'Variance',garch(1,1));
    EstMd1 = estimate(Md1,train);           
    [res,cond_var] = infer(EstMd1,train); 
    %calculate conditional mean
    cond_mean=train-res; 
    
    %T distribution
    Md1_T= arima('ARLags',1,'MALags',1,'Variance',garch(1,1));
    Md1_T.Distribution = 't';
    EstMd1_T = estimate(Md1_T,train); 
    [res_T,cond_var_T] = infer(EstMd1_T,train);  
    cond_mean_T=train-res_T;               
    if i ~= 1
        %get the 200th conditional mean into the mean list
        mu_list_N=[mu_list_N,cond_mean(roll)];
        var_list_N=[var_list_N,cond_var(roll)];
        res_list_N=[res_list_N,res(roll)];
        
        mu_list_T=[mu_list_T,cond_mean_T(roll)];
        %var_list_T=[var_list_T,cond_var_T(roll)];
        var_list_T=[var_list_T,cond_var_T(roll)];
        res_list_T=[res_list_T,res_T(roll)];
        dof_list=[dof_list,EstMd1_T.Distribution.DoF];
        %calculate unconditional mean and variance
        mu_list=[mu_list,mean(train)];
        var_list=[var_list,var(train)];
                
    else
        mu_list_N=cond_mean';
        var_list_N=cond_var';
        res_list_N=res';
        
        mu_list_T=cond_mean_T';
        var_list_T=cond_var_T';
        res_list_T=res_T';
        
        dof_list=EstMd1_T.Distribution.DoF*ones(1,roll);
        mu_list=ones(1,roll)*mean(train);
        var_list=ones(1,roll)*var(train);
        
    end
    disp(i)
end
%%plot mean and variance 
subplot(2,1,1);
plot(mu_list_N)
hold on
plot(mu_list_T)
hold off
xlim([0,115])
xlabel('time')
ylabel('value')
title('Conditional Mean with gaussian and T distribution ')
legend('gaussian distribution','t distribution')

subplot(2,1,2)
plot(var_list_N)
hold on
plot(var_list_T)
hold off
xlabel('time')
ylabel('value')
title('Conditional variance with gaussian and T distribution ')
legend('gaussian distribution','t distribution')

%% calculate VaR and ES(normal)
% define alpha=0.99
alpha=0.99
Z=norminv(alpha,0,1)
VaR=mu_list_N+sqrt(var_list_N)*Z
ES=mu_list_N+sqrt(var_list_N)*normpdf(Z,0,1)/(1-alpha);

figure
plot(VaR(200:end))
hold on 
plot(ES(200:end))
plot(data(200:end))
hold off
xlabel('time')
ylabel('value')
title('VaR and ES with gaussian distribution(alpha=0.99) ')
legend('VaR','ES','Real Losses')

%% t distribution
t_inv= tinv(alpha,dof_list);
Es_L=tpdf(t_inv,dof_list)/(1-alpha).*(dof_list+(t_inv.^2))./(dof_list-1);
VaR_T=mu_list_T+sqrt(var_list_T).*t_inv;
ES_T=mu_list_T+sqrt(var_list_T).*Es_L;

figure
plot(VaR_T(200:end))
hold on 
plot(ES_T(200:end))
plot(data(200:end))
hold off
xlabel('time')
ylabel('value')
title('VaR and ES with t distribution(alpha=0.99) ')
legend('VaR','ES','real losses')

%% unconditional VaR and ES
VaR_UN=mu_list+sqrt(var_list)*Z
ES_UN=mu_list+sqrt(var_list)*normpdf(Z,0,1)/(1-alpha);
VaR_UT=mu_list+sqrt(var_list).*t_inv;
ES_UT=mu_list+sqrt(var_list).*Es_L;

%% plot 
subplot(2,1,1)
plot(VaR(200:end))
hold on 
plot(VaR_UN(200:end))
hold off
xlabel('time')
ylabel('value')
title('compare VaR under normal distribution (alpha=0.95) ')
legend('VaR(Conditional)','VaR(Unconditional)')

subplot(2,1,2)
plot(VaR_T(200:end))
hold on
plot(VaR_UT(200:end))
hold off
xlabel('time')
ylabel('value')
title('compare VaR under t distribution (alpha=0.95) ')
legend('VaR(Conditional)','VaR(Unconditional)')
%%
subplot(2,1,1)
plot(ES(200:end))
hold on
plot(ES_UN(200:end))
hold off
xlabel('time')
ylabel('value')
title('compare ES under normal distribution (alpha=0.95) ')
legend('ES(Conditional)','ES(Unconditional)')

subplot(2,1,2)

plot(ES_T(200:end))
hold on 
plot(ES_UT(200:end))
hold off
xlabel('time')
ylabel('value')
title('compare ES under t distribution (alpha=0.95) ')
legend('ES(Conditional)','ES(Unconditional)')