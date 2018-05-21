filename = "WeeklyLosses.xlsx";
distribution = 'Gaussian';
[Cond_mean_Normal, Cond_variance_Normal] = calibrateGarch(filename,distribution);
distribution = 't';
[Cond_mean_T, Cond_variance_T] = calibrateGarch(filename,distribution);

% plot_graph(Cond_mean_Normal,Cond_variance_Normal);
% plot_graph(Cond_mean_T,Cond_variance_T);

function plot_graph(Cond_mean,Cond_Variance)
    
    T = length(Cond_mean_Normal);
    figure
    hold on 
    subplot(2,1,1)
    plot(Cond_mean_Normal)
    xlim([0,T])
    title("Conditional Mean")
    
    subplot(2,1,2)
    plot(Cond_variance_Normal)
    xlim([0,T])
    title("Conditional Variance")
    hold off
    
end


function [L_act] = calculateWeeklyLosses(filename, week_num)
    data = xlsread(filename);
    [row,col] = size(data)
    L_act = data((week_num+1)-200:week_num,:);    
end

function [Con_mu1,Con_Var1] =  calibrateGarch(filename,distribution)
    Con_mu1 = [];
    Con_Var1 = [];
    residuals = [];
    if distribution == 'Gaussian'
        j = 200;
    elseif distribution == 't'
        j = 225
    end
    for i = j:311
        L_act = calculateWeeklyLosses(filename, i);
        mdl = arima('ARLags',1,'MALags',1,'Variance',garch(1,1));
        mdl.Distribution=distribution;
        estMdl = estimate(mdl, L_act);
        [res,v] = infer(estMdl, L_act);
        Con_mu1 = [Con_mu1, (L_act(200) - res(200))];
        Con_Var1 = [Con_Var1, v(200)];
        residuals = [residuals, res(200)];
    end
    Con_mu1 = Con_mu1';
    Con_Var1 = Con_Var1';
end

