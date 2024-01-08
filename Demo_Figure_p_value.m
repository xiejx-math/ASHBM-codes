% This Matlab file is used to compare RBKU, RABK, AmRBKU, and AmRABK
% the coefficient matrix is Gaussian matrix

close all;
clear;

%%
m=2^10;
n=2^7;
rank=100;
kappa=40; %Desired condition number
%%
setvaluep=[0:1:log2(m)];
sizeP=length(setvaluep);
k_time=sizeP; % used for setting the value of p

run_time=50; % average times

%%
Rho_RABK=zeros(run_time,k_time);
Rho_AmRABK=zeros(run_time,k_time);
Rho_upperbound=zeros(run_time,k_time);

%%
Iter_RABK=zeros(run_time,k_time);
Iter_AmRABK=zeros(run_time,k_time);
Iter_RBKU=zeros(run_time,k_time);
Iter_AmRBKU=zeros(run_time,k_time);

%%
CPU_RABK=zeros(run_time,k_time);
CPU_AmRABK=zeros(run_time,k_time);
CPU_RBKU=zeros(run_time,k_time);
CPU_AmRBKU=zeros(run_time,k_time);

for ii=1:k_time
    p=2^(setvaluep(ii));% the size of the block

    for jj=1:run_time
        [U,~]=qr(randn(m, rank), 0);
        [V,~]=qr(randn(n, rank), 0);
        D = diag(1+(kappa-1).*rand(rank, 1));
        A=U*D*V';
        clear U V D

        %% generated the right-hand vector b
        x=randn(n,1);
        b=A*x;
        xLS=lsqminnorm(A,b);

        %% parameter setup
        opts.xstar=xLS;
        %opts.TOL=10^(-12);
        opts.TOL1=eps^2;
        %opts.Max_iter=5000;
        opts.permS=randperm(m);
        S=opts.permS;
        


        %% RABK with momentum
        [xAmRABK,OutAmRABK]=My_AmRABK(A,b,p,opts);

        rho_AmRABK=compute_rhok(OutAmRABK.error);

        %% RABK 
        [xRABK,OutRABK]=My_RABK(A,b,p,opts);

        rho_RABK=compute_rhok(OutRABK.error);

         %% RBKU with momentum
        [xAmRBKU,OutAmRBKU]=My_AmRBKU(A,b,p,opts);


        %% RBKU 
        [xRBKU,OutRBKU]=My_RBKU(A,b,p,opts);

        %% estimate the upper bound
        tau=floor(m/p);
        blockAnormfro=zeros(tau,1);
        beta_max=0;
        A=A(S,:);
        b=b(S);

        for i=1:tau
            if i==tau
                ps=((i-1)*p+1):1:m;
            else
                ps=((i-1)*p+1):1:(i*p);
            end
            blockAnormfro(i)=norm(A(ps,:),'fro')^2;
            beta_max=max(beta_max,norm(A(ps,:))^2/blockAnormfro(i));
        end

        if rank==n
            i_kappa_F=svds(A,1,'smallestnz')^2/norm(A,'fro')^2;
        else
            S=svd(A);
            i_kappa_F=S(rank)^2/norm(A,'fro')^2;
        end

        rho_upperbound=1-i_kappa_F/beta_max;
        
        %% update and store the results
        %%%%%%
        Rho_RABK(jj,ii)=rho_RABK;
        Rho_AmRABK(jj,ii)=rho_AmRABK;
        Rho_upperbound(jj,ii)=rho_upperbound;

        %%%%%%
        Iter_RABK(jj,ii)=OutRABK.iter;
        Iter_AmRABK(jj,ii)=OutAmRABK.iter;
        Iter_RBKU(jj,ii)=OutRBKU.iter;
        Iter_AmRBKU(jj,ii)=OutAmRBKU.iter;

        %%%%%
        CPU_RABK(jj,ii)=OutRABK.times(end);
        CPU_AmRABK(jj,ii)=OutAmRABK.times(end);
        CPU_RBKU(jj,ii)=OutRBKU.times(end);
        CPU_AmRBKU(jj,ii)=OutAmRBKU.times(end);

        %% print

        fprintf('Upper bound =%2.4f, RABK=%2.4f, AmRABK=%2.4f\n',rho_upperbound,rho_RABK,rho_AmRABK);
    end
     %% print
    fprintf('Done,iter=%d\n',ii);
end


%% plot the results

%% set the colour 
lightgray =   [0.8 0.8 0.8];
mediumgray =  [0.6 0.6 0.6];
lightred =    [1 0.9 0.9];
mediumred =   [1 0.6 0.6];
lightgreen =  [0.9 1 0.9];
mediumgreen = [0.6 1 0.6];
lightblue =   [0.9 0.9 1];
mediumblue =  [0.6 0.6 1];
lightmagenta =   [1 0.9 1];
mediummagenta =  [1 0.6 1];


%%
num_iter_array=setvaluep';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot the k*p/m
%%%
P_matrix=zeros(run_time,k_time);
for kk=1:run_time
    P_matrix(kk,:)=2.^(num_iter_array');
end

RABK_value=Iter_RABK.*P_matrix/m;
AmRABK_value=Iter_AmRABK.*P_matrix/m;
RBKU_value=Iter_RBKU.*P_matrix/m;
AmRBKU_value=Iter_AmRBKU.*P_matrix/m;

%%%%%%%
display_names1 = {'RBKU', 'AmRBKU','RABK', 'AmRABK'};
arrsIter1 = {RBKU_value', AmRBKU_value',RABK_value', AmRABK_value'};
num_methods1 = length(arrsIter1);
%line_colors1 = {'black', 'blue', 'green', 'magenta'};
%minmax_colors1 = {lightgray,lightblue, lightgreen, lightmagenta};
%quant_colors1 = {mediumgray, mediumgreen,mediumgreen, mediummagenta};
line_colors1 = {'black', 'blue', 'green', 'red'};
minmax_colors1 = {lightgray,lightblue, lightgreen, lightred};
quant_colors1 = {mediumgray, mediumgreen,mediumgreen, mediumred};
display_legend1 = false;
max_val_in_plot1 = 1e5;

%%
opts1.m=m;
opts1.n=n;
opts1.rank=rank;
opts1.kappa=kappa;
opts1.comparison=1;
opts1.plotaxes=1;
opts1.axesposition=[0.21,0.57,0.3,0.32];
opts1.startp=setvaluep(1);

%%
[x_arrays_iter1, quantiles_iter1] = compute_and_plot_p(num_iter_array, arrsIter1, ...
    num_methods1, line_colors1, display_names1, ...
    minmax_colors1, quant_colors1, display_legend1, max_val_in_plot1,opts1);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot the CPU time
display_names2 = {'RBKU', 'AmRBKU','RABK', 'AmRABK'};
arrsIter2 = {CPU_RBKU', CPU_AmRBKU',CPU_RABK', CPU_AmRABK'};
num_methods2 = length(arrsIter2);
% line_colors2 = {'black', 'blue', 'green', 'magenta'};
% minmax_colors2 = {lightgray,lightblue, lightgreen, lightmagenta};
% quant_colors2 = {mediumgray, mediumgreen,mediumgreen, mediummagenta};

line_colors2 = {'black', 'blue', 'green', 'red'};
minmax_colors2 = {lightgray,lightblue, lightgreen, lightred};
quant_colors2 = {mediumgray, mediumgreen,mediumgreen, mediumred};

display_legend2 = false;
max_val_in_plot2 = 1e2;

%%
opts2.m=m;
opts2.n=n;
opts2.rank=rank;
opts2.kappa=kappa;
opts2.CPU=1;
opts2.plotaxes=1;
opts2.axesposition=[0.21,0.57,0.3,0.32];
opts2.startp=setvaluep(5);
%%
[x_arrays_iter2, quantiles_iter2] = compute_and_plot_p(num_iter_array, arrsIter2, ...
    num_methods2, line_colors2, display_names2, ...
    minmax_colors2, quant_colors2, display_legend2, max_val_in_plot2,opts2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot the convergence factor
display_names = {'Upper bound', 'RABK', 'AmRABK'};
arrsIter = {Rho_upperbound', Rho_RABK', Rho_AmRABK'};
num_methods = length(arrsIter);
% line_colors = {'black','green', 'magenta'};
% minmax_colors = { lightgray, lightgreen, lightmagenta};
% quant_colors = { mediumgray,mediumgreen, mediummagenta};
line_colors = {'black','green', 'red'};
minmax_colors = { lightgray, lightgreen, lightred};
quant_colors = { mediumgray,mediumgreen, mediumred};

display_legend = false;
max_val_in_plot = 1;

%%
opts.m=m;
opts.n=n;
opts.rank=rank;
opts.kappa=kappa;
opts.axesposition=[0.21,0.17,0.3,0.32];
opts.rate=1;
opts.plotaxes=1;
opts.startp=setvaluep(1);
%%
[x_arrays_iter, quantiles_iter] = compute_and_plot_p(num_iter_array, arrsIter, ...
    num_methods, line_colors, display_names, ...
    minmax_colors, quant_colors, display_legend, max_val_in_plot,opts);


%% function to estimate the convergence factor

function rhok=compute_rhok(errors)

k=length(errors)-1;
rhok=nthroot(errors(end),k);
end


