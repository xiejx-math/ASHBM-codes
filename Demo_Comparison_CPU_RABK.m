% This Matlab file is used to compare AmRABK, mRABK, and RABK
% the coefficient matrix is Gaussian matrix

close all;
clear;

%%
n=100; % Number of columns
rank=n; % Number of rank
kappa=40; % upper bound for the condition number

ell=30; % size of the block
beta=0.7; % the momentum parameter for mRABK

%%
run_time=50; % average times
m1=500; % the initial setting for the number of rows
k_time=10; % used for updating the number of rows

%% some vectors are used to store the computed results
CPU_RABK=zeros(run_time,k_time);
CPU_mRABK=zeros(run_time,k_time);
CPU_AmRABK=zeros(run_time,k_time);

Iter_RABK=zeros(run_time,k_time);
Iter_mRABK=zeros(run_time,k_time);
Iter_AmRABK=zeros(run_time,k_time);

%% execute the iteration
for ii=1:k_time
    m=ii*m1; % set the number of row

    %%%% for the fixed m and n, executing "run_time" times of the algorithms
    for jj=1:run_time
        %% generated the matrix A
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
        opts.TOL1=eps^2;
        opts.permS=randperm(m);
        %opts.Max_iter=100000;
        
        %% run RABK
        [xRABK,OutRABK]=My_RABK(A,b,ell,opts);

        %% run RABK with adaptive momentum
        [xAmRABK,OutAmRABK]=My_AmRABK(A,b,ell,opts);

        %% run mRABK
        [xmRABK,OutmRABK]=My_mRABK(A,b,beta,ell,opts);

        %% store the compute results
        CPU_RABK(jj,ii)=CPU_RABK(jj,ii)+OutRABK.times(end);
        CPU_mRABK(jj,ii)=CPU_mRABK(jj,ii)+OutmRABK.times(end);
        CPU_AmRABK(jj,ii)=CPU_AmRABK(jj,ii)+OutAmRABK.times(end);

        Iter_RABK(jj,ii)=Iter_RABK(jj,ii)+OutRABK.iter;
        Iter_mRABK(jj,ii)=Iter_mRABK(jj,ii)+OutmRABK.iter;
        Iter_AmRABK(jj,ii)=Iter_AmRABK(jj,ii)+OutAmRABK.iter;

        fprintf('Number of iterations: %d,%d,%d\n',OutRABK.iter,OutmRABK.iter,OutAmRABK.iter)
    end
    fprintf('Done,iter=%d\n',ii);
end



%% plot the result

%% setting the colour

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
timeMN=1:k_time;
timeMN=m1/n*timeMN;
num_iter_array=timeMN';

%%
display_names = {'RABK', 'mRABK', 'AmRABK'};
arrsIter = {Iter_RABK', Iter_mRABK', Iter_AmRABK'};

num_methods = length(arrsIter);
%line_colors = {'black', 'red', 'blue'};
%minmax_colors = {lightgray, lightred, lightblue};
%quant_colors = {mediumgray, mediumred, mediumblue};

line_colors = {'blue','green', 'red'};
minmax_colors = { lightblue, lightgreen, lightred};
quant_colors = { mediumblue,mediumgreen, mediumred};

%line_colors = {'blue','green', 'magenta'};
%minmax_colors = { lightblue, lightgreen, lightmagenta};
%quant_colors = { mediumblue,mediumgreen, mediummagenta};

display_legend = true;
max_val_in_plot = 1e5;

%% plot the number of iterations vs m/n
[x_arrays_iter, quantiles_iter] = compute_and_plot_Iter(num_iter_array, arrsIter, ...
    num_methods, line_colors, display_names, ...
    minmax_colors, quant_colors, display_legend, max_val_in_plot);
ylabel('Iter')
xlabel('m/n')

txt=title( ['{\tt randn}',',$n=$ ',num2str(n),',$r=$ ',num2str(rank),',$\kappa=$',num2str(kappa)]);

set(txt, 'Interpreter', 'latex');



%% plot the CPU time vs m/n
arrsCPU = {CPU_RABK', CPU_mRABK', CPU_AmRABK'};

[x_arrays_CPU, quantiles_CPU] = compute_and_plot_Iter(num_iter_array, arrsCPU, ...
    num_methods, line_colors, display_names, ...
    minmax_colors, quant_colors, display_legend, max_val_in_plot);
ylabel('CPU time')
xlabel('m/n')

txt=title( ['{\tt randn}',',$n=$ ',num2str(n),',$r=$ ',num2str(rank),',$\kappa=$',num2str(kappa)]);
set(txt, 'Interpreter', 'latex');




