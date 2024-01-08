% This Matlab file is used to compare AmRABK, mRABK, and RABK
% based on the data from SuiteSparse Matrix Collection

close all;
clear;

%%
run_time=50; % average times

%% generated the matrix A using the data from SuiteSparse Matrix Collection

load ash958;% beta=0.6
%load nemsafm; % beta=0.5
%load WorldCities;% beta=0.9
%load Franz1.mat;% beta=0.7
%load crew1.mat;% beta=0.8
%load ch8-8-b1.mat;%beta=0.3
%load model1.mat;% beta=0.5
%load bibd_16_8.mat; % beta=0.7
%load mk10-b2 % beta=0.4
A=Problem.A;
[m,n]=size(A);

beta=0.6; % set the momentum paramter


opts.sparsity=1;

%% some vectors are used to store the desired numerical results
CPU_RABK=zeros(run_time,1);
CPU_mRABK=zeros(run_time,1);
CPU_AmRABK=zeros(run_time,1);

Iter_RABK=zeros(run_time,1);
Iter_mRABK=zeros(run_time,1);
Iter_AmRABK=zeros(run_time,1);

%% executing "run_time" times of the algorithms
for jj=1:run_time
    
    %% generated the right-hand vector b
    x=randn(n,1);
    b=A*x;
    xLS=lsqminnorm(A,b);

    %% parameter setup
    opts.xstar=xLS;
    opts.TOL1=eps^2;
    %opts.Max_iter=100000;
    %ell=floor(norm(A,'fro')^2/norm(A)^2); % size of the block
    ell=30;

    %% run RABK
    [xRABK,OutRABK]=My_RABK(A,b,ell,opts);

    %% run RABK with adaptive momentum
    [xAmRABK,OutAmRABK]=My_AmRABK(A,b,ell,opts);

    %% run mRABK
    
    [xmRABK,OutmRABK]=My_mRABK(A,b,beta,ell,opts);

    %% store the numerical results
    CPU_RABK(jj)=CPU_RABK(jj)+OutRABK.times(end);
    CPU_mRABK(jj)=CPU_mRABK(jj)+OutmRABK.times(end);
    CPU_AmRABK(jj)=CPU_AmRABK(jj)+OutAmRABK.times(end);

    Iter_RABK(jj)=Iter_RABK(jj)+OutRABK.iter;
    Iter_mRABK(jj)=Iter_mRABK(jj)+OutmRABK.iter;
    Iter_AmRABK(jj)=Iter_AmRABK(jj)+OutAmRABK.iter;

    fprintf('Number of iterations: %d,%d,%d\n',OutRABK.iter,OutmRABK.iter,OutAmRABK.iter)
end

%% print the result

fprintf('Average number of iterations: RABK=%8.2f, mRABK=%8.2f, AmRABK=%8.2f\n',mean(Iter_RABK),mean(Iter_mRABK),mean(Iter_AmRABK))

fprintf('Average CPU time: RABK=%8.4f, mRABK=%8.4f, AmRABK=%8.4f\n',mean(CPU_RABK),mean(CPU_mRABK),mean(CPU_AmRABK))

fprintf('The momentum parameter for mRABK: %2.2f\n',beta)

%fprintf(' %8.2f& %8.4f& %8.2f&%8.4f &%2.2f&%8.2f &%8.4f\n',median(Iter_RABK),median(CPU_RABK),median(Iter_mRABK),median(CPU_mRABK),beta,median(Iter_AmRABK),median(CPU_AmRABK))

%fprintf(' %8.2f& %8.4f& %8.2f&%8.4f &%2.2f&%8.2f &%8.4f\n',mean(Iter_RABK),mean(CPU_RABK),mean(Iter_mRABK),mean(CPU_mRABK),beta,mean(Iter_AmRABK),mean(CPU_AmRABK))



