% This Matlab file is used to determine the appropriate momentum parameter 
% for the mRABK method.
%
% Based on the manuscript: 
% [1] Deren Han, Jiaxin Xie. On pseudoinverse-free randomized methods for
% linear systems: Unified framework and acceleration,  arXiv:2208.05437
%
% Coded by Jiaxin Xie, Beihang University, xiejx@buaa.edu.cn
%

close all;
clear;

%% the block size  and the number of iterations
ell=30; % the block size
opts.Max_iter=2000; % the number of iterations

%% generated the Gaussian matrix A
m=5000;
n=100;
rank=100; % rank of the matrix
kappa=10; %the upper bound of the condition number

[U,~]=qr(randn(m, rank), 0);
[V,~]=qr(randn(n, rank), 0);
D = diag(1+(kappa-1).*rand(rank, 1));
A=U*D*V';
clear U V D

%% real-world data

%%%%%% SuiteSparse Matrix Collection

%load ash958;% beta=0.6
%load nemsafm; % beta=0.5
%load WorldCities;% beta=0.9
%load Franz1.mat;% beta=0.7
%load crew1.mat;% beta=0.8
%load ch8-8-b1.mat;%beta=0.3
%load model1.mat;% beta=0.5
%load bibd_16_8.mat; % beta=0.7
%load mk10-b2 % beta=0.4

%A=Problem.A;
%[m,n]=size(A);

%%%%%% LIBSVM data

%load aloi; % beta=0.8
%A=aloi_inst;
%clear aloi;

%load a9a; %beta=0.9
%A=a9a_inst;
%clear a9a;

%load cod-rna % beta=0.9
%A=cod_rna_inst;
%clear cod_rna;

%load ijcnn1 % beta=0.7
%A=ijcnn1_inst;
%clear ijcnn1
%[m,n]=size(A);


%% generated the right-hand vector b
x=randn(n,1);
b=A*x;
xLS=lsqminnorm(A,b);

%% parameter setup
opts.xstar=xLS;
%opts.sparsity=1; % if real-world data, we should use this setting
opts.TOL1=eps^(2);


%% run mRABK with different values of the momentum parameters
[xmRABK0,OutmRABK0]=My_mRABK(A,b,0.0,ell,opts);
fprintf('Done\n')
[xmRABK1,OutmRABK1]=My_mRABK(A,b,0.1,ell,opts);
fprintf('Done\n')
[xmRABK2,OutmRABK2]=My_mRABK(A,b,0.2,ell,opts);
fprintf('Done\n')
[xmRABK3,OutmRABK3]=My_mRABK(A,b,0.3,ell,opts);
fprintf('Done\n')
[xmRABK4,OutmRABK4]=My_mRABK(A,b,0.4,ell,opts);
fprintf('Done\n')
[xmRABK5,OutmRABK5]=My_mRABK(A,b,0.5,ell,opts);
fprintf('Done\n')
[xmRABK6,OutmRABK6]=My_mRABK(A,b,0.6,ell,opts);
fprintf('Done\n')
[xmRABK7,OutmRABK7]=My_mRABK(A,b,0.7,ell,opts); 
%fprintf('Done\n')
[xmRABK8,OutmRABK8]=My_mRABK(A,b,0.8,ell,opts); % sometimes disconvergence
%fprintf('Done\n')
%[xmRABK9,OutmRABK9]=My_mRABK(A,b,0.9,ell,opts);
%fprintf('Done\n')

%% plot the results
figure
semilogy(OutmRABK0.error,'LineWidth',1)
hold on
semilogy(OutmRABK1.error,'LineWidth',1)
%legend('0','0.1')
semilogy(OutmRABK2.error,'LineWidth',1)
%legend('0','0.1','0.2')
semilogy(OutmRABK3.error,'LineWidth',1)
%legend('0','0.1','0.2','0.3')
semilogy(OutmRABK4.error,'LineWidth',1)
%legend('0','0.1','0.2','0.3','0.4')
semilogy(OutmRABK5.error,'LineWidth',1)
%legend('0','0.1','0.2','0.3','0.4','0.5')
semilogy(OutmRABK6.error,'LineWidth',1)
%legend('0','0.1','0.2','0.3','0.4','0.5','0.6')
semilogy(OutmRABK7.error,'LineWidth',1)
%legend('0','0.1','0.2','0.3','0.4','0.5','0.6','0.7')
semilogy(OutmRABK8.error,'LineWidth',1)
legend('0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8')
%semilogy(OutmRABK9.error,'LineWidth',1)
%legend('0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9')
xlabel('Iter')
ylabel('RSE')
title('mRABK with different values of the momentum parameters')
fprintf('Size of the matrix m=%d, n=%d\n',m,n)

%fprintf('Rank of the matrix r=%d\n',rank)




