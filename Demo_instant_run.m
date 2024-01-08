% This Matlab file provided is an instant run code for
% the RBKU, mRBKU, AmBKU, RABK, mRABK, and AmRABK methods

close all;
clear;

%% generated the matrix A
m=1000;
n=100;
rank=n;
ell=50;% the size of the block
kappa=40; %Desired condition number
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
opts.Max_iter=100000;

%% a uniform random permutation on [m]
opts.permS=randperm(m);


%% run AmRABK 
[xAmRABK,OutAmRABK]=My_AmRABK(A,b,ell,opts);

%% run AmRBKU 
[xAmRBKU,OutAmRBKU]=My_AmRBKU(A,b,ell,opts);

%% run RABK
[xRABK,OutRABK]=My_RABK(A,b,ell,opts);

%% run RBKU
[xRBKU,OutRBKU]=My_RBKU(A,b,ell,opts);

%% run mRBKU
%%%compute the step-size alpha
tic;
normAfro=norm(full(A),'fro')^2;
AAt=A*A';
diagAAt=diag(diag(AAt));
if ell==1
    beta3=m*norm(full(diagAAt));
else
    beta3=m*(ell-1)/((m-1)*ell)*norm(AAt+(m-ell)/(ell-1)*diagAAt);
end
alpha=1*normAfro/(beta3);
timeX=toc;

%%%% set the momentum parameter beta
betamRBKU=0.5;
[xmRBK,OutmRBKU]=My_mRBKU(A,b,alpha,betamRBKU,ell,opts);


%% run mRABK
betamRABK=0.6;% the momentum parameter
[xmRABK,OutmRABK]=My_mRABK(A,b,betamRABK,ell,opts);

%% plot the numerical results

%%%%%%% plot the RSE vs the number of iterations
figure
semilogy(OutAmRABK.error,'LineWidth',1.2)
hold on
semilogy(OutmRABK.error,'LineWidth',1.2)
semilogy(OutRABK.error,'LineWidth',1.2)
semilogy(OutAmRBKU.error,'LineWidth',1.2)
semilogy(OutmRBKU.error,'LineWidth',1.2)
semilogy(OutRBKU.error,'LineWidth',1.2)
legend('AmRABK','mRABK','RABK','AmRBKU','mRBKU','RBKU')
ylabel('RSE')
xlabel('Iter')
txt=title( ['{\tt randn}',',$m=$ ',num2str(m),',$n=$ ',num2str(n),',$r=$ ',num2str(rank),',$\kappa=$',num2str(kappa)]);
%txt=title( ['{\tt randn }',',$n=$ ',num2str(n)]);
set(txt, 'Interpreter', 'latex');


%%%%%%% plot the RSE vs the CPU time
figure
semilogy(OutAmRABK.times,OutAmRABK.error,'LineWidth',1.2)
hold on
semilogy(OutmRABK.times,OutmRABK.error,'LineWidth',1.2)
semilogy(OutRABK.times,OutRABK.error,'LineWidth',1.2)
semilogy(OutAmRBKU.times,OutAmRBKU.error,'LineWidth',1.2)
semilogy(OutmRBKU.times,OutmRBKU.error,'LineWidth',1.2)
semilogy(OutRBKU.times,OutRBKU.error,'LineWidth',1.2)
legend('AmRABK','mARBK','RABK','AmRBKU','mRBKU','RBKU')
ylabel('RSE')
xlabel('CPU time')
txt=title( ['{\tt randn}',',$m=$ ',num2str(m),',$n=$ ',num2str(n),',$r=$ ',num2str(rank),',$\kappa=$',num2str(kappa)]);
%txt=title( ['{\tt randn }',',$n=$ ',num2str(n)]);
set(txt, 'Interpreter', 'latex');





