% This Matlab file is used to depict the convergence of AmRABK and RABK

close all;
clear;

m=2^10;
n=2^7;
rank=n;
kappa=10000; %Desired condition number
opts.Max_iter=20000; %Max iteration


run_time=10; % average times

%% the vector is used to store the numerical results 
RSE_RABK=zeros(run_time,opts.Max_iter);
RSE_AmRABK=zeros(run_time,opts.Max_iter);
RSE_UB=zeros(run_time,opts.Max_iter);

for ii=1:run_time

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
    opts.TOL=10^(-32);
    opts.TOL1=eps^2;

    opts.permS=randperm(m);
    S=opts.permS;
  

    %% RABK with momentum
    p=2^5;% the size of the block

    [xAmRABK,OutAmRABK]=My_AmRABK(A,b,p,opts);

    [xRABK,OutRABK]=My_RABK(A,b,p,opts);

    %%
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

    max_iter=max(length(OutAmRABK.error),length(OutRABK.error));

    kk=0:1:(opts.Max_iter-1);
    ratev=rho_upperbound.^kk;


    %%

    RSE_RABK(ii,:)=OutRABK.error(1:opts.Max_iter);%
    RSE_AmRABK(ii,:)=OutAmRABK.error(1:opts.Max_iter);%
    RSE_UB(ii,:)=ratev;

    fprintf('Done %d\n',ii)
end



%% plot the convergence factor
%%

xlabel_i=1:opts.Max_iter;
num_iter_array=xlabel_i';

%% plot errors
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


display_names = {'Upper bound','RABK','AmRABK'};
%arrsIter = {(RSE_UB(:,Xlabel_Numb))',(RSE_RABK(:,Xlabel_Numb))',(RSE_AmRABK(:,Xlabel_Numb))'};
arrsIter = {RSE_UB',RSE_RABK',RSE_AmRABK'};

num_methods = length(arrsIter);

% line_colors = {'black','green','magenta'};
% minmax_colors = { lightgray, lightgreen,lightmagenta};
% quant_colors = { mediumgray,mediumgreen,mediummagenta};

line_colors = {'black','green','red'};
minmax_colors = { lightgray, lightgreen,lightred};
quant_colors = { mediumgray,mediumgreen,mediumred};

%line_colors = {'black', 'green', 'red', 'blue'};
%minmax_colors = {lightgray, lightgreen, lightred, lightblue};
%quant_colors = {mediumgray, mediumgreen, mediumred, mediumblue};

display_legend = true;
max_val_in_plot = 1;

%%
[x_arrays_iter, quantiles_iter] =  compute_and_plot_quantiles_in_logscale(num_iter_array, arrsIter, ...
    num_methods, line_colors, display_names, ...
    minmax_colors, quant_colors, display_legend, max_val_in_plot);
ylabel('RSE')
xlabel('Iter')
txt=title( ['{\tt randn}',',$m=$ ',num2str(m),',$n=$ ',num2str(n),',$r=$ ',num2str(rank),',$\kappa=$',num2str(kappa)]);
set(txt, 'Interpreter', 'latex');




node_Numb=opts.Max_iter;
Gap_Numb=ceil(opts.Max_iter/node_Numb);
Xlabel_Numb=[1:Gap_Numb:opts.Max_iter,opts.Max_iter];

figure
semilogy(Xlabel_Numb,median(RSE_UB(:,Xlabel_Numb)),'k-','LineWidth',1.2)
hold on
semilogy(Xlabel_Numb,median(RSE_RABK(:,Xlabel_Numb)),'g--','LineWidth',1.5)
semilogy(Xlabel_Numb,median(RSE_AmRABK(:,Xlabel_Numb)),'r:','LineWidth',1.5)
legend('Upper bound','RABK','AmRABK','Location','best')
xlabel('Number of iterations')
ylabel('RSE')
txt=title( ['{\tt randn}',',$m=$ ',num2str(m),',$n=$ ',num2str(n),',$r=$ ',num2str(rank),',$\kappa=$',num2str(kappa)]);
%txt=title( ['{\tt randn}',',$n=$ ',num2str(n)]);
set(txt, 'Interpreter', 'latex');




