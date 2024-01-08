% This Matlab file is used to compare AmRABK, mRABK, and RABK
% based on the data from LIBSVM

clear;
close all;

%%
opts.Max_iter=150000; % Number of iterations
ell=300;% size of the block
run_time=50; % average times

%% LIBSVM data
load aloi; % beta=0.8 % Max_iter=150000
A=aloi_inst;
clear aloi;

%load a9a; %beta=0.9 % Max_iter=3500
%A=a9a_inst;
%clear a9a;

%load cod-rna % beta=0.9  % Max_iter=8000
%A=cod_rna_inst;
%clear cod_rna;

%load ijcnn1 % beta=0.7 % Max_iter=50
%A=ijcnn1_inst;
%clear ijcnn1

beta=0.8;

[m,n]=size(A);

%%
opts.TOL=10^(-32); 
opts.TOL1=eps^2;
opts.sparsity=1;
%% the vector is used to store the numerical results
RSE_RABK=zeros(run_time,opts.Max_iter);
RSE_AmRABK=zeros(run_time,opts.Max_iter);
RSE_mRABK=zeros(run_time,opts.Max_iter);

CPU_RABK=zeros(run_time,opts.Max_iter);
CPU_AmRABK=zeros(run_time,opts.Max_iter);
CPU_mRABK=zeros(run_time,opts.Max_iter);

for ii=1:run_time
    x=randn(n,1);
    b=A*x;

    xLS=lsqminnorm(A,b);

    %% parameter setup
    opts.xstar=xLS;
    opts.permS=randperm(m);
    S=opts.permS;
    
    %%
    tau=floor(m/ell);
    %prob=zeros(t,1);
    A=A(S,:);
    b=b(S);
    blockAnormfro=zeros(tau,1);
    normAfro=norm(A,'fro')^2;
    %blockAnormfro=zeros(t,1);
    

    beta_max=0;
    for i=1:tau
        if i==tau
            ps=((i-1)*ell+1):1:m;
        else
            ps=((i-1)*ell+1):1:(i*ell);
        end

        Aps=A(ps,:);
        blockAnormfro(i)=norm(A(ps,:),'fro')^2;
        Aarrs{i}=Aps;
        barrs{i}=b(ps);
        %blockAnormfro(i)=norm(A(ps,:),'fro')^2;
        %ps=(floor((i-1)*m/t)+1):1:floor(i*m/t);
        %blockAnormfro(i)=norm(A(ps,:),'fro')^2;
        %beta_max=max(beta_max,normest(Aps)^2/blockAnormfro(i));
        beta_max=max(beta_max,norm(full(Aps))^2/blockAnormfro(i));
    end
    prob=blockAnormfro/normAfro;
    alpha=1/beta_max;
    cumsumpro=cumsum(prob);
    %%
    opts.Aarrs=Aarrs;
    opts.barrs=barrs;
    opts.cumsumpro=cumsumpro;
    opts.alpha=alpha;
    opts.blockAnormfro=blockAnormfro;

    opts.probset=1;

    %% RABK
    [xRABK,OutRABK]=My_RABK(A,b,ell,opts);

    %% RABK with adaptive momentum
    [xAmRABK,OutAmRABK]=My_AmRABK(A,b,ell,opts);

    %% mRABK

    [xmRABK,OutmRABK]=My_mRABK(A,b,beta,ell,opts);

    %%

    RSE_RABK(ii,:)=OutRABK.error(1:opts.Max_iter);%
    RSE_AmRABK(ii,:)=OutAmRABK.error(1:opts.Max_iter);%
    RSE_mRABK(ii,:)=OutmRABK.error(1:opts.Max_iter);


    CPU_RABK(ii,:)=OutRABK.times(1:opts.Max_iter);
    CPU_AmRABK(ii,:)=OutAmRABK.times(1:opts.Max_iter);
    CPU_mRABK(ii,:)=OutmRABK.times(1:opts.Max_iter);

    fprintf('Done,iter=%d\n',ii);
end





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


display_names = {'RABK','mRABK','AmRABK'};
%arrsIter = {(RSE_UB(:,Xlabel_Numb))',(RSE_RABK(:,Xlabel_Numb))',(RSE_AmRABK(:,Xlabel_Numb))'};
arrsIter = {RSE_RABK',RSE_mRABK',RSE_AmRABK'};

num_methods = length(arrsIter);

% line_colors = {'black','green','magenta'};
% minmax_colors = { lightgray, lightgreen,lightmagenta};
% quant_colors = { mediumgray,mediumgreen,mediummagenta};

line_colors = {'green','blue','red'};
minmax_colors = { lightgreen,lightblue,lightred};
quant_colors = { mediumgreen,mediumblue,mediumred};

%line_colors = {'black', 'green', 'red', 'blue'};
%minmax_colors = {lightgray, lightgreen, lightred, lightblue};
%quant_colors = {mediumgray, mediumgreen, mediumred, mediumblue};

display_legend = true;
max_val_in_plot = 1;

%%
[x_arrays_iter, quantiles_iter] =  compute_and_plot_LIBSVM_quantiles_in_logscale(num_iter_array, arrsIter, ...
    num_methods, line_colors, display_names, ...
    minmax_colors, quant_colors, display_legend, max_val_in_plot);
ylabel('RSE')
xlabel('Iter')
txt=title( ['{\tt aloi}',',$m=$ ',num2str(m),',$n=$ ',num2str(n)]);
set(txt, 'Interpreter', 'latex');

%% plot RSE vs CPU time

maxCPU_AmRABK=max(max(CPU_AmRABK));
maxCPU_mRABK=max(max(CPU_mRABK));
maxCPU_RABK=max(max(CPU_RABK));
maxCPU=max(maxCPU_RABK,max(maxCPU_mRABK,maxCPU_AmRABK));

minCPU_AmRABK=min(min(CPU_AmRABK));
minCPU_mRABK=min(min(CPU_mRABK));
minCPU_RABK=min(min(CPU_RABK));
minCPU=min(minCPU_RABK,min(minCPU_mRABK,minCPU_AmRABK));

xlabel_i=maxCPU/opts.Max_iter*[1:opts.Max_iter];
num_iter_array=xlabel_i';

%%
[x_arrays_iter, quantiles_iter] =  compute_and_plot_LIBSVM_quantiles_in_logscale(num_iter_array, arrsIter, ...
    num_methods, line_colors, display_names, ...
    minmax_colors, quant_colors, display_legend, max_val_in_plot);
ylabel('RSE')
xlabel('CPU time')
txt=title( ['{\tt aloi}',',$m=$ ',num2str(m),',$n=$ ',num2str(n)]);
set(txt, 'Interpreter', 'latex');


return
%%

figure
semilogy(median(RSE_RABK),'g-','LineWidth',1.2);
hold on
semilogy(median(RSE_mRABK),'b--','LineWidth',1.2);
semilogy(median(RSE_AmRABK),'r:','LineWidth',1.2);
legend('RABK','mRABK','AmRABK')
xlabel('Iter')
ylabel('RSE')
txt=title( ['{\tt cod-rna}',',$m=$ ',num2str(m),',$n=$', num2str(n)] );
set(txt, 'Interpreter', 'latex');

figure
semilogy(median(CPU_RABK),median(RSE_RABK),'g-','LineWidth',1.2);
hold on
semilogy(median(CPU_mRABK),median(RSE_mRABK),'b--','LineWidth',1.2);
semilogy(median(CPU_AmRABK),median(RSE_AmRABK),'r:','LineWidth',1.2);
legend('RABK','mRABK','AmRABK')
xlabel('CPU time')
ylabel('RSE')
txt=title( ['{\tt cod-rna}',',$m=$ ',num2str(m),',$n=$', num2str(n)] );
set(txt, 'Interpreter', 'latex');

%RSE_RBK=RSE_RBK/run_time;
%RSE_mRBK=RSE_mRBK/run_time;
%RSE_AmRBK=RSE_AmRBK/run_time;
%RSE_RBK=RSE_RBK.^2;
%RSE_mRBK=RSE_mRBK.^2;
%RSE_AmRBK=(RSE_AmRBK).^2;

%CPU_RBK=CPU_RBK/run_time;
%CPU_mRBK=CPU_mRBK/run_time;
%CPU_AmRBK=CPU_AmRBK/run_time;


return
%opts.Max_iter=136012;
x_label=1:1:opts.Max_iter+1;
%jiange=ceil(opts.Max_iter/20);
%maker_idx=1:jiange:opts.Max_iter+1;

jiange2=ceil(opts.Max_iter/100);
plot_idx=1:jiange2:opts.Max_iter+1;

jiange3=ceil(length(plot_idx)/20);
maker_idx3=1:jiange3:length(plot_idx);
maker_idx3=[maker_idx3,length(plot_idx)];

figure
semilogy(x_label(plot_idx),RSE_RBK(plot_idx),'b-o','LineWidth',1.2,'MarkerIndices',maker_idx3);
hold on
semilogy(x_label(plot_idx),RSE_mRBK(plot_idx),'g-p','LineWidth',1.2,'MarkerIndices',maker_idx3);
semilogy(x_label(plot_idx),RSE_AmRBK(plot_idx),'m-^','LineWidth',1.2,'MarkerIndices',maker_idx3);
legend('RBKU','mRBKU','AmRBKU')
xlabel('Iter')
ylabel('RSE')
txt=title( ['{\tt cod-rna}',',$m=$ ',num2str(m),',$n=$', num2str(n)] );
set(txt, 'Interpreter', 'latex');

figure
semilogy(CPU_RBK(plot_idx),RSE_RBK(plot_idx),'b-o','LineWidth',1.2,'MarkerIndices',maker_idx3);
hold on
semilogy(CPU_mRBK(plot_idx),RSE_mRBK(plot_idx),'g-p','LineWidth',1.2,'MarkerIndices',maker_idx3);
semilogy(CPU_AmRBK(plot_idx),RSE_AmRBK(plot_idx),'m-^','LineWidth',1.2,'MarkerIndices',maker_idx3);
legend('RBKU','mRBKU','AmRBKU')
xlabel('CPU time')
ylabel('RSE')
txt=title( ['{\tt cod-rna}',',$m=$ ',num2str(m),',$n=$', num2str(n)] );
set(txt, 'Interpreter', 'latex');
