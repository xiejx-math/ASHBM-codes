% This Matlab file is used to compare AmRABK, pinv, and lsqminnorm

close all;
clear;

n=100;
r=n; % the rank of the coefficient matrix, which shound be full-rank
kappa=10; %Desired condition number
mi=1000; % the initial setting for the number of rows
testsize=[1:1:10]; % used for updating the number of rows
run_time=5;  % average times
p=30;% size of the block % here we set p=30,  we can set it as other values,
% for example p=20

opts.TOL=3*10^(-30);% change the tolence if the size of problem change

%% some vectors are used to store the desired numerical results
lstime=zeros(length(testsize),1);
pinvtime=zeros(length(testsize),1);
mRKtime=zeros(length(testsize),1);
AmRABKtime=zeros(length(testsize),1);

CPU_LS=zeros(run_time,length(testsize));
CPU_pinv=zeros(run_time,length(testsize));
CPU_AmRABK=zeros(run_time,length(testsize));


for ii=1:length(testsize)
    m=mi*testsize(ii);
    for jj=1:run_time
        %% generated the matrix A
        [U,~]=qr(randn(m, r), 0);
        [V,~]=qr(randn(n, r), 0);
        D = diag(1+(kappa-1).*rand(r, 1));
        A=U*D*V';
        clear U V D

        %% generated the right-hand vector b
        x=randn(n,1);
        b=A*x;

        %% Matlab function solvers
        tic
        xLS=lsqminnorm(A,b);
        MTls_CPU=toc;

        tic
        xpinv=pinv(A)*b;
        MTpinv_CPU=toc;

        %% parameter setup
        opts.xstar=x;
        
        opts.TOL1=eps^2;
        opts.Max_iter=100000;

     

        %% AmRABK
        [xAmRABK,OutAmRABK]=My_AmRABK(A,b,p,opts);

        %%
        CPU_AmRABK(jj,ii)=OutAmRABK.times(end);
        CPU_LS(jj,ii)=MTls_CPU;
        CPU_pinv(jj,ii)=MTpinv_CPU;

        %%
        fprintf('lsqminnorm: %8e, pinv: %8e,AmRABK: %8e\n',norm(x-xLS),norm(x-xpinv),norm(xAmRABK-x))
        %fprintf('Matlab function pinv: %8e\n',norm(x-xpinv))
        %fprintf('RABK: %8e\n',norm(xRABK-x))
        %fprintf('AmRABK: %8e\n',norm(xAmRABK-x))
    end
    fprintf('Done %d\n',ii)
end


%%
xlabel_i=mi*testsize;
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
%display_names = {'RABK','AmRABK','pinv','lsqminnorm'};
%arrsIter = {CPU_RABK', CPU_AmRABK',CPU_pinv',CPU_LS'};

display_names = {'pinv','lsqminnorm','AmRABK'};
arrsIter = {CPU_pinv',CPU_LS',CPU_AmRABK'};

num_methods = length(arrsIter);

%line_colors = {'blue','green','magenta'};
%minmax_colors = { lightblue, lightgreen,lightmagenta};
%quant_colors = { mediumblue,mediumgreen,mediummagenta};

line_colors = {'black','green','red'};
minmax_colors = { lightgray, lightgreen,lightred};
quant_colors = { mediumgray,mediumgreen,mediumred};

%line_colors = {'black', 'green', 'red', 'blue'};
%minmax_colors = {lightgray, lightgreen, lightred, lightblue};
%quant_colors = {mediumgray, mediumgreen, mediumred, mediumblue};

display_legend = true;
max_val_in_plot = 1e5;

%%
[x_arrays_iter, quantiles_iter] = compute_and_plot_Iter(num_iter_array, arrsIter, ...
    num_methods, line_colors, display_names, ...
    minmax_colors, quant_colors, display_legend, max_val_in_plot);
%title('Relative residuals over iterations, with quantiles')
%legend('RBKU','mRBKU','AmRBKU','Location','northwest')
%legend('RABK','AmRABK','{\tt pinv}','{\tt lsqminnorm}','Location','northwest','Interpreter', 'latex')
ylabel('CPU time')
xlabel('Number of rows (m)')
txt=title( ['{\tt randn}',',$n=$ ',num2str(n),',$\kappa=$',num2str(kappa)]);
%txt=title( ['{\tt randn}',',$n=$ ',num2str(n)]);
set(txt, 'Interpreter', 'latex');



%%
% figure
% plot(xlabel_i,median(CPU_AmRABK),'b-d','LineWidth',1.2)
% hold on
% plot(xlabel_i,median(CPU_pinv),'g-o','LineWidth',1.2)
% plot(xlabel_i,median(CPU_LS),'m-s','LineWidth',1.2)
% xlabel('Number of rows $(m)$','Interpreter', 'latex')
% ylabel('CPU time')
% legend('AmRABK','{\tt pinv}','{\tt lsqminnorm}','Location','northwest','Interpreter', 'latex')
% txt=title( ['$n=$ ',num2str(n),',$\kappa=$ ',num2str(kappa)]);
% set(txt, 'Interpreter', 'latex');


