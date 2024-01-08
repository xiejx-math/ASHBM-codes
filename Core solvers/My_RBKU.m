function [x,Out]=My_RBKU(A,b,ell,opts)

% randomized  block Kaczmarz method for solving linear systems
%              Ax=b
%
% we use the uniform sampling strategy (RBKU)
%
%Input: the coefficent matrix A, the vector b and opts
%opts.initial: the initial vector x^0
%opts.TOL: the stopping rule
%.....
%
%Output: the approximate solution x and Out
% Out.error: the relative iterative error \|x^k-x^*\|^2/\|x^k\|^2
% Out.iter: the total number of iteration
% ....
%
% Based on the manuscript: On adaptive stochastic heavy ball momentum for
% solving linear systems, Yun Zeng, Deren Han, Yansheng Su, Jiaxin Xie, arXiv:2305.05482
%
% Coded by Jiaxin Xie, Beihang University, xiejx@buaa.edu.cn
%

tic
[m,n]=size(A);

%% setting some parameter
flag=exist('opts');
%%%% setting the max iteration
if (flag && isfield(opts,'Max_iter'))
    Max_iter=opts.Max_iter;
else
    Max_iter=200000;
end
%%%% setting the tolerance
if (flag && isfield(opts,'TOL'))
    TOL=opts.TOL;
else
    TOL=10^-12;
end

%%%% setting the tolerance for checking the values of S_k(Ax_k-b)
if (flag && isfield(opts,'TOL1'))
    TOL1=opts.TOL1;
else
    TOL1=10^-20;
end

%%%% setting the initial point
if (flag && isfield(opts,'initial'))
    initialx=opts.initial;
else
    initialx=zeros(n,1);
end
x=initialx;


%%%% determining what to use as the stopping rule
if (flag && isfield(opts,'xstar'))
    xstar=opts.xstar;
    if m>=n
        normxstar=norm(xstar)^2;
        error1=norm(xstar-x)^2/normxstar;
        strategy=1;
    else
        strategy=0;
    end
else
    strategy=0;
end

if (flag && isfield(opts,'strategy'))
    strategy=opts.strategy;
    normxstar=norm(xstar)^2;
end

if ~strategy
    normb=norm(b)^2+1;
    error1=norm(A*x-b)^2/normb;
end

RSE(1)=error1;

%%
% diag_D=sqrt(sum(A.^2,2));
% 
% A=A./diag_D;
% b=b./diag_D;
% 
% xstar_x1x2=0;

%% executing the AmRBKU method
stopc=0;
iter=0;
times(1)=toc;
while ~stopc
    tic
    iter=iter+1;

    stopsampling=0;
    while ~stopsampling
        indexR=randperm(m,ell);
        AindexR=A(indexR,:);
        bindexR=b(indexR);
        Axb=AindexR*x-bindexR;
        normAxb=norm(Axb)^2;
        if normAxb>TOL1
            stopsampling=1;
            dk=AindexR'*Axb;
            norm_dk=norm(dk)^2;
        end
    end

    %% update step-size alpha
    alpha=normAxb/norm_dk;

    %% update x
    x=x-alpha*dk;

 %% stopping rule
    if strategy
        error1=norm(x-xstar)^2/normxstar;
        RSE(iter+1)=error1;
        if error1<TOL  || iter>=Max_iter
            stopc=1;
        end
    else
        %%%% Note that we do not us this stopping rule during our test
        error1=norm(A*x-b)^2/normb;
        RSE(iter+1)=error1;
        if  error1<TOL || iter>=Max_iter
            stopc=1;
        end
    end


    times(iter+1)=times(iter)+toc;
end
%% setting Output
Out.error=RSE;
Out.iter=iter;
Out.times=times;
end

