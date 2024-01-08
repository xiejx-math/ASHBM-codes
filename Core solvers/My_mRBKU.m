function [x,Out]=My_mRBKU(A,b,alpha,omega,ell,opts)

% randomized block Kaczmarz method with momentum for solving linear systems
%              Ax=b
%
% we use the uniform sampling strategy  (mRBKU)
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
% Based on the manuscript: 
% [1] Deren Han, Jiaxin Xie. On pseudoinverse-free randomized methods for 
% linear systems: Unified framework and acceleration, arXiv:2208.05437
%
% Coded by Jiaxin Xie, Beihang University, xiejx@buaa.edu.cn

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

%%%% setting the initial point
if (flag && isfield(opts,'initial'))
    initialx=opts.initial;
else
    initialx=zeros(n,1);
end
x=initialx;
xold=x;

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

%%%% seting the parameter alpha
FnormA=norm(A,'fro');
newalpha=(alpha*m)/(ell*FnormA^2);

%% executing the mRBKU method
stopc=0;
iter=0;
times(1)=toc;
while ~stopc
    tic
    iter=iter+1;

    indexR=randperm(m,ell);

    %% update x
    xoold=xold;
    xold=x;
    AindexR=A(indexR,:);
    bindexR=b(indexR);
    Axb=AindexR*x-bindexR;
    dk=AindexR'*Axb;
    x=x-newalpha*dk+omega*(x-xoold);
    

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

