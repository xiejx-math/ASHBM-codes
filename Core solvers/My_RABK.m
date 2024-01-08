function [x,Out]=My_RABK(A,b,ell,opts)

% randomized average block Kaczmarz method for solving linear systems
%              Ax=b
%
%
% Input: the coefficent matrix A, the vector b, the momentum parameter beta,
% the number of blocks t and opts.
% For the opts:
% opts.TOL: the stopping rule
% opts.initial: the initial point
% ...
% ...
%
% Output: the approximate solution x and Out
% Out.error: the relative iterative residual the relative iterative error
% \|x^k-x^*\|^2/\|x^k\|^2
% Out.iter: the total number of iteration
% ....
%
% Based on the manuscript:
% [1] Deren Han, Jiaxin Xie. On pseudoinverse-free randomized methods for
% linear systems: Unified framework and acceleration, arXiv:2208.05437
%
% Coded by Jiaxin Xie, Beihang University, xiejx@buaa.edu.cn
%

tic
[m,n]=size(A);

%% setting some parameters
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

if (flag && isfield(opts,'sparsity'))
    sparsity=opts.sparsity;
else
    sparsity=0;
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

%%
RSE(1)=error1;

%%
%%%% a uniform random permutation for both A and b
if (flag && isfield(opts,'permS'))
    S=opts.permS;
    A=A(S,:);
    b=b(S);
else
    S=randperm(m);
    A=A(S,:);
    b=b(S);
end

%% setting the probability
if (flag && isfield(opts,'probset'))
    probset=opts.probset;
else
    probset=0;
end

if probset
    Aarrs=opts.Aarrs;
    barrs=opts.barrs;
    cumsumpro=opts.cumsumpro;
else
    normAfro=norm(A,'fro')^2;
    tau=floor(m/ell);
    blockAnormfro=zeros(tau,1);
    %prob=zeros(tau,1);
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
    end
    prob=blockAnormfro/normAfro;
    cumsumpro=cumsum(prob);
end

%% executing the RABK method
stopc=0;
iter=0;
times(1)=toc;
while ~stopc
    tic
    iter=iter+1;

    %%
    stopsampling=0;
    while ~stopsampling
        l=sum(cumsumpro<rand)+1;
        AindexR=Aarrs{l};
        bindexR=barrs{l};

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

    %% time count
    times(iter+1)=times(iter)+toc;

end
%% setting Output
Out.error=RSE;
Out.iter=iter;
Out.times=times;
end

