function [x,Out]=My_mRABK(A,b,beta,ell,opts)

% randomized average block Kaczmarz method with momentum for solving
% linear systems
%              Ax=b
%
%
% Input: the coefficent matrix A, the vector b, the momentum parameter
% beta,
% the number of blocks ell and opts.
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
% [2] Du K, Si W T, Sun X H. Randomized extended average block Kaczmarz for
% solving least squares. SIAM Journal on Scientific Computing, 2020,
% 42(6): A3541-A3559.
%
% Coded by Jiaxin Xie, Beihang University, xiejx@buaa.edu.cn
%


[m,n]=size(A);

%% starting count the wall-clock time
tic

%% setting some parameters
flag=exist('opts');


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

%% setting the stepsize alpha and the probability
% Based on [Du K, Si W T, Sun X H. Randomized extended average block Kaczmarz for solving least squares.
% SIAM Journal on Scientific Computing, 2020, 42(6): A3541-A3559.]

if (flag && isfield(opts,'sparsity'))
    sparsity=opts.sparsity;
else
    sparsity=0;
end

%%%%%%%%%%%%%%%%%%

if (flag && isfield(opts,'probset'))
    probset=opts.probset;
else
    probset=0;
end

if probset
    Aarrs=opts.Aarrs;
    barrs=opts.barrs;
    cumsumpro=opts.cumsumpro;
    alpha=opts.alpha;
    blockAnormfro=opts.blockAnormfro;
else
    tau=floor(m/ell);
    blockAnormfro=zeros(tau,1);
    if sparsity
        %B=full(A);
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
            % if normest can estimate the ell_2 norm, then we can use this line
            % beta_max=max(beta_max,normest(Aps)^2/blockAnormfro(i));
            % otherwise, we use
            beta_max=max(beta_max,norm(full(Aps))^2/blockAnormfro(i));
        end
        prob=blockAnormfro/normAfro;
        alpha=1/beta_max;
        cumsumpro=cumsum(prob);
    else
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
            blockAnormfro(i)=norm(A(ps,:),'fro')^2;
            beta_max=max(beta_max,norm(Aps)^2/blockAnormfro(i));
        end
        prob=blockAnormfro/normAfro;
        alpha=1/beta_max;
        cumsumpro=cumsum(prob);
    end
end

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

%%
RSE(1)=error1;


%% executing the mRABK method
stopc=0;
iter=0;
times(1)=toc;
while ~stopc
    tic
    iter=iter+1;
    %% probility

    l=sum(cumsumpro<rand)+1;

    %% set some parameter for updating x

    %     if l==tau
    %         indexR=((l-1)*ell+1):1:m;
    %     else
    %         indexR=((l-1)*ell+1):1:(l*ell);
    %     end
    %     AindexR=A(indexR,:);
    %     bindexR=b(indexR);
    AindexR=Aarrs{l};
    bindexR=barrs{l};
    Axb=AindexR*x-bindexR;
    dk=AindexR'*Axb/blockAnormfro(l);


    %% update x
    xoold=xold;
    xold=x;
    x=x-alpha*dk+beta*(x-xoold);


    %% stopping rule
    if strategy
        error1=norm(x-xstar)^2/normxstar;
        RSE(iter+1)=error1;
        if error1<TOL  || iter>=Max_iter
            stopc=1;
        end
    else
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
%Out.beta_max=beta_max;
Out.iter=iter;
Out.times=times;
end

