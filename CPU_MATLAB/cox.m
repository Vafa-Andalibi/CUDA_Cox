function [betahat betaci]=cox(A,tsp,neuron_idx,delta,tau,alphas,alphar,pval )
global ET; % timelapse variable
%    cox returns the estimate of the parameters of the Cox method which 
%    characterises the strength of influence from the reference spike 
%    trains to the target spike train. 
%   
%    [betahat betaci]=cox(A,tsp,delta,tau,alphas,alphar,pval) 
%    betahat shows the estimated Cox coeffifient
%    betaci shows the 100*(1-pval)confidence intervals of the estimated 
%    Cox coefficient.
%
%    A is the target spike train (consecutive times of spiking). 
% 
%    tsp is a matrix of reference spike trains (each column of the matrix 
%    represents reference spike train and row represents consecutive times 
%    of spiking of the reference spike train). 
%    
%    For example, x1, x2 and x3 are three reference spike trains with times 
%    of spiking x1=[2 5 7 9], x2=[3 6 8 10 12] and x3=[1 4 7 11 13 15]
%    then the tsp will be 
%                        tsp= | 2    3    1 |
%                             | 5    6    4 |
%                             | 7    8    7 |
%                             | 9   10   11 |
%                             | 0   12   13 |
%                             | 0    0   15 |
%
%
%    delta is the vector of time shifts in spike propagation from each 
%    reference spike trains to the target spike train. 
% 
%    tau is a vector which takes into account a history of influences from 
%    each reference spike trains to the target spike train over time 
%    interval (t-tau, t).
%
%    alphas is the vector of characteristic decay time of each postsynaptic 
%    potentials. 
%    
%    alphar is the vector of characteristic rise time of each postsynaptic 
%    potentials. 
%
%    We accept the hypothesis that reference spike train does not influence
%    target spike train A if the confidence interval includes zero.
% tic
if  nargin < 3
    error('stats:regress:TooFewInputs', ...
          'cox requires at least two input arguments.');
    elseif nargin == 3
           [n,p]=size(tsp);
           delta=zeros(1,p);
           tau=zeros(1,p);
           alphas = 10*ones(1,p);
           alphar = .1*ones(1,p);
           pval=.05;
    elseif nargin==4
           [n,p]=size(tsp);
           tau=zeros(1,p);
           alphas = 10*ones(1,p);
           alphar = .1*ones(1,p);
           pval=.05;
    elseif nargin==5
           [n,p]=size(tsp);
           alphas = 10*ones(1,p);
           alphar = .1*ones(1,p);
           pval=.05;
    elseif nargin==6
        alphar = .1*ones(1,p);
           pval=.05;
end;

[n,p]=size(tsp);
tol=0.0001;
flag=1;
tspa=A';
isi=tspa(2:end)-tspa(1:end-1);
v(1:p,1:length(A))=0;
v1(1:p,1:length(A))=0;
la=[];

for i=1:p
    index=find((tspa-delta(i))>0);
    k=min(index);
    start=tspa(k)-delta(i);
    isia=[start isi(k:end)];
    la=[la length(isia)];
    tspam=cumsum(isia);
    v(i,1:la(i))=isia(1:la(i));
    v1(i,1:la(i))=tspam(1:la(i));
end;

laf=min(la);
isiat=v(1:p,1:laf);
tspamt=v1(1:p,1:laf);


for i=1:p
%   t(i)= tsp(:,i);
    s1=['t' int2str(i) '=tsp(:,i);'];
    eval(s1);
end;

for i=1:p
    s = ['tss' int2str(i) '=find(t' int2str(i) '~=0);'];
    eval(s);
    s1=['ts' int2str(i) '=t' int2str(i) '(tss' int2str(i) ');'];
    eval(s1);
end;

for i=1:p
    s1=['tsp' int2str(i) '=[0,ts' int2str(i) '''];'];
    eval(s1);
end;

for h=1:p
    s1=['isit' int2str(h) '=isiat(h,:);'];
    eval(s1);
    s1=['tspat' int2str(h) '=tspamt(h,:);'];
    eval(s1);
end;

for j=1:p
    rand('twister',5555 );
    s=['m' int2str(j) '=randperm(length(isiat));'];
    eval(s);
     s2=['[a' int2str(j) ',inda' int2str(j) ']=sort(isit' int2str(j) '(m' int2str(j) '));'];
     eval(s2);
    s2=['[a2' int2str(j) ',inda2' int2str(j) ']=sort(isit' int2str(j) ');'];
    eval(s2);
end;
%%%%%%% creating the _all varirables 
tspat_all = zeros (p,length(tspat1));
m_all = zeros(p, length(m1));
inda_all = zeros(p,length(inda1));
inda2_all = zeros(p,length(inda1));
isit_all = zeros (p,length(isit1));
tsp_all = cell (1,p);
z_all = cell(1,p);
for i=1:p
    tmp_str = ['tspat_all (i,:) = tspat' num2str(i) ';'];
    eval(tmp_str); 
    tmp_str = ['m_all (i,:) = m' num2str(i) ';'];
    eval(tmp_str); 
    tmp_str = ['inda_all (i,:) = inda' num2str(i) ';'];
    eval(tmp_str); 
    tmp_str = ['inda2_all (i,:) = inda2' num2str(i) ';'];
    eval(tmp_str); 
    tmp_str = ['isit_all (i,:) = isit' num2str(i) ';'];
    eval(tmp_str); 
    tmp_str = ['tsp_all {i} = tsp' num2str(i) ';'];
    eval(tmp_str); 
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% toc
% tic
t3 = timetic; 
tic(t3)
poolobj = gcp('nocreate');
if isempty(poolobj)
   poolobj=parpool(4); 
end
parfor m=1:p
%for i=1:length(isia)
tspat_cur= tspat_all(m,:);
m_cur = m_all(m,:);
inda_cur = inda_all(m,:);
inda2_cur = inda2_all(m,:);
isit_cur = isit_all(m,:);
tsp_cur = tsp_all{m};
z = zeros (laf); 
    for i=1:laf
        for j=1:i
%             t1 = tspat_cur(m_cur(inda_cur(i))) - isit_cur(m_cur(inda_cur(i))) + isit_cur(m_cur(inda_cur(j)));
            t1 = tspat_cur(inda2_cur(i)) - isit_cur(inda2_cur(i)) + isit_cur(inda2_cur(j));
            if tau(m)==0
               k = find (tsp_cur<t1, 1 , 'last');
               bwt = t1 - tsp_cur(k);
                  if alphas(m)==alphar(m)
                     z(i,j)=(bwt/alphas(m))*exp(1-bwt/alphas(m));
                  else
                     tm=log(alphas(m)/alphar(m))/(1/alphar(m)-1/alphas(m));
                     gm=(exp(-tm/alphas(m))-exp(-tm/alphar(m)))/(alphas(m)-alphar(m));
                     z(i,j)=1/gm*(exp(-bwt/alphas(m))-exp(-bwt/alphar(m)))/(alphas(m)-alphar(m));
                  end;
            else
               index = (t1- tau(m))<= tsp_cur & tsp_cur < t1;
               bwt = t1-tsp_cur(index); 
                  if alphas(m)==alphar(m)
                     z(i,j)=sum((bwt/alphas(m)).*exp(1-bwt/alphas(m)));
                  else
                     tm=log(alphas(m)/alphar(m))/(1/alphar(m)-1/alphas(m));
                     gm=(exp(-tm/alphas(m))-exp(-tm/alphar(m)))/(alphas(m)-alphar(m));
                     z(i,j)=sum(1/gm*(exp(-bwt/alphas(m))-exp(-bwt/alphar(m)))/(alphas(m)-alphar(m)));
                 end;   
           end;
       end;
   end;
   z_all{m} = z; 
   
end;
ET.Zs(neuron_idx) = toc(t3);
% toc

bet=.2*ones(1,p)';
scc_all = zeros(p,laf,laf);
landa=1 ;
for i=1:20
%     tic
    for l=1:p
        scc_all(l,:,:)= bet(l)*z_all{l};
    end;
    ssum = squeeze(sum(scc_all));
    sumte=sum(tril(exp(ssum)));

    clear v;
    v=[];
    score = zeros(1,p);
    for n=1:p
        score(n) = trace(z_all{n})-sum(sum(tril(z_all{n}.*exp(ssum)))./sumte);
    end;
    infc = zeros(p); 
%     toc
%     tic
    t4 = timetic;
    tic (t4)
    parfor m=1:p
        for j=1:p
            infc(m,j) = sum(sum(tril(z_all{m}.*z_all{j}.*exp(ssum)))./sumte)-sum(sum(tril(z_all{m}.*exp(ssum))).*sum(tril(z_all{j}.*exp(ssum)))./sumte.^2);
        end;
    end;
    ET.hessian(neuron_idx,i) = toc (t4);
%     toc
%     tic
    dot_tmp = infc'*infc;
    estimate=bet+  (inv(dot_tmp+ landa*diag(diag(dot_tmp))) * infc')*score';
%     estimate=bet+inv(infc)*score';
    if i==1
       initial_score = zeros(size(score));
    end
    if i>2
        if norm(score)<norm(initial_score)
           landa=landa/2 ;
        else
            landa=landa*2;
        end 
    end
    initial_score = score;
    if( all(abs(bet-estimate)<tol))
       bet_result=estimate;
       flag=0;
    break;
    end;
    bet=estimate;
%     toc
end;
% tic
% ET.hessian(:,1) = [];
if(flag==1)
    bet_result=100000;
    disp(' Iteration do not converge! bet_result and betahat are not defined')
    betahat=10000000;
    betaci=[1000000 10000000]
end;  

if (flag==0)
   betahat=bet_result;
   nx = norminv([pval/2 1-pval/2],0,1);
   s = zeros (p,2);
   for i=1:p
       s(i,1) = betahat(i)+nx(1)/(sqrt(infc(i,i)));
       s(i,2) = betahat(i)+nx(2)/(sqrt(infc(i,i)));
   end;
   betaci=s;
end;
% toc