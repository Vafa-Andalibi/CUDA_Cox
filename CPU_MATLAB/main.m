%clear all ; clc; close all ; 
m = 4;
duration = 25;
experiment_type = 'SB'; % neuron based or spike based 
folder = 'E:\';
filename = ['restest1_' num2str(duration)];
global ET;
ET = struct ;
ET.total = 0 ;
ET.coxes_all = 0 ; 
ET.Zs_all = 0;
ET.hessian_all = 0 ;
ET.coxes = zeros(1,m);
ET.Zs = zeros(1,m); 
ET.hessian = zeros (m,1);
ET.hessian_sum = zeros(m,1);

%%%%%%%%%%%%%%%%%%%%% FOR NENGO 
% path = 'C:\Users\andalibi\Local\Matlab Data\';
% dir_list = dir (path);
% [m,~] = size (dir_list) ;
% m = m-2;
% [train, spike_struct] = getFiles(path);

% 
% n = (m*3)-1;
% results = zeros ( m,m) ;
% confidence = zeros ( m,n) ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% path = 'E:\Cloud-Drives\Google Drive\TUT\Work\CUDA_Richiardi\Revision\spike generator\512 neurons\restest1.res';
path = [folder filename '.res'];

spike_struct = res_to_struct (path, m);
t1 = timetic;
tic(t1)
results = zeros ( m,m) ;
n = (m*3)-1;
confidence = zeros ( m,n) ;

ztime = 0;
%%%%%%%%%%%%%%%%%%%%%%% file creator for C
% neuron = 1; 
%    s = ['fid=fopen (''E:\TUT\Seminar Course on Networking\Cox Method Modified\spike generator\spike generator\temp',num2str(neuron),'.txt'',''wt'');'];
%    eval (s)
%    s = ['fprintf(fid, ''%0.0f\n'',spike_struct (',num2str(neuron),').Target);'];
%    eval (s)
%    fclose(fid);
% for q = 2 : length (spike_struct(neuron).Ref)
%    s = ['fid=fopen (''E:\TUT\Seminar Course on Networking\Cox Method Modified\spike generator\spike generator\temp',num2str(q),'.txt'',''wt'');'];
%    eval (s)
%    temp = spike_struct(neuron).Ref;
%    s = ['fprintf(fid, ''%0.0f\n'',temp(:,',num2str(q-1),'));'];
%    eval (s)
%    fclose(fid);
% end
%    
   
   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%load 'C:\Users\andalibi\Local\Matlab Data\restest1.res' -ascii;
% load 'C:\Users\andalibi\Local\Matlab Data\restest1.res' -ascii;
% n=5;
% T=20000;
% a=reshape(restest1,T,n);
% for i = 1:n
%     s = ['train',num2str(i),'= find (a(:,',num2str(i),'));'];
%     eval(s);
%     s = ['train',num2str(i)];
%     if length(eval(s))== 0
%        s = ['train',num2str(i),'= 0;'] ;
%     end
% end

%spike_struct = struct;


% for i = 1:5
%  s= ['spike_struct(i).Target = train',num2str(i)];
%  eval(s)
%  spike_struct(i).Ref = zeros (5,4);
% end
% largest = 0;
% for i = 1:5
%    s = ['if length(train' , num2str(i) , ') > largest ,largest = length(train',num2str(i),');,end;'];
%    eval (s)
% end
% for i = 1: 5
%     s = ['difference = largest - length (train',num2str(i),');'];
%     eval  (s) ;
%     q = zeros ( difference,1);
%  s = ['if length ( train',num2str(i),')<largest, train',num2str(i), ' = [train',num2str(i),'; q];,end'];
%  eval (s) ; 
% end 
% 
% spike_struct(1).Ref = [train2,train3,train4,train5];
% spike_struct(2).Ref = [train1,train3,train4,train5];
% spike_struct(3).Ref = [train1,train2,train4,train5];
% spike_struct(4).Ref = [train1,train2,train3,train5];
% spike_struct(5).Ref = [train1,train2,train3,train4];

% init = zeros (4,1);
% [betas,conf] = Newton_Raphson (spike_struct,1,init,9)

% t1 = timetic;
% tic(t1)
for i = 1:m
    if i == 1 
%     deltas = zeros (1,m-1);
%     ref_temp = spike_struct(i).Ref;
%     for delt = 1:m-1
%      delt_temp = 0;
%      hat_temp = 1;
%      reftemp = ref_temp(:,delt);
%      reftemp(reftemp==0) = [];
%      while hat_temp >0
%         [hat_temp ,ci_temp] = cox (spike_struct(i).Target , reftemp,delt_temp) ;
%         hat_temp
%          delt_temp  = delt_temp +1 ;
%      end
%      deltas(delt) = delt_temp-1;
%     end
    t2 = timetic; 
    tic(t2) 
    [temp1 ,temp2] = cox (spike_struct(i).Target , spike_struct(i).Ref ,i) ;
    ET.coxes(i) = toc(t2);
    results (2:m,1) = temp1;
    confidence(2:m,1:2) = temp2 ;
%     ztime= ztime + zt;
    elseif i == m
     t2 = timetic; 
     tic (t2)
     [temp1,temp2 ] = cox (spike_struct(i).Target , spike_struct(i).Ref ,i)  ;
     ET.coxes(i) = toc(t2);
     results (1:(m-1),m) = temp1 ; 
     confidence(1:(m-1),n-1:n) = temp2 ;
%      ztime = ztime + zt;
    else
        t2 = timetic; 
        tic(t2)
     [temp1,temp2] = cox (spike_struct(i).Target , spike_struct(i).Ref ,i)     ;
     ET.coxes(i) = toc (t2); 
%      ztime = ztime+zt;
     results (1:i-1 , i )  = temp1 (1:i-1,1) ;
     results (i+1:m ,i ) = temp1 (i:end,1) ; 
     ind = 2*(i)+i-2;
     confidence  (1:i-1,ind : ind+1) =  temp2 (1:i-1,:);
     confidence  (i+1:end,ind : ind+1) = temp2 (i:end , : );
    end
disp (['Neuron ' num2str(i) ' completed']);
end
% total = toc(t1)

r = results'
c = confidence'
p = 1 ;
from = [];
to = [];
thickness = [];
for i = 1:m
    for j = 1:m
        if (c(p+1,j) > 0 && c(p,j)>0)||(c(p+1,j) < 0 && c(p,j)<0)
    from = [from; j];
    to = [to;i];
    thickness = [thickness;r(i,j)] ;
        end
    end
     p = p + 3 ; 
end
% to_r = [from, to , thickness]
% save ('C:\g.mat', 'to_r')
% 
% beep
spike_num_avg = 0;
for i = 1:m 
    spike_num_avg = spike_num_avg + length(spike_struct(i).Target);
end
spike_num_avg = spike_num_avg/m
% ztime/m
ET.total = toc(t1);
ET.hessian_sum = sum(ET.hessian,2);
ET.coxes_all = sum(ET.coxes);
ET.Zs_all = sum(ET.Zs);
ET.hessian_all = sum (ET.hessian_sum);
save([folder 'CG_' experiment_type '_' filename(strfind(filename,'_')+1:end) '.mat'],'ET');
folder = 'E:\Data_Cuda\For CPU-GPU\';
to_save = {'nurons' 'duration' 'spikes' 'total' 'coxes' 'Zs' 'hessians'};
if ~exist([folder 'CG_' experiment_type '.xls'] , 'file')
    to_save(end+1,:) = {m duration spike_num_avg ET.total ET.coxes_all ET.Zs_all ET.hessian_all};
    xlswrite([folder 'CG_' experiment_type '.xls'],to_save);
else
    xls_cur = num2cell(xlsread([folder 'CG_' experiment_type '.xls']));
    [rows,~] = size(xls_cur);
    to_save(end+1:end+rows,:) = xls_cur ;
    to_save(end+1,:) = {m duration spike_num_avg ET.total ET.coxes_all ET.Zs_all ET.hessian_all};
    xlswrite([folder 'CG_' experiment_type '.xls'],to_save);
end
beep