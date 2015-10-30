__author__ = 'V_AD'
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from numpy import * 
from datetime import *
from scipy.stats import norm
from scipy.io import *

def cox (nn,maxi, target,tsp,delta):
    whole= datetime.now ()
    p = nn-1
    if p == 1 :
        gamma0 = 0.95
    else:
        gamma0 = 1 - (1-0.05)/ (p*(p-1))
    if gamma0 < 0.95:
        gamma0 = 0.95
    pval = 1 - gamma0
    tol = 0.0001*ones((p))
    flag = 1 
    tspa = target
    isi = target[1:] - target [:len(target)-1]
    v = zeros([p,len(tspa)])
    v1 = zeros ([p,len(tspa)])
    la = []
    
    for i in range (0,p):
        index = where((tspa-delta[i])>0)[0]
        k = min(index)
        start = tspa[k] - delta [i]
        isia = append(start,isi[k:])
        la = append(la,len(isia))
        tspam = cumsum(isia)
        v[i,0:la[i]] = isia [0:la[i]]
        v1[i,0:la[i]]= tspam [0:la[i]]
    
    laf = min (la)
    isiat = v [0:p,0:laf]
    tspamt = v1 [0:p, 0:laf]
    b = zeros(p)
    tspz = append(b,tsp )
    tspz = reshape(tspz, (maxi+1,p))
    inda = zeros_like(isiat)
    a = zeros_like(isiat)
    for i in range (0,p):
        inda [i,:] = sort (isiat[i,:])
        atmp = [[ii for (v, ii) in sorted((v, ii) for (ii, v) in enumerate(isiat[i]))]]
        a[i,:] = array(atmp)
        
   
    mod = SourceModule("""
    #include <stdio.h>
    #include <math.h>
    __global__ void z_function(float *tspamt, float *inda, float *a, float *isiat, float *tspz,  float *z, int *p_d , int *maxi_d)
    {
    float gm = 0.0955 ;
    float alphas = 10 ; 
    float alphar = 0.1 ; 
    float t1;
    int m = threadIdx.y + threadIdx.x * blockDim.y;
    int i = blockIdx.y;
    int j = blockIdx.x;
    int maxi = (int) maxi_d + 1;
    int p = (int)p_d +1; 
  
    if (i>=j)
        {   
        int temp = a[m*gridDim.y+i];
        int temp2 = a[m*gridDim.y+j];
        
        int index = 0 ;
        t1 = tspamt [m*gridDim.y+temp] - isiat [m*gridDim.y+temp] + isiat [m*gridDim.y+temp2] ;
       
       for (int k = m; k < p*maxi ;k+=p)
        {  
                if (tspz [k] < t1 && tspz [k] != -1)
            { 
             if (index < k) 
             { 
               index= k ;
             }     
             }  
        }
        float bwt;
        bwt = t1 - tspz [index];
        //float A = exp(-bwt/alphas)-exp(-bwt/alphar);
        //float B = A/ (alphas-alphar);
        //float C = (1/gm) * B;
        //z[gridDim.y*gridDim.y*m + gridDim.y*i + j] = C;
        z[gridDim.y*gridDim.y*m + gridDim.y*i + j] = (1/gm)*((exp(-bwt/alphas)-exp(-bwt/alphar))/(alphas-alphar));
    
    }
    }
    """)
    
    
    mod2 = SourceModule("""
    #include <stdio.h>
    #include <math.h>
    __global__ void hess(float *z2,float *ssum_d,float *sumte_d ,int laf,int p, float *vi)
    {
    int m = threadIdx.x ;
    int n = blockIdx.x ;
     
    float temp1 = 0;
    float temp2 = 0;
    float temp3 = 0; 
    float part1 = 0;
    float part2 = 0;
    float part3 = 0;
    float part4 = 0;
    
    for (int j = 0; j<laf ;j++)
    { for (int i = j; i<laf*laf; i += laf )
    {
    temp1 +=   z2[m*laf*laf + i] * z2[n*laf*laf + i] * ssum_d[i];
    temp2 += z2[m*laf*laf + i] * ssum_d[i];
    temp3 += z2[n*laf*laf + i] * ssum_d[i];
    
    }
    part1 += temp1/sumte_d[j];
    part2 += temp2 ;
    part3 += temp3 ; 
    part4 += (temp2*temp3)/ (sumte_d[j]*sumte_d[j]);
    temp1 = 0;   
    temp2 = 0;
    temp3 = 0;
    }
    vi[threadIdx.x * gridDim.x + blockIdx.x] = part1-part4; 
    //if (threadIdx.x == 0 && blockIdx.x == 0 )
     //{ 
   // printf ("%f ", part1);
    //}
    }
    
    """)
    
    func = mod.get_function("z_function")
    tspamt =tspamt.astype(float32) 
    inda = inda.astype(float32)
    a = a.astype(float32)
    isiat = isiat.astype(float32)
    tspz = tspz.astype(float32)
    b = zeros((p,laf,laf))
    z = b.astype(float32)

    tspamt_d = tspamt
    inda_d = inda
    a_d = a 
    isiat_d = isiat
    tspz_d = tspz
    p_d = p-1 
    maxi_d = maxi  
    start = datetime.now()
    
    func(cuda.InOut(tspamt_d),cuda.InOut(inda_d),cuda.InOut(a_d),cuda.InOut(isiat_d),cuda.InOut(tspz_d),cuda.InOut(z),int64(p_d),int32(maxi_d),block = (p,1,1), grid = (int_(laf),int_(laf)))
    end = datetime.now()
    ztime= end-start
    print(ztime)
    bet = 0.2*ones(p)
    landa = 1 ; 
    for i in range (0,100):
        scc = zeros_like(z) ;
        for l in range (0,p):
            scc [l,:,:] = bet[l] * z[l,:,:]
        ssum = zeros((laf,laf))
        for g in range (0,p):
            ssum = ssum + scc[g,:,:]
        sumte = sum(tril(exp(ssum)),axis=0)
        
        score = zeros((p))
        for n in range (0,p):
            temp = sum(divide(sum(tril(multiply(z[n,:,:],exp(ssum))),axis = 0),sumte))
            score[n] = trace(z[n,:,:])-temp
        vi = zeros ((p,p));
        vi =vi.astype(float32) 
        laf_d = laf.astype(int32)
        z2 = z.astype(float32)
        func2 = mod2.get_function("hess")
        ssum_d = exp(ssum)
        ssum_d= ssum_d.astype(float32)
        sumte_d = sumte.astype(float32)
        func2(cuda.InOut(float32(z2)),cuda.InOut(ssum_d),cuda.InOut(sumte_d),int32(laf_d),int32(p),cuda.InOut(vi),block = (p,1,1) ,grid = (p,1,1))
        dot_temp = dot(vi.T,vi)

        estimate = bet + reshape(dot(linalg.inv(vi),reshape(score, (p,1))),(1,p))[0]
        
        if i == 0:
            initial_score = zeros_like(score)
        if i > 1:
            if linalg.norm(score)<linalg.norm(initial_score):
                landa = landa/2
            else:
                landa = landa*2
        initial_score = score
        dif_temp = abs(bet-estimate)
        if ((dif_temp< tol).all()):
            bet_result = estimate
            flag = 0
            break
        bet = estimate
    if (flag==1):
        bet_result = 100000
        betahat = 1000000
        betaci = [1000000,1000000]
    else:
        betahat = bet_result
    x = norm.ppf(1-pval/2)
    nx = [-x,x]
    betaci = zeros((p,2))
    for i in range (0,p):
        betaci[i,0] = betahat[i] + nx[0] / sqrt(vi[i,i])
        betaci[i,1] = betahat[i] + nx[1] / sqrt(vi[i,i])
    whole_end = datetime.now()-whole
    print ("whols is: ", whole_end)
    return (betahat, betaci,ztime) 

