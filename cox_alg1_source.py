__author__ = 'V_AD'
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from numpy import * 
from scipy.stats import norm
from scipy.io import *

def cox (nn,maxi, target,tsp,delta):
    '''The main cox method'''
    p = nn-1
    if p == 1 :
        gamma0 = 0.95
    else:
        gamma0 = 1 - (1-0.05)/ (p*(p-1))
    if gamma0 < 0.95:
        gamma0 = 0.95
    #: Doc comment for instance attribute qux.
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
        __global__ void z_function(float *tspamt, float *a, float *isiat, float *tspz,  float *z, long p , int maxi)
        {
        int m = threadIdx.x ;
        int i = blockIdx.y;
        int j = blockIdx.x;
        if (i>=j)
            {
            float gm = 0.0955 ;
            float t1;

            float alphas = 10 ;
            float alphar = 0.1 ;
            int temp = a[m*gridDim.y+i];
            int temp2 = a[m*gridDim.y+j];
            int index = 0 ;
            t1 = tspamt [m*gridDim.y+temp] - isiat [m*gridDim.y+temp] + isiat [m*gridDim.y+temp2] ;
           // if ( i==2  && m == 0){
         //printf("%f ",t1 );
       // }
            for (int k = m; k < p*maxi ;k+=p)
            {
                    if (tspz [k] < t1 && tspz [k] != -1 && index < k)
                {
                   index= k ;
                 }
            }
            float bwt;
            bwt = t1 - tspz [index];
            z[gridDim.y*gridDim.y*m + gridDim.y*i + j] = (1/gm)*((exp(-bwt/alphas)-exp(-bwt/alphar))/(alphas-alphar));
      }
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
    temp1 = zeros([p,p,laf]).astype(float32)
    (free,total)=cuda.mem_get_info()
    assert z.nbytes + temp1.nbytes < free, "The GPU memory is not sufficient for this amount of data."
    tspamt_d = tspamt
    inda_d = inda
    a_d = a
    isiat_d = isiat
    maxi_d = maxi.item()

    func(cuda.InOut(tspamt_d),  cuda.InOut(a_d), cuda.InOut(isiat_d), cuda.InOut(tspz),cuda.InOut(z), int_(p), int_(maxi_d), block=(p, 1, 1), grid=(int_(laf), int_(laf)))

    bet = 0.2*ones(p)
    landa = 1.
    mod2 = SourceModule("""
        #include <stdio.h>
        #include <math.h>
        __global__ void temp1_calculator(float *z2,float *ssum_d,float *sumte_d ,int laf, float *temp1)
        {
        int m = threadIdx.x ;
        int j = blockIdx.x ;
        int k = blockIdx.y;
       //  if(j==0 && k ==0){
        //  printf("Number: %d out of %d       ",m,blockDim.x);
         // }

        float t1=0;
         for (int i = k; i<laf*laf; i += laf )
        {
        t1 += z2[m*laf*laf + i] * z2[j*laf*laf + i] * ssum_d[i];
        }

        temp1[m*gridDim.x*gridDim.y + j*gridDim.y + k] = t1;
        }
        """)
    func2 = mod2.get_function("temp1_calculator")
    mod3 = SourceModule("""
        #include <stdio.h>
        #include <math.h>
        __global__ void temp2_calculator(float *z2,float *ssum_d,float *sumte_d ,int laf,float *temp2)
        {
        int m = threadIdx.x ;
        int j = blockIdx.x ;
        int k = blockIdx.y;

        float t2=0;
         for (int i = k; i<laf*laf; i += laf )
        {
        t2 += z2[m*laf*laf + i] * ssum_d[i];
        }

        temp2[m*gridDim.x*gridDim.y + j*gridDim.y + k] = t2;
        }
        """)
    func3 = mod3.get_function("temp2_calculator")
    mod4 = SourceModule("""
        #include <stdio.h>
        #include <math.h>
        __global__ void temp3_calculator(float *z2,float *ssum_d,float *sumte_d ,int laf,float *temp3)
        {
        int m = threadIdx.x ;
        int j = blockIdx.x ;
        int k = blockIdx.y;

        float t3=0;
         for (int i = k; i<laf*laf; i += laf )
        {
        t3 += z2[j*laf*laf + i] * ssum_d[i];
        }

        temp3[m*gridDim.x*gridDim.y + j*gridDim.y + k] = t3;
        }
        """)
    func4 = mod4.get_function("temp3_calculator")
    mod5 = SourceModule("""
        #include <stdio.h>
        #include <math.h>
        __global__ void hessian(float *z2,float *ssum_d,float *sumte_d ,int laf, float *vi,  float *temp1, float *temp2, float *temp3)
        {
         int m = threadIdx.x ;
         int n = blockIdx.x ;
         int b = gridDim.x;


         float part1 = 0;
         float part2 = 0;
         for (int j = 0; j<laf ;j++)
            {
            part1 += temp1[m*b*laf+n*laf+j]/sumte_d[j];
            part2 += (temp2[m*b*laf+n*laf+j]*temp3[m*b*laf+n*laf+j])/ (sumte_d[j]*sumte_d[j]);
            }
            vi[m*b+n] = part1-part2;
        }
        """)
    func5 = mod5.get_function("hessian")
    # mod6 = SourceModule("""
    # #include <stdio.h>
    # #include <math.h>
    # __global__ void temp_calculator(float *z2,float *ssum_d,float *sumte_d ,int laf, float *temp1, float *temp2, float *temp3)
    # {
    # int m = threadIdx.x;
    # int j = blockIdx.x ;
    # int k = blockIdx.y ;
    #
    # float t1=0;
    # float t2=0;
    # float t3=0;
    #  for (int i = m; i<laf*laf; i += laf )
    # {
    # t1 += z2[j*laf*laf + i] * z2[k*laf*laf + i] * ssum_d[i];
    # t2 += z2[j*laf*laf + i] * ssum_d[i];
    # t3 += z2[k*laf*laf + i] * ssum_d[i];
    # }
    # temp1[j*blockDim.x * gridDim.y  + k*blockDim.x + m] =t1;
    # temp2[j*blockDim.x * gridDim.y  + k*blockDim.x + m] =t2;
    # temp3[j*blockDim.x * gridDim.y  + k*blockDim.x + m] =t3;
    # }
    # """)
    # func6 = mod6.get_function("temp_calculator")

    for i in range (0,100):
        scc = zeros_like(z)
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
        vi = zeros ((p,p))
        vi =vi.astype(float32) 
        laf_d = laf.astype(int32)
        # z2 = z.astype(float32)
        # func2 = mod2.get_function("hess")
        ssum_d = exp(ssum)
        ssum_d= ssum_d.astype(float32)
        sumte_d = sumte.astype(float32)
        # temp1 = zeros([p,p,laf]).astype(float32)
        temp2 = zeros([p,p,laf]).astype(float32)
        temp3 = zeros([p,p,laf]).astype(float32)
        # func3(cuda.InOut(z),cuda.InOut(ssum_d),cuda.InOut(sumte_d),int32(laf_d),cuda.InOut(temp1),cuda.InOut(temp2),cuda.InOut(temp3),block = (p,1,1) ,grid = (p,int_(laf),1))
        func2(cuda.InOut(z),cuda.InOut(ssum_d),cuda.InOut(sumte_d),int32(laf_d),cuda.InOut(temp1),block = (p,1,1) ,grid = (p,int_(laf),1))
        func3(cuda.InOut(z),cuda.InOut(ssum_d),cuda.InOut(sumte_d),int32(laf_d),cuda.InOut(temp2),block = (p,1,1) ,grid = (p,int_(laf),1))
        func4(cuda.InOut(z),cuda.InOut(ssum_d),cuda.InOut(sumte_d),int32(laf_d),cuda.InOut(temp3),block = (p,1,1) ,grid = (p,int_(laf),1))

        func5(cuda.InOut(z),cuda.InOut(ssum_d),cuda.InOut(sumte_d),int32(laf_d),cuda.InOut(vi),cuda.InOut(temp1),cuda.InOut(temp2),cuda.InOut(temp3),block = (p,1,1) ,grid = (p,1,1))
        # func2(cuda.InOut(z2),cuda.InOut(ssum_d),cuda.InOut(sumte_d),int32(laf_d),cuda.InOut(vi),block = (p,1,1) ,grid = (p,1,1))
        # dot_temp = dot(vi.T,vi)
        # estimate = bet + dot(dot(linalg.inv(dot_temp + landa * diag(diag (dot_temp))), vi.T) , score)
        estimate = bet + reshape(dot(linalg.inv(vi),reshape(score, (p,1))),(1,p))[0]

        # if i == 0:
        #     initial_score = zeros_like(score)
        # if i > 1:
        #     if linalg.norm(score)<linalg.norm(initial_score):
        #         landa = landa/2
        #     else:
        #         landa = landa*2
        # initial_score = score
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
    return (betahat, betaci)

