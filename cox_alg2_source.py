__author__ = 'V_AD'
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from numpy import * 
from scipy.stats import norm
from scipy.io import *

class Cox_Method(object) :
    def __init__(self, nn,maxi, target,tsp,delta,gm=0.0955, alphas=10, alphar=0.1):
        self.gm = gm
        self.alphas = alphas
        self.alphar = alphar
        self.nn = nn
        self.maxi = maxi
        self.target = target
        self.tsp = tsp
        self.delta = delta
        self.p = nn-1
        self.gamma0 = 0.95 if self.p==1 else 1 - (1-0.05)/ (self.p*(self.p-1))
        self.gamma0 = 0.95 if self.gamma0<0.95 else self.gamma0
        self.pval = 1 - self.gamma0
        self.tol = 0.0001 * ones((self.p))
        self.tspa = target
        self.isi = target[1:] - target[:len(target) - 1]  #: Interspike intervals.
        self.v = zeros([self.p, len(self.tspa)])
        self.v1 = zeros([self.p, len(self.tspa)])
        self.la = []
        for i in range(0, self.p):
            index = where((self.tspa - delta[i]) > 0)[0]
            k = min(index)
            start = self.tspa[k] - delta[i]
            self.isia = append(start, self.isi[k:])
            self.la = append(self.la, len(self.isia))
            self.tspam = cumsum(self.isia)
            self.v[i, 0:self.la[i]] = self.isia[0:self.la[i]]
            self.v1[i, 0:self.la[i]] = self.tspam[0:self.la[i]]

        self.laf = min(self.la)
        self.isiat = self.v[0:self.p, 0:self.laf]
        self.tspamt = self.v1[0:self.p, 0:self.laf]
        self.b = zeros(self.p)
        self.tspz = append(self.b, tsp)
        self.tspz = reshape(self.tspz, (maxi + 1, self.p))
        self.inda = zeros_like(self.isiat)
        self.a = zeros_like(self.isiat)
        for i in range(0, self.p):
            self.inda[i, :] = sort(self.isiat[i, :])
            self.atmp = [[ii for (self.v, ii) in sorted((self.v, ii) for (ii, self.v) in enumerate(self.isiat[i]))]]
            self.a[i, :] = array(self.atmp)
        self.tspamt = self.tspamt.astype(float32)
        self.inda = self.inda.astype(float32)
        self.a = self.a.astype(float32)
        self.isiat = self.isiat.astype(float32)
        self.tspz = self.tspz.astype(float32)
        self.b = zeros((self.p, self.laf, self.laf))
        self.z = self.b.astype(float32)
        self.tspamt_d = self.tspamt
        self.inda_d = self.inda
        self.a_d = self.a
        self.isiat_d = self.isiat
        self.maxi_d = self.maxi.item()
        self.laf_s = int_(sqrt(self.laf) + 1)
        self.bet = 0.2 * ones(self.p)
        self.landa = 1.
    def alg1 (self):
        '''
        :param nn: Number of neurons or cluster of neurons in the neural network.
        :param maxi: The length of the longest reference spike train.
        :param target: The array of target spike train.
        :param tsp: The matrix of reference spike trains. The size of this matrix is "maxi by nn-1" where each column corresponds\
        to a reference spike train and in case the length of the spike train is shorter than "maxi", it should be padded by zeros.
        :param delta: The array of propagation delays, if known, from the references to the target neuron.
        :return: The array of betahats for each reference spike train as well as the confidence interval corresponding to each betahat value.
        '''

        self.mod_z1 = SourceModule("""
        #include <stdio.h>
        #include <math.h>
        __global__ void z_function(float *tspamt, float *a, float *isiat, float *tspz,  float *z, long p , int maxi, float gm, float alphas, float alphar)
        {
        int j = threadIdx.x;
        int m = blockIdx.x;
        int i = blockIdx.y;

        if (i>=j)
            {
         //   float gm = 0.0955 ;
            float t1;
         //   float alphas = 10 ;
           // float alphar = 0.1 ;
            int temp = a[m*gridDim.y+i];
            int part2 = a[m*gridDim.y+j];
            int index = 0 ;
            t1 = tspamt [m*gridDim.y+temp] - isiat [m*gridDim.y+temp] + isiat [m*gridDim.y+part2] ;
            for (int k = m; k < p*maxi ;k+=p)
            {
                    if (tspz [k] < t1 && tspz [k] != -1 && index < k)
                {
                   index= k ;
                 }
            }
            float bwt;
            bwt = t1 - tspz [index];
            z[blockDim.x*gridDim.y*blockIdx.x + gridDim.y*i + j] = (1/gm)*((exp(-bwt/alphas)-exp(-bwt/alphar))/(alphas-alphar));
        }
        }
        """)

        z1_func = self.mod_z1.get_function("z_function")
        z1_func(cuda.InOut(self.tspamt_d),  cuda.InOut(self.a_d), cuda.InOut(self.isiat_d), cuda.InOut(self.tspz),cuda.InOut(self.z), \
             int_(self.p), int_(self.maxi_d),float32(self.gm),float32(self.alphas),float32(self.alphar), block=(int_(self.laf),1, 1), grid=(self.p, int_(self.laf)))
        return self.hessian()
    def alg2(self):
        self.mod_z2 = SourceModule("""
            #include <stdio.h>
            #include <math.h>
            __global__ void z_function(float *tspamt, float *a, float *isiat, float *tspz,  float *z, long p , int maxi,float gm, float alphas, float alphar)
            {
            int m = threadIdx.x ;
            int i = blockIdx.y;
            int j = blockIdx.x;
            if (i>=j)
                {
              //  float gm = 0.0955 ;
                float t1;

                //float alphas = 10 ;
               // float alphar = 0.1 ;
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

        z2_func = self.mod_z2.get_function("z_function")
        z2_func(cuda.InOut(self.tspamt_d), cuda.InOut(self.a_d), cuda.InOut(self.isiat_d), cuda.InOut(self.tspz), cuda.InOut(self.z), int_(self.p),
             int_(self.maxi_d),float32(self.gm),float32(self.alphas),float32(self.alphar), block=(self.p, 1, 1), grid=(int_(self.laf), int_(self.laf)))
        return self.hessian()
    def hessian(self):
        self.mod_p1 = SourceModule("""
            #include <stdio.h>
            #include <math.h>
            __global__ void part1_calculator(float *z2,float *ssum_d,float *sumte_d ,int laf, float *part1)
            {

            int k = threadIdx.x ;
            int j = blockIdx.x ;
            int m = blockIdx.y;
           //  if(j==0 && m ==0){
            ///  printf("Number: %d out of %d       ",k,blockDim.x);
             // }

            float t1=0;
             for (int i = k; i<laf*laf; i += laf )
            {
            t1 += z2[m*laf*laf + i] * z2[j*laf*laf + i] * ssum_d[i];
            }
            //if (j==0 ){printf("%d   " , m*gridDim.x*gridDim.y + j*gridDim.y + k);}
            part1[m*gridDim.x*blockDim.x + j*blockDim.x + k] = t1;
            }
            """)
        part1_func = self.mod_p1.get_function("part1_calculator")
        self.mod_p2 = SourceModule("""
            #include <stdio.h>
            #include <math.h>
            __global__ void part2_calculator(float *z2,float *ssum_d,float *sumte_d ,int laf,float *part2)
            {
            int k = threadIdx.x ;
            int j = blockIdx.x ;
            int m = blockIdx.y;

            float t2=0;
             for (int i = k; i<laf*laf; i += laf )
            {
            t2 += z2[m*laf*laf + i] * ssum_d[i];
            }

            part2[m*gridDim.x*blockDim.x + j*blockDim.x + k] = t2;
            }
            """)
        part2_func = self.mod_p2.get_function("part2_calculator")
        self.mod_p3 = SourceModule("""
            #include <stdio.h>
            #include <math.h>
            __global__ void part3_calculator(float *z2,float *ssum_d,float *sumte_d ,int laf,float *part3)
            {
            int k = threadIdx.x ;
            int j = blockIdx.x ;
            int m = blockIdx.y;

            float t3=0;
             for (int i = k; i<laf*laf; i += laf )
            {
            t3 += z2[j*laf*laf + i] * ssum_d[i];
            }

            part3[m*gridDim.x*blockDim.x + j*blockDim.x + k] = t3;
            }
            """)
        part3_func = self.mod_p3.get_function("part3_calculator")
        self.mod_h = SourceModule("""
            #include <stdio.h>
            #include <math.h>
            __global__ void hessian(float *z2,float *ssum_d,float *sumte_d ,int laf, float *vi,  float *part1, float *part2, float *part3)
            {
             int m = threadIdx.x ;
             int n = blockIdx.x ;
             int b = gridDim.x;


             float section1 = 0;
             float section2 = 0;
             for (int j = 0; j<laf ;j++)
                {
                section1 += part1[m*b*laf+n*laf+j]/sumte_d[j];
                section2 += (part2[m*b*laf+n*laf+j]*part3[m*b*laf+n*laf+j])/ (sumte_d[j]*sumte_d[j]);
                }
                vi[m*b+n] = section1-section2;
            }
            """)
        func_hessian = self.mod_h.get_function("hessian")
        for i in range (0,100):
            self.scc = zeros_like(self.z) ;
            for l in range (0,self.p):
                self.scc [l,:,:] = self.bet[l] * self.z[l,:,:]
            self.ssum = zeros((self.laf,self.laf))
            for g in range (0,self.p):
                self.ssum = self.ssum + self.scc[g,:,:]
            self.sumte = sum(tril(exp(self.ssum)),axis=0)
            self.score = zeros((self.p))
            for n in range (0,self.p):
                self.temp = sum(divide(sum(tril(multiply(self.z[n,:,:],exp(self.ssum))),axis = 0),self.sumte))
                self.score[n] = trace(self.z[n,:,:])-self.temp
            self.vi = zeros ((self.p,self.p))
            self.vi =self.vi.astype(float32)
            self.laf_d = self.laf.astype(int32)
            # self.z2 = z.astype(float32)
            # part1_func = mod2.get_function("hess")
            self.ssum_d = exp(self.ssum)
            self.ssum_d= self.ssum_d.astype(float32)
            self.sumte_d = self.sumte.astype(float32)
            self.part1 = zeros([self.p, self.p, self.laf]).astype(float32)
            self.part2 = zeros([self.p, self.p, self.laf]).astype(float32)
            self.part3 = zeros([self.p, self.p, self.laf]).astype(float32)

            # part1_func(cuda.InOut(z2),cuda.InOut(ssum_d),cuda.InOut(sumte_d),int32(laf_d),cuda.InOut(part1),cuda.InOut(part2),cuda.InOut(part3),block = (int_(laf) ,1,1) ,grid = (p,p,1))
            # part2_func(cuda.InOut(z2),cuda.InOut(ssum_d),cuda.InOut(sumte_d),int32(laf_d),cuda.InOut(vi),cuda.InOut(part1),cuda.InOut(part2),cuda.InOut(part3),block = (p,1,1) ,grid = (p,1,1))
            # part1_func(cuda.InOut(float32(z2)),cuda.InOut(ssum_d),cuda.InOut(sumte_d),int32(laf_d),int32(p),cuda.InOut(vi),block = (p,1,1) ,grid = (p,1,1))
            part1_func(cuda.InOut(self.z), cuda.InOut(self.ssum_d), cuda.InOut(self.sumte_d), int32(self.laf_d), cuda.InOut(self.part1), block=(int_(self.laf), 1, 1),grid=(self.p, self.p, 1))
            part2_func(cuda.InOut(self.z), cuda.InOut(self.ssum_d), cuda.InOut(self.sumte_d), int32(self.laf_d), cuda.InOut(self.part2), block=(int_(self.laf), 1, 1),grid=(self.p,self.p, 1))
            part3_func(cuda.InOut(self.z), cuda.InOut(self.ssum_d), cuda.InOut(self.sumte_d), int32(self.laf_d), cuda.InOut(self.part3), block=(int_(self.laf), 1, 1), grid=(self.p, self.p, 1))
            func_hessian(cuda.InOut(self.z), cuda.InOut(self.ssum_d), cuda.InOut(self.sumte_d), int32(self.laf_d), cuda.InOut(self.vi), cuda.InOut(self.part1),
                  cuda.InOut(self.part2), cuda.InOut(self.part3), block=(self.p, 1, 1), grid=(self.p, 1, 1))
            # dot_temp = dot(vi.T,vi)
            # estimate = bet + dot(dot(linalg.inv(dot_temp + landa * diag(diag (dot_temp))), vi.T) , score)
            self.estimate = self.bet + reshape(dot(linalg.inv(self.vi),reshape(self.score, (self.p,1))),(1,self.p))[0]

            if i == 0:
                self.initial_score = zeros_like(self.score)
            if i > 1:
                if linalg.norm(self.score)<linalg.norm(self.initial_score):
                    self.landa = self.landa/2
                else:
                    self.landa = self.landa*2
            self.initial_score = self.score
            self.dif_temp = abs(self.bet-self.estimate)
            if ((self.dif_temp< self.tol).all()):
                self.bet_result = self.estimate
                self.flag = 0
                break
            self.bet = self.estimate
        if (self.flag==1):
            self.bet_result = 100000
            self.betahat = 1000000
            self.betaci = [1000000,1000000]
        else:
            self.betahat = self.bet_result
        self.x = norm.ppf(1-self.pval/2)
        self.nx = [-self.x,self.x]
        self.betaci = zeros((self.p,2))
        for i in range (0,self.p):
            self.betaci[i,0] = self.betahat[i] + self.nx[0] / sqrt(self.vi[i,i])
            self.betaci[i,1] = self.betahat[i] + self.nx[1] / sqrt(self.vi[i,i])
        return (self.betahat, self.betaci)

