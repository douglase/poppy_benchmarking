#!/usr/bin/env python
# coding: utf-8

# Import dependencies:

# 
# libraries:
# 
#     conda install pyculib
#     conda install cudatoolkit=7.5
#     sudo apt-get install libfftw3-dev
#     pip install pyFFTW
#     

# In[1]:

import sys
import astropy.time
now=astropy.time.Time.now()
now.format="isot"
sys.stdout = open('benchmark_'+str(now)+'.log', 'w')
print('Benchmark recorded on'+str(now))

import numpy as np
import numexpr as ne

#import os
import importlib

import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['image.interpolation'] = 'nearest'
matplotlib.rcParams['font.size'] = 9

#import astropy.io
import astropy.units as u

import poppy
import pyfftw
import logging
logging.basicConfig(format='%(levelname)s:%(message)s',
                    level=logging.WARN)



get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





get_ipython().run_cell_magic('bash', '', 'git rev-parse HEAD')


# ## Print system info

# ### Software versions

# In[ ]:


from  poppy.accel_math import  _USE_CUDA, _USE_NUMEXPR, _FFTW_AVAILABLE

print(_USE_CUDA, _USE_NUMEXPR,_FFTW_AVAILABLE)
print("current POPPY version: "+str(poppy.__version__))


# In[ ]:


print(np.__version__)
print(np.__config__.show())


# In[ ]:


print("NumExpr info")
print(ne.__version__)
ne.ncores,ne.nthreads,ne.show_config()


# In[ ]:


print("FFTW info")
print(pyfftw.__version__)


# In[ ]:





# ### Hardware info

# In[ ]:


get_ipython().run_cell_magic('bash', '', "cat /proc/cpuinfo | grep 'model name' | uniq\ncat /proc/cpuinfo | grep 'cpu family' | uniq\n lscpu | grep 'Model:' | uniq\n lscpu | egrep '^Thread|^Core|^Socket|^CPU\\('\n")


# In[ ]:





# In[ ]:

try:
    get_ipython().run_cell_magic('bash', '', 'nvidia-smi\n')

except Exception as err:
    print(err)


# ## Benchmark some useful math

# In[ ]:


n=4096
y=np.ones([n,n],dtype=np.complex128)
x=2*np.ones([n,n],dtype=np.complex128)
x64bit=2*np.ones([n,n],dtype=np.complex64)

y64bit=2*np.ones([n,n],dtype=np.complex64)

z=9#2*np.ones([n,n])#*1.j


# In[ ]:


print("NumPy:")
get_ipython().run_line_magic('timeit', 'np.exp((x**2 + y**2)/z)')

try:
    import numba
    import cmath
    dtype=numba.complex128
    @numba.vectorize([dtype(dtype,dtype,dtype)],)
    def fexp(x, y,z):
        return cmath.exp((x**2 + y**2)/z)
    print("Numba optimized w/o GPU:")
    get_ipython().run_line_magic('timeit', 'fexp(x,y,z) #numba optimized')
except Exception as err:
    print(err)
    
try:
    import numba
    @numba.vectorize([numba.complex128(numba.float64,numba.float64,
                            numba.float64,numba.float64,
                            numba.float64,numba.float64)],
          target="parallel")
    
    def f_xyz_def(xr,xi,
         yr,yi,
         zr,zi):
        x=complex(xr,xi)

        y=complex(yr,yi)

        z=complex(zr,zi)
        return cmath.cos((x**2+y**2)/z) + 1j*cmath.sin((x**2+y**2)/z)
        #return cmath.exp((x**2+y**2)/z)
    print("Numba Optimized with GPU:")
    get_ipython().run_line_magic('timeit', 'f_xyz_def(x.real,x.imag,y.real,y.imag,z.real,z.imag)')
except Exception as err:
    print(err)


print("numexpr optimized:")
get_ipython().run_line_magic('timeit', 'ne.evaluate("exp((x**2 + y**2)/z)")')



# In[ ]:


x.shape


# In[ ]:


#confirm that exponents are faster than trigonometry

get_ipython().run_line_magic('timeit', 'ne.evaluate("exp(x)")')
get_ipython().run_line_magic('timeit', 'ne.evaluate("cos(x)+1j*sin(x)")')
get_ipython().run_line_magic('timeit', 'np.exp(x)')
get_ipython().run_line_magic('timeit', 'np.cos(x)+1j*np.sin(x)')


# In[ ]:


try:
    try:
        import pyculib as cuda
    except:
        print("failed to import pyculib FFT package")
        import accelerate.cuda as cuda

    print("unplanned CUDA FFT:")
    t = get_ipython().run_line_magic('timeit', '-o cuda.fft.ifft(y,out=x)')
    print("unplanned CUDA FFT, 64 bit complex:")
    get_ipython().run_line_magic('timeit', 'cuda.fft.ifft(y64bit,out=x64bit)')
    
    plan=cuda.fft.FFTPlan(y.shape,np.complex128,np.complex128)
    plan64=cuda.fft.FFTPlan(x64bit.shape,np.complex64,np.complex64)
    print("planned CUDA FFT:")
    get_ipython().run_line_magic('timeit', 'plan.inverse(y,out=x)')
    print("planned CUDA FFT, 64 bit complex:")
    get_ipython().run_line_magic('timeit', 'plan64.inverse(x64bit,out=y64bit)')



except Exception as err:
    print(err)
    
try:
    import pyfftw
    print("pyFFTW 32 threads:")
    get_ipython().run_line_magic('timeit', 'pyfftw.interfaces.numpy_fft.ifft2(y,threads=32)')
    print("pyFFTW 32 threads, 64 bit complex:")
    get_ipython().run_line_magic('timeit', 'pyfftw.interfaces.numpy_fft.ifft2(x64bit,threads=32)')
    print("pyFFTW 16 threads:")
    get_ipython().run_line_magic('timeit', 'pyfftw.interfaces.numpy_fft.ifft2(y,threads=16)')
except Exception as err:
    print(err)
    
print("numpy FFT:")
get_ipython().run_line_magic('timeit', 'np.fft.ifft2(y)')
print("numpy FFT, 64 bit complex: ")
get_ipython().run_line_magic('timeit', 'np.fft.ifft2(x64bit)')


# In[ ]:



#import pyculib
#import accelerate.cuda 
#%timeit -o accelerate.cuda.fft.ifft(y,out=x)
#get_ipython().run_line_magic('timeit', '-o pyculib.fft.ifft(y,out=x)')



# In[ ]:





# In[ ]:


# Test FFT shifts
get_ipython().run_line_magic('timeit', 'np.fft.fftshift(x)')
get_ipython().run_line_magic('timeit', 'poppy.accel_math._fftshift(x)')
print("8192 arrays:")
array8192=np.ones([8192,8192],dtype=np.complex128)
npshift_t = get_ipython().run_line_magic('timeit', '-o np.fft.fftshift(array8192)')
try:
    cudashift_t = get_ipython().run_line_magic('timeit', '-o poppy.accel_math._fftshift(array8192)')
except Exception as err:
    print(err)


# In[ ]:





# ## Define test system

# In[ ]:





# In[ ]:


import os
#export environment variable:
os.environ['WEBBPSF_PATH'] = os.path.expanduser('~/STScI/WFIRST/webbpsf-data')
import poppy
import astropy.units as u




def WFIRSTSPC(npix=256,ratio=0.25):
    Tel_fname = os.path.join(os.environ['WEBBPSF_PATH'], "AFTA_CGI_C5_Pupil_onax_256px_flip.fits")
    SP_fname = os.path.join(os.environ['WEBBPSF_PATH'], "CGI/optics/CHARSPC_SP_256pix.fits.gz")
    FPM_fname = os.path.join(os.environ['WEBBPSF_PATH'], "CGI/optics/CHARSPC_FPM_25WA90_2x65deg_-_FP1res4_evensamp_D072_F770.fits.gz")
    LS_fname = os.path.join(os.environ['WEBBPSF_PATH'], "CGI/optics/SPC_LS_30D88_256pix.fits.gz")


    D_prim = 2.37 * u.m
    D_relay = 20 * u.mm
    fr_pri = 7.8
    fl_pri = D_prim * fr_pri
    fl_m2 = fl_pri * D_relay / D_prim
    fr_m3 = 20.
    fl_m3 = fr_m3 * D_relay



    oversamp=4
    wfirst_optsys = poppy.FresnelOpticalSystem(pupil_diameter=D_prim, beam_ratio=ratio,
                                               npix=npix)

    telap = poppy.FITSOpticalElement(transmission=Tel_fname)
    SP = poppy.FITSOpticalElement(transmission=SP_fname)

    #default FPM pixelscale is in arcsecs
    FPM = poppy.FITSOpticalElement(transmission=FPM_fname,planetype=poppy.poppy_core.PlaneType.intermediate,
                                  pixelscale=0.005)
    SP.pixelscale=0.5*u.cm/SP.shape[0]/u.pix
    FPM.pixelscale=0.5*u.cm/SP.shape[0]/u.pix
    m1 = poppy.QuadraticLens(fl_pri, name='Primary')
    m2 = poppy.QuadraticLens(fl_m2, name='M2')
    m3 = poppy.QuadraticLens(fl_m3, name='M3')
    m4 = poppy.QuadraticLens(fl_m3, name='M4')
    m5 = poppy.QuadraticLens(fl_m3, name='M5')
    m6 = poppy.QuadraticLens(fl_m3, name='M6')

    wfirst_optsys.add_optic(telap)
    wfirst_optsys.add_optic(m1)
    wfirst_optsys.add_optic(m2, distance = fl_pri + fl_m2)
    wfirst_optsys.add_optic(m3, distance = fl_m2 + fl_m3)
    wfirst_optsys.add_optic(m4, distance = 2*fl_m3)
    wfirst_optsys.add_optic(SP, distance = fl_m3)
    wfirst_optsys.add_optic(m5, distance = fl_m3)
    wfirst_optsys.add_optic(FPM, distance = fl_m3)
    wfirst_optsys.add_optic(m5, distance = 2*fl_m3)

    wfirst_optsys.add_optic(poppy.ScalarTransmission(planetype=poppy.poppy_core.PlaneType.intermediate,
                                                     name='focus',),
                            distance=fl_m3+0.39999923*u.m)
    return wfirst_optsys
wavelen = 770e-9


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

psf,wf=WFIRSTSPC(npix=64).calc_psf(wavelength=wavelen,display_intermediates=True, return_intermediates=True)


# In[ ]:


import logging
logger = logging.getLogger("poppy")
logger.setLevel(logging.DEBUG)


for w in wf:
    print(w.pixelscale)


# In[ ]:





# In[ ]:


test_optsys= WFIRSTSPC()#optsys()
poppy.accel_math._FFTW_AVAILABLE= True
poppy.accel_math._CUDA_AVAILABLE= False
poppy.accel_math._USE_CUDA= False
poppy.accel_math._USE_NUMEXPR=True




print(poppy.accel_math._USE_CUDA,poppy.accel_math._CUDA_AVAILABLE,poppy.accel_math._USE_NUMEXPR,poppy.conf.use_fftw )
test_optsys=WFIRSTSPC(npix=512,ratio=0.25)
#run one test case so FFTW wisdom is cached.
#t1=test_optsys.calc_psf(wavelength=wavelen, display_intermediates=False, return_intermediates=False)
        #run timed test


# In[ ]:


#time = %timeit -o test_optsys.calc_psf(wavelength=wavelen, display_intermediates=False, return_intermediates=False)
psf_timing = get_ipython().run_line_magic('prun', '-r  -s  tottime test_optsys.calc_psf(wavelength=wavelen, display_intermediates=False, return_intermediates=False)')


# In[ ]:





# In[ ]:





# In[ ]:


#fraunhofer system
osys_fraun = poppy.OpticalSystem(npix=2048,oversample=4)
osys_fraun.add_pupil( poppy.CircularAperture(radius=3))    # pupil radius in meters
osys_fraun.add_detector(pixelscale=0.010, fov_arcsec=2.0)  # image plane coordinates in arcseconds

get_ipython().run_line_magic('timeit', 'psf = osys_fraun.calc_psf(2e-6,display_intermediates=False)# wavelength in microns')


# ## Benchmark baseline optical system with acceleration

# In[ ]:


#from astropy.table import Table
import pandas as pd
pixlist = pd.Series([2**4,2**6,2**7,
                     2**8,2**9,700,
                     #2**10,2**11, #expect tens of minute runtime with these enabled
                    ])


# In[ ]:


scenarios=[[False,False,False], #plain numpy
          [False,False,True], #FFTW
          [False,True,False], #numexpr
        [False,True,True], #numexpr, FFT
        [True,True,True], #CUDA, Numexpr, FFTW
          [True,True,False], #CUDA, Numexpr
          ]


# In[ ]:


timings={}
import importlib
logger.setLevel(logging.CRITICAL)


for combo in scenarios:
    df = pd.DataFrame(np.zeros([pixlist.size,4]), index=pixlist, columns=["avg",
                                                                          "std",
                                                                         "avg_fraun",
                                                                         "std_fraun",])
    print(combo)
    poppy.accel_math._CUDA_AVAILABLE  = combo[0]
    poppy.accel_math._USE_CUDA= combo[0]
    poppy.accel_math._NUMEXPR_AVAILABLE  = combo[1]
    poppy.accel_math._USE_NUMEXPR=combo[1]
    poppy.accel_math._FFTW_AVAILABLE= combo[2]
    poppy.accel_math._USE_FFTW= combo[2]

    time_fraun = get_ipython().run_line_magic('timeit', '-o osys_fraun.calc_psf(wavelen,display_intermediates=False)')


    print(poppy.accel_math._USE_CUDA,poppy.accel_math._USE_CUDA,poppy.accel_math._FFTW_AVAILABLE )
    for i,npix in enumerate(pixlist):
        test_optsys=WFIRSTSPC(npix=npix,ratio=0.25)
        #run one test case so FFTW wisdom is cached.
        t1=test_optsys.calc_psf(wavelength=wavelen, display_intermediates=False, return_intermediates=False)
        #run timed test
        time = get_ipython().run_line_magic('timeit', '-o test_optsys.calc_psf(wavelength=wavelen, display_intermediates=False, return_intermediates=False)')

        df.iloc[i]["std"] = time.stdev
        df.iloc[i]["avg"] = time.average
        df.iloc[i]["std_fraun"] = time_fraun.stdev
        df.iloc[i]["avg_fraun"] = time_fraun.average
    outname='cuda'+str(poppy.accel_math._USE_CUDA)+'NumExpr'+str(poppy.accel_math._USE_NUMEXPR)+'FFT'+str(poppy.accel_math._FFTW_AVAILABLE)+'.csv'
    df.to_csv(outname,float_format="%3.3e")
    print(outname)
    psf_timing = get_ipython().run_line_magic('prun', '-r  -s  tottime test_optsys.calc_psf(display_intermediates=False, return_intermediates=False)')
    psf_timing.print_stats()
    


# In[ ]:


#make sure system not broken
psf,wf=WFIRSTSPC(npix=64).calc_psf(wavelength=wavelen,display_intermediates=True, return_intermediates=True)


# In[ ]:





# In[ ]:


plt.figure(dpi=400, figsize=[7,5])
ax=plt.subplot(111)

y=pd.read_csv("noMKL"+'cuda'+str(False)+'NumExpr'+str(False)+'FFT'+str(False)+'.csv',index_col=0)


y_MKL=pd.read_csv("cuda"+str(False)+'NumExpr'+str(True)+'FFT'+str(False)+'.csv',index_col=0)


y_FFTW=pd.read_csv(""+'cuda'+str(False)+'NumExpr'+str(False)+'FFT'+str(True)+'.csv',index_col=0)

ax.errorbar(pixlist,y_FFTW["avg"]/y["avg"],
            yerr=y_FFTW["avg"]/y["avg"]*np.sqrt((y_FFTW["std"]/y_FFTW["avg"])**2+(y["std"]/y["avg"])**2),
            label="FFTW",linestyle="--" )
ax.errorbar(pixlist,y_MKL["avg"]/y["avg"],
            yerr=y_MKL["avg"]/y["avg"]*np.sqrt((y_MKL["std"]/y_MKL["avg"])**2+(y["std"]/y["avg"])**2),
            label=" NumExpr + MKL FFT",linestyle="--" )

y_numexpr=pd.read_csv(""+'cuda'+str(False)+'NumExpr'+str(True)+'FFT'+str(True)+'.csv',index_col=0)

ax.errorbar(pixlist,y_numexpr["avg"]/y["avg"],
            yerr=y_numexpr["avg"]/y["avg"]*np.sqrt((y_numexpr["std"]/y_numexpr["avg"])**2+(y["std"]/y["avg"])**2),
            label="NumExpr+FFTW",linestyle="-." )

y_all=pd.read_csv(""+'cuda'+str(True)+'NumExpr'+str(True)+'FFT'+str(True)+'.csv',index_col=0)

ax.errorbar(pixlist,y_all["avg"]/y["avg"],
            yerr=y_all["avg"]/y["avg"]*np.sqrt((y_all["std"]/y_all["avg"])**2+(y["std"]/y["avg"])**2),

            label="Numexpr+CUDA",  )


ax.plot([0,np.max(pixlist)],[1,1],linewidth=2,alpha=.5,color="black")
ax.text(256,1.02,"NumPy",color="gray")
#plt.xscale("log")
ax.set_xticks([pixlist[0]]+list(pixlist[4:]))
#ax.set_xticks(range(len(pixlist)))
ax.set_xticklabels(np.int_(ax.get_xticks()/test_optsys.beam_ratio))
ax.set_yticks(np.arange(0,1.1,.1))
ax.set_ylabel("Fractional Run Time")
plt.legend()
plt.grid()
plt.xlabel("Array Dimensions [pix]")
plt.ylim([0,1.1])
#plt.yscale("log")
plt.savefig("benchmarks%icores"%(ne.ncores)+str(now)+".pdf",bbox_inches="tight")


# In[ ]:


plt.figure(figsize=[4,3])
ax=plt.subplot(111)

ticks=np.int_(pixlist*4)
y=pd.read_csv('cuda'+str(False)+'NumExpr'+str(False)+'FFT'+str(False)+'.csv',
              index_col=0)

y["avg"].plot.bar(yerr=y["std"],
                       #ticks=ticks,
                  label="NumPy",color="orange",#,linestyle="--",
                   alpha=.7,ax=ax)

#ax.text(256,1.02,"NumPy",color="gray")
ax.set_ylabel("System Run Time [sec]")
#plt.legend()
plt.minorticks_on()

plt.grid(b=True, which='major', color='b', linestyle='-',alpha=.1)

plt.grid(b=True, which='minor', color='r', linestyle='--',alpha=.1)
plt.xlabel("Array $N$ [pix]")
#plt.yscale("log")
plt.yscale("log")
ax.set_xticklabels(ticks)


plt.savefig("NumPy_runtime%icores"%(ne.ncores)+str(now)+".pdf",bbox_inches="tight")


# In[ ]:


#print LaTeX table of Numpy Results:
df=pd.read_csv('cudaFalseNumExprFalseFFTFalse.csv',index_col=0)
df.drop(df.columns[[2,3]], axis=1, inplace=True)
df.index =df.index*4
x=df.to_latex()
print(x)


# ### where's the remaining bottleneck?
# 
# NumExpr evaluations still dominate

# In[ ]:


psf_timing = get_ipython().run_line_magic('prun', '-r  -s  tottime  WFIRSTSPC(npix=1024,ratio=0.25).calc_psf(display_intermediates=False, return_intermediates=False)')
psf_timing.print_stats()


# In[ ]:





# In[ ]:





# ## Talbot Effect Illustration

# In[ ]:





# In[ ]:


import poppy
import astropy.units as u
sineWFE=poppy.wfe.SineWaveWFE(spatialfreq=500,amplitude=5e-9)
wf_f = poppy.fresnel.FresnelWavefront(beam_radius=2*u.cm,wavelength=0.5*u.um,npix=256,oversample=8)
wf_f*=sineWFE
Z_t=2*((1/sineWFE.sine_spatial_freq))**2/wf_f.wavelength
       
       
wf_f *= poppy.CircularAperture(radius=wf_f.diam/2)
wf_f.propagate_fresnel(Z_t/10000.)


plt.figure(figsize=[8,4])
ax=wf_f.display(what="both",imagecrop=0.05,colorbar=True)
plt.suptitle("d=%.2f$Z_T$"%(wf_f.z/Z_t).decompose())



max_phase=6.28*(sineWFE.sine_amplitude/wf_f.wavelength).decompose()
amp_min=0.95
ax[0].images[0].set_clim(amp_min,1)
ax[1].images[0].set_clim(-max_phase,max_phase)
ax[1].images[0].set_cmap(plt.cm.magma)

plt.tight_layout()
plt.savefig("zt0.pdf",bbox_inches="tight")



plt.figure(figsize=[8,4])

wf_f.propagate_fresnel(Z_t*.05-wf_f.z)
ax=wf_f.display(what="both",imagecrop=0.05,colorbar=True)
ax[0].images[0].set_clim(amp_min,1)

ax[1].images[0].set_clim(-max_phase,max_phase)
ax[1].images[0].set_cmap(plt.cm.magma)
plt.suptitle("d=%.2f$Z_T$"%(wf_f.z/Z_t).decompose())

plt.tight_layout()
plt.savefig("zt1.pdf",bbox_inches="tight")


wf_f.propagate_fresnel(Z_t*.25-wf_f.z)
plt.figure(figsize=[8,4])
ax=wf_f.display(what="both",imagecrop=0.05,colorbar=True)
ax[0].images[0].set_clim(amp_min,1)

ax[1].images[0].set_clim(-max_phase,max_phase)
ax[1].images[0].set_cmap(plt.cm.magma)
plt.suptitle([(wf_f.z/Z_t).decompose(),np.std(wf_f.amplitude[wf_f.amplitude>1e-9])])
plt.tight_layout()


plt.suptitle("d=%.2f$Z_T$"%(wf_f.z/Z_t).decompose())



plt.tight_layout()
plt.savefig("zt2"+str(now)+".pdf",bbox_inches="tight")
print(Z_t.decompose())

