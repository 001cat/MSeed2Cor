import os,obspy,time
import numpy as np
import multiprocessing as mp
from Triforce.pltHead import *
from Triforce.utils import get_current_memory
from Triforce.obspyPlus import obspyFilter
from Triforce.fftOperation import Y2F,F2Y,fillNegtiveWing

def get_current_memory_rss() -> float: 
    ''' get memory usage of current process '''
    import os,psutil
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.rss / 1024. / 1024.

def _fastSlice(stShared,i,stSliced,starttime,endtime):
    stSliced[i] = [stShared[0].slice(starttime,endtime),starttime,endtime]
def fastSlice(st,starttime,endtime,segL,nproc=12):
    t0 = starttime
    sliceTimeList = []
    while t0 < endtime:
        sliceTimeList.append([t0,t0+segL])
        t0 += segL
    N = len(sliceTimeList)
    manage = mp.Manager()
    stShared = manage.list([st])
    stSliced = manage.list(range(N))
    argInLst = [ [stShared,i,stSliced,sliceTime[0],sliceTime[1]] 
                 for i,sliceTime in enumerate(sliceTimeList)]
    pool = mp.Pool(processes=nproc)
    pool.starmap(_fastSlice, argInLst)
    pool.close()
    pool.join()
    return list(stSliced)



def smoothMovingAvg(x,n):
    '''moving average smoothing'''
    w = np.ones(2*n+1,'d')
    y = np.convolve(w,x,mode='same')
    y[:n]   /= n+1+np.arange(n)
    y[n:-n] /= 2*n+1
    y[-n:]  /= n+np.arange(n,0,-1)
    return y

def flatHann(f,f1,f2,f3,f4):
    win = np.zeros(f.shape)
    I = (f>f1)*(f<=f2)
    win[I] = ((1-np.cos(np.pi*(f1-f)/(f2-f1))) * 0.5)[I]
    I = (f>f2)*(f<=f3)
    win[I] = 1
    I = (f>f3)*(f<=f4)
    win[I] = ((1+np.cos(np.pi*(f3-f)/(f4-f3))) * 0.5)[I]
    return win

def temperalNorm(tr):
    poleThres = 1e-20
    winL = 1000
    def markEventSegs(trIn,freqBand=None):
        tr = trIn.copy()
        if freqBand is not None:
            tr = trIn.copy()
            if freqBand[0] is None:
                tr.filter('lowpass',freq=freqBand[1])
            elif freqBand[1] is None:
                tr.filter('highpass',freq=freqBand[0])
            else:
                tr.filter('bandpass',freqmin=freqBand[0],freqmax=freqBand[1],zerophase=True)
        win_max = np.array([a.data.max() for a in tr.slide(window_length=winL, step=winL)])
        win_max_sorted = win_max.copy();win_max_sorted.sort()
        win_max_sorted = win_max_sorted[win_max_sorted>poleThres]
        noisemax = 2*win_max_sorted[:30].mean() # 30 quiet windows works for one day segL and 1000 winL
                                                # should be changed if using other settings
        if len(win_max_sorted[win_max_sorted<noisemax]) == 0:
            return [False]*len(win_max)
        window_avg = win_max_sorted[win_max_sorted<noisemax].mean()
        window_std = win_max_sorted[win_max_sorted<noisemax].std()
        return win_max < window_avg+2*window_std
    npole = (abs(tr.data) < poleThres).sum()
    if npole > 600/tr.stats.delta:
        print(f'Too many poles: {npole}')
        return None,None
    keep = markEventSegs(tr,[1/50,1/15])
    if np.sum(keep) < 10:
        print(f'No enough AN available: {np.sum(keep)}')
        return None,None
    # print(keep)
    t0 = tr.stats.starttime
    iHead,iTail = None,None
    recSegs = []
    for i in range(len(keep)):
        if not keep[i]:
            tr.slice(t0+i*winL,t0+(i+1)*winL).data[:] = 0
            if iHead is not None:
                iTail = i-1
        elif i==0 or keep[i-1] == False:
            iHead = i
        elif i == len(keep)-1:
            iTail = i
        if iHead is not None and iTail is not None:
            if iTail - iHead < 2:
                tr.slice(t0+iHead*winL,t0+(iTail+1)*winL).data[:] = 0
            else:
                tr.slice(t0+iHead*winL,t0+(iTail+1)*winL).taper(0.5,max_length=150)
                recSegs.append((iHead*winL,(iTail+1)*winL))
            iHead,iTail = None,None
    tr.slice(t0+(i+1)*winL,t0+(i+2)*winL).data[:] = 0       # remaining part less than winL
    return tr,recSegs

def spectralNorm(tr,freqmin,freqmax):
    f,F = Y2F(tr.stats.delta,tr.data,fftw=True)
    f2,f3 = freqmin,freqmax
    f1,f4 = f2*0.8,f3*1.2
    I = (f>=f1) * (f<=f4)
    F[I] /= smoothMovingAvg(abs(F[I]),int(np.floor(0.0002/(f[1]-f[0])+0.5)))
    F *= flatHann(f,f1,f2,f3,f4)
    F = fillNegtiveWing(F)
    amp,pha = np.abs(F),np.angle(F)
    return f,amp,pha

def calSpecAN(trIn,freqmin=0.005,freqmax=0.4):
    tr = trIn.copy()
    tr,recSegs = temperalNorm(tr)
    if tr is None:
        return None
    f,amp,pha=spectralNorm(tr,freqmin=freqmin,freqmax=freqmax)
    return [f,amp,pha,recSegs,tr.stats.starttime,tr.stats.delta]

def _ambientNoiseSpectrumEntry(st,t0,specDict,freqmin,freqmax,removeResponse=True):
    print(t0.strftime("%Y%m%d"))
    st.detrend('linear')
    st.merge(fill_value=0)
    if len(st) != 1:
        print(f'Warning: {len(st)} traces found, only 1 trace allowed!')
    else:
        if removeResponse:
            st.remove_response(pre_filt=[freqmin,freqmin/0.8,freqmax/1.2,freqmax],taper=False)
        else:
            st.filter('bandpass',freqmin=freqmin,freqmax=freqmax)
        spec = calSpecAN(st[0],freqmin=freqmin,freqmax=freqmax)
        if spec is not None:
            # print(f'Finished {t0}')
            specDict[t0.strftime("%Y%m%d")] = spec

def ambientNoiseSpectrum(stIn,starttime,endtime,freqmin=0.005,freqmax=0.4,
                         segL=86400,overwrite=False,nproc=12,removeResponse=True):
    net,sta = stIn[0].stats.network,stIn[0].stats.station
    if os.path.exists(f'{net}.{sta}.npz') and not overwrite:
        return 0
    print(f'{net}.{sta}:')
    for tr in stIn:
        if np.any(np.isnan(tr.data)):
            print(f'Error: NaN found in trace {tr}')
            return -1
    starttime = obspy.UTCDateTime(starttime); endtime = obspy.UTCDateTime(endtime)
    manager = mp.Manager()
    specDict = manager.dict()
    print('Slicing...')
    stSliced = fastSlice(stIn,starttime,endtime,segL,nproc=nproc)
    argInLst = [[st.copy(),t0,specDict,freqmin,freqmax,removeResponse] for st,t0,_ in stSliced]

    print('Calculating...')
    pool = mp.Pool(processes=nproc)
    pool.starmap(_ambientNoiseSpectrumEntry, argInLst)
    pool.close()
    pool.join()

    specDict = dict(specDict)
    if len(specDict.keys()) == 0:
        print('No output!')
        return -1
    np.savez_compressed(f'{net}.{sta}.npz',specDict=specDict)
    print('Finished!')
    
def calCorrAN(spec1,spec2,lagT,interp=True):
    f1,amp1,pha1,recSegs1,starttime1,dt1 = spec1
    f2,amp2,pha2,recSegs2,starttime2,dt2 = spec2

    if len(f1)!=len(f2) or (not np.allclose(f1,f2)) or (dt1-dt2>1e-4):
        print('Incompatible found!')
        return None,None,None

    dt = dt1
    timemisfit = starttime2-starttime1
    # if abs(timemisfit) > 0.002:
    #     print(f'Warning: Different start time found: {timemisfit}')
    #     # return None,None,None

    corAmp = amp1*amp2
    corPha = pha2-pha1
    corF = np.zeros(amp1.shape,dtype=complex)
    corF.real = corAmp*np.cos(corPha)
    corF.imag = corAmp*np.sin(corPha)

    _,Y = F2Y(f1,corF,fftw=True)
    if interp:
        nlag = int(lagT/dt+10)
    else:
        nlag = int(np.floor(lagT/dt+0.5))

    cor          = np.zeros(2*nlag+1)
    cor[:nlag]   = Y.real[nlag:0:-1]
    cor[nlag]    = Y.real[0]
    cor[nlag+1:] = Y.real[-1:-nlag-1:-1]

    def calRecCount(segs1,segs2,nlag,dt):
        def overlapL(a0,a1,b0,b1):
            if b0>a1 or a0>b1:
                return 0
            x = [a0,a1,b0,b1]
            x.sort()
            return x[2]-x[1]
        recCount = np.zeros(2*nlag+1)
        for seg1 in segs1:
            for seg2 in segs2:
                for ilag in range(-nlag,nlag+1):
                    L = overlapL(seg1[0],seg1[1],seg2[0]-ilag*dt,seg2[1]-ilag*dt)
                    recCount[ilag+nlag] += int(np.floor(L/dt+0.5))
        return recCount
    recCount = calRecCount(recSegs1,recSegs2,nlag,dt)
    cor /= recCount
    lag = np.arange(-nlag,nlag+1)*dt
    if interp:
        lagN = lag[abs(lag)<lagT+dt/2]
        cor = np.interp(lagN,lag+timemisfit,cor)
        lag,timemisfit = lagN,0
    return lag,cor,timemisfit

def stackCorrAN(corrList,minStackPercent=0):
    stackN = 0
    shifts = np.array([corr[2] for corr in corrList])
    Nshift = [np.sum(abs(shifts-shift) < 0.002) for shift in shifts]
    shift0 = shifts[np.argmax(Nshift)]
    for corr in corrList:
        lag,cor,shift = corr
        if lag is None:
            continue
        if abs(shift0-shift)>0.0002:
            continue
        if stackN == 0:
            corStack = cor
        else:
            # if abs(shift0-shift)>0.0005:
            #     print(f'Time shift is not constant: {shift0} {shift}')
            #     return None,None,None,None
            corStack += cor
        stackN += 1
    if stackN == 0 or stackN < minStackPercent*len(corrList):
        return None,None,None,None
    else:
        corStack /= stackN
        return lag,corStack,shift0,stackN

def _ambientNoiseCorr(spec1,spec2,date,lagT,corrList):
    lag,cor,shift = calCorrAN(spec1,spec2,lagT)
    if lag is not None:
        corrList[date] = [lag,cor,shift]
    pass

def ambientNoiseCorr(anSpec1,anSpec2,starttime=None,endtime=None,lagT=3000,nproc=12):
    try:
        specDict1 = np.load(anSpec1,allow_pickle=True)['specDict'][()]
        specDict2 = np.load(anSpec2,allow_pickle=True)['specDict'][()]
    except:
        return None,None,None,None
    argInLst = []
    manager = mp.Manager()
    corrList = manager.dict()
    for date in specDict1.keys():
        if date not in specDict2.keys():
            continue
        if endtime is not None and obspy.UTCDateTime(date) > obspy.UTCDateTime(endtime):
            continue
        if starttime is not None and obspy.UTCDateTime(date) < obspy.UTCDateTime(starttime):
            continue
        spec1 = specDict1[date]
        spec2 = specDict2[date]
        argInLst.append([spec1,spec2,date,lagT,corrList])


    pool = mp.Pool(processes=nproc)
    pool.starmap(_ambientNoiseCorr, argInLst)
    pool.close()
    pool.join()

    
    

    corrList = [corrList[date] for date in corrList.keys()]
    # return corrList
    return stackCorrAN(corrList)


def runTest(dataDir,id1,id2,starttime,endtime,freqmin,freqmax):
    net1,sta1,_,_ = id1.split('.')
    net2,sta2,_,_ = id2.split('.')
    st1 = obspy.read(f'{dataDir}/{sta1}.{net1}.mseed').select(id=id1)
    st1.attach_response(obspy.read_inventory(f'{dataDir}/IRISDMC-{sta1}.{net1}.xml'))
    st2 = obspy.read(f'{dataDir}/{sta2}.{net2}.mseed').select(id=id2)
    st2.attach_response(obspy.read_inventory(f'{dataDir}/IRISDMC-{sta2}.{net2}.xml'))


    ambientNoiseSpectrum(st1,starttime,endtime)
    ambientNoiseSpectrum(st2,starttime,endtime)
    lag,cor,shift,N = ambientNoiseCorr(f'{net1}.{sta1}.npz',f'{net2}.{sta2}.npz',lagT=3000)

    np.savez_compressed(f'{net1}.{sta1}-{net2}.{sta2}.npz',lag=lag,cor=cor,shift=shift,N=N)

    dt = lag[1]-lag[0]
    cor = obspyFilter(dt,cor,'bandpass',freqmin=freqmin,freqmax=freqmax,zerophase=True)
    plt.figure()
    plt.plot(lag,cor)

    os.system('rm *.npz')

if __name__ == "__main__":
    # test1:2002  test2:200408
    # runTest('test1','CI.PHL..LHZ','CI.MLAC..LHZ','20020105','20020106',1/10,1/5)
    # runTest('test2','IU.GUMO.00.BHZ','IU.TATO.00.BHZ','20040809','20040810',1/50,1/20)
    # runTest('test1','CI.PHL..LHZ','CI.MLAC..LHZ','20020101','20030101',1/10,1/5)
    runTest('test2','IU.GUMO.00.BHZ','IU.TATO.00.BHZ','20040801','20040901',1/50,1/20)












