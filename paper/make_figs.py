import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, NullFormatter, FuncFormatter
from src.plots import set_style, METHOD_COLORS as C, METHOD_MARKERS as M, INK
from src.data_gen import generate_frames
OUT="/home/claude/paper/"
set_style(); plt.rcParams.update({"font.size":8,"axes.labelsize":8,"legend.fontsize":7,"xtick.labelsize":7,"ytick.labelsize":7,"axes.titlesize":8})
def save(fig,n): fig.savefig(OUT+n+".pdf"); fig.savefig(OUT+n+".png",dpi=300)
def logfmt(ax,ticks):
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p: f"{v:g}")); ax.yaxis.set_minor_formatter(NullFormatter()); ax.set_yticks(ticks)

# Fig1 constellation (single column, 4 panels 2x2)
rng=np.random.default_rng(0); snrs=[20,10,0,-5]
ideal=np.array([1+1j,1-1j,-1+1j,-1-1j])/np.sqrt(2)
fig,axes=plt.subplots(1,4,figsize=(7.16,1.95),sharex=True,sharey=True)
for ax,s in zip(axes,snrs):
    r=generate_frames(1,2000,snr_db=s,rng=rng)[0][0]
    ax.scatter(r.real,r.imag,s=1.5,alpha=.35,color=C["M2M4"],linewidths=0)
    ax.scatter(ideal.real,ideal.imag,s=22,marker="x",color=INK,linewidths=1)
    ax.set_title(f"SNR = {s} dB"); ax.set_aspect("equal"); ax.set_xlim(-2.5,2.5); ax.set_ylim(-2.5,2.5); ax.set_xlabel("In-phase (I)")
axes[0].set_ylabel("Quadrature (Q)"); fig.tight_layout(); save(fig,"fig_constellation")

# Fig2 RMSE vs SNR
d=pd.read_csv("results/tables/exp1_per_snr.csv",header=[0,1],index_col=0)
fig,ax=plt.subplots(figsize=(3.45,2.5))
for m in ["M2M4","Feature + MLP","1D CNN"]:
    ax.plot(d.index,d[(m,"rmse")],marker=M[m],color=C[m],lw=1.6,ms=4,label=m)
ax.set_yscale("log"); logfmt(ax,[0.2,0.5,1,2,5,10]); ax.set_xlabel("True SNR (dB)"); ax.set_ylabel("RMSE (dB)"); ax.legend()
fig.tight_layout(); save(fig,"fig_rmse_vs_snr")

# Fig3 frame length
d2=pd.read_csv("results/tables/exp2_frame_length.csv")
fig,axes=plt.subplots(1,3,figsize=(7.16,2.2),sharex=True,sharey=True)
for ax,col,t in zip(axes,["RMSE tüm","RMSE ≤ -4 dB","RMSE > -4 dB"],["All test SNRs","SNR ≤ −4 dB","SNR > −4 dB"]):
    for m in ["M2M4","Feature + MLP","1D CNN"]:
        x=d2[d2["Yöntem"]==m]; ax.plot(x["N"],x[col],marker=M[m],color=C[m],lw=1.6,ms=4,label=m)
    ax.set_xscale("log",base=2); ax.set_xticks([128,256,512,1024]); ax.set_xticklabels([128,256,512,1024])
    ax.set_yscale("log"); logfmt(ax,[0.5,1,2,5,10]); ax.set_ylim(0.4,16); ax.set_title(t); ax.set_xlabel("Frame length N (samples)")
axes[0].set_ylabel("RMSE (dB)"); axes[0].legend(); fig.tight_layout(); save(fig,"fig_frame_length")

# Fig4 robustness grouped bars
d3=pd.read_csv("results/tables/exp3_robustness.csv")
scen=["AWGN","Faz kayması","Faz + frekans kayması"]; scen_en=["AWGN","Phase\noffset","Phase + frequency\noffset"]
var=[("M2M4","M2M4","M2M4",False),("Feature + MLP (AWGN)","Feature + MLP","F-MLP (AWGN-trained)",True),("Feature + MLP (bozucu)","Feature + MLP","F-MLP (impairment-trained)",False),
     ("1D CNN (AWGN)","1D CNN","1D CNN (AWGN-trained)",True),("1D CNN (bozucu)","1D CNN","1D CNN (impairment-trained)",False)]
fig,ax=plt.subplots(figsize=(3.45,2.7)); w=0.16; x=np.arange(3)
plt.rcParams["hatch.linewidth"]=0.6
for i,(k,b,lab,awgn) in enumerate(var):
    v=[d3[(d3["Senaryo"]==s)&(d3["Yöntem"]==k)]["RMSE tüm"].item() for s in scen]
    ax.bar(x+(i-2)*w,v,w*0.92,color="white" if awgn else C[b],edgecolor=C[b],hatch="////" if awgn else None,lw=0.8,label=lab)
ax.set_xticks(x); ax.set_xticklabels(scen_en); ax.set_ylabel("RMSE over all test SNRs (dB)"); ax.grid(axis="x",visible=False)
ax.legend(fontsize=6,ncol=2,loc="upper center",bbox_to_anchor=(0.5,-0.24),frameon=False); ax.set_ylim(0,6.6)
fig.tight_layout(); save(fig,"fig_robustness")
print("ok")
