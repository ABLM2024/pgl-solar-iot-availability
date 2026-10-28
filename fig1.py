import sys; sys.path.insert(0,'.'); from style import *
from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(7.2,3.5)); ax.set_xlim(0,100); ax.set_ylim(0,50); ax.axis('off')
def box(x,y,w,h,title,body,fc,ec):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4,rounding_size=1.2',fc=fc,ec=ec,lw=1.0))
    ax.text(x+w/2,y+h-1.6,title,ha='center',va='top',fontsize=7.6,fontweight='bold',color='#2b2b28')
    ax.text(x+w/2,y+h-5.2,body,ha='center',va='top',fontsize=6.4,color='#3a3a36',linespacing=1.25)
def arr(x0,y0,x1,y1): ax.annotate('',xy=(x1,y1),xytext=(x0,y0),arrowprops=dict(arrowstyle='-|>',color='#55554f',lw=0.9))
B,O,A,G='#eaf2fc','#fdeee7','#e6f6ef','#f4f4f1'
box(1,30,20,17,'Inputs','node telemetry (20 nodes)\ndaily availability $y_{n,t}$\nenclosure temperature\nsiting record\nclear-sky + climatology',G,'#9a9a94')
box(26,30,23,18,'Physics layer','VRLA energy balance\nLVD + reconnect + cold latch\nper-node harvest multiplier $s_n$\n(calibrated on training days)',A,'#1baf7a')
box(54,30,21,18,'State assimilation','reported down $\\rightarrow$ reserve empty\nreported up $\\rightarrow$ load connected\ncausal: data up to day $t$',A,'#1baf7a')
box(79,30,20,18,'h-day propagation','climatological drivers only\n$\\hat{E}_{t+h}$, min reserve,\nsimulated up/down',A,'#1baf7a')
box(26,6,23,15,'Data features','own lags, network context\ntemperature statistics\ncalendar, mounting type',O,'#eb6834')
box(54,6,21,15,'Physics prior','calibrated link\n$p_{\\mathrm{phys}}=\\sigma(\\mathbf{w}^{\\top}\\mathbf{z}+b)$',A,'#1baf7a')
box(79,6,20,15,'Physics-guided learner','boosted residual on\nlogit $p_{\\mathrm{phys}}$, monotone\nphysics constraints\n$\\rightarrow p_{n,t+h}$',B,'#2a78d6')
box(1,6,20,15,'Evaluations','E1 temporal (fixed + rolling)\nE2 leave-one-node-out\nE3 physics contribution\nPM$_{2.5}$ representativeness',G,'#9a9a94')
arr(21,39,26,39); arr(49,39,54,39); arr(75,39,79,39); arr(89,30,89,21.5); arr(64.5,30,64.5,21.5); arr(75,13.5,79,13.5)
ax.annotate('',xy=(83,21.5),xytext=(37.5,21.5),arrowprops=dict(arrowstyle='-|>',color='#55554f',lw=0.9,connectionstyle='arc3,rad=-0.18'))
arr(21,31,26,20)
fig.savefig('figs/fig1_framework.png'); plt.close()
