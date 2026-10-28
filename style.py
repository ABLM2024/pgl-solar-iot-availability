import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','Times New Roman','DejaVu Serif'],'font.size':8.5,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':'#e6e6e3','grid.linewidth':0.6,
    'axes.edgecolor':'#8a8a85','axes.linewidth':0.7,'xtick.color':'#55554f','ytick.color':'#55554f','axes.labelcolor':'#2b2b28',
    'legend.frameon':False,'savefig.dpi':600,'savefig.bbox':'tight','lines.linewidth':1.6})
C=dict(prop='#2a78d6',ml='#eb6834',phys='#1baf7a',pers='#eda100',v1='#e87ba4',clim='#9a9a94',obs='#2b2b28',concat='#6250d6')
MK=dict(prop='o',ml='s',phys='^',pers='D',v1='v',clim='x',concat='P')
LS=dict(prop='-',ml='--',phys='-.',pers=':',v1=(0,(3,1,1,1)),clim=(0,(1,2)),concat=(0,(5,2)))
