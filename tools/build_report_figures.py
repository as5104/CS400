import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path.cwd(); out=root/'output/final-results/report-figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
comparison=json.loads((root/'output/final-results/search-comparison/comparison.json').read_text())
models=list(csv.DictReader((root/'output/final-results/models.csv').open()))
severity=list(csv.DictReader((root/'output/final-results/severity-summary.csv').open()))
arms=['main-random-full','main-motpe-halving','main-random-halving','main-motpe-full']
labels=['Random / full','MOTPE / pruning','Random / pruning','MOTPE / full*']
styles=['-','--','-.',':'];markers=['o','s','^','D']
def save(name,fig):
    fig.savefig(out/(name+'.png'),dpi=200,bbox_inches='tight',facecolor='white')
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight',facecolor='white')
    plt.close(fig)

fig,ax=plt.subplots(figsize=(7,3.7))
for item,label,style,marker in zip(comparison['searches'],labels,styles,markers):
    points=item['hypervolume_history'];xs=[0]+[p['active_seconds']/3600 for p in points];ys=[0]+[p['hypervolume'] for p in points]
    ax.step(xs,ys,where='post',label=label,linestyle=style,color='black',linewidth=1.5,marker=marker,markersize=3)
ax.set(xlabel='Active search time (hours)',ylabel='Hypervolume',xlim=(0,5.25),ylim=(0,.55))
ax.grid(alpha=.2);ax.legend(fontsize=9,loc='lower right');fig.tight_layout();save('hypervolume',fig)

fig,axes=plt.subplots(1,2,figsize=(7.1,3.4))
main=[r for r in models if r['run']=='main-motpe-halving']
for front,label,marker in [('True','Observed Pareto set','o'),('False','Other completed','x')]:
    selected=[r for r in main if r['pareto']==front]
    for ax,column,scale in [(axes[0],'parameters',1e6),(axes[1],'training_seconds',60)]:
        ax.scatter([float(r[column])/scale for r in selected],[100*float(r['validation_accuracy']) for r in selected],
                   label=label,color='black',marker=marker,s=38)
        ax.grid(alpha=.2)
axes[0].set(xlabel='Parameters (millions)',ylabel='Final validation accuracy (%)')
axes[1].set(xlabel='Measured training time (minutes)',ylabel='Final validation accuracy (%)')
axes[0].legend(fontsize=8,loc='lower right');fig.tight_layout();save('pareto',fig)

fig,ax=plt.subplots(figsize=(7,3.6))
for arm,label,style,marker in zip(arms,labels,styles,markers):
    rows=[r for r in severity if r['run']==arm]
    ax.plot([int(r['severity']) for r in rows],[100*float(r['mean_accuracy']) for r in rows],
            color='black',linestyle=style,marker=marker,label=label,linewidth=1.3,markersize=4)
ax.set(xlabel='Corruption severity',ylabel='Archive mean corrupted accuracy (%)',xticks=range(1,6))
ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();save('severity',fig)

