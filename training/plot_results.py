import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
fig,axes=plt.subplots(1,3,figsize=(15,5))

cql_data={
    'Planner':{'epochs':list(range(1,11)),
        'top1':[0.2722,0.8298,0.8483,0.8581,0.8633,0.8646,0.8669,0.8669,0.8674,0.8676],
        'top3':[0.5686,0.9260,0.9450,0.9533,0.9571,0.9589,0.9598,0.9600,0.9604,0.9603],
        'top5':[0.7660,0.9649,0.9771,0.9827,0.9847,0.9865,0.9867,0.9869,0.9871,0.9871]},
    'Reacter':{'epochs':list(range(1,11)),
        'top1':[0.1711,0.2133,0.2436,0.2656,0.2895,0.3124,0.3318,0.3425,0.3442,0.3457],
        'top3':[0.4438,0.4927,0.5250,0.5454,0.5759,0.6093,0.6405,0.6563,0.6624,0.6633],
        'top5':[0.6331,0.6876,0.7162,0.7329,0.7536,0.7853,0.8076,0.8206,0.8255,0.8262]},
    'Both':{'epochs':list(range(1,11)),
        'top1':[0.2281,0.6551,0.6941,0.7107,0.7209,0.7262,0.7300,0.7320,0.7330,0.7336],
        'top3':[0.5243,0.8498,0.8849,0.9010,0.9110,0.9161,0.9200,0.9215,0.9220,0.9225],
        'top5':[0.7271,0.9231,0.9529,0.9645,0.9707,0.9739,0.9760,0.9768,0.9770,0.9772]}
}
colors={'Planner':'#2ecc71','Reacter':'#e74c3c','Both':'#3498db'}
for name,d in cql_data.items():
    axes[0].plot(d['epochs'],np.array(d['top1'])*100,'-o',label=f'{name}',color=colors[name],linewidth=2,markersize=6)
axes[0].set_xlabel('Epoch',fontsize=12)
axes[0].set_ylabel('Top-1 Accuracy (%)',fontsize=12)
axes[0].set_title('CQL-LSTM Card Prediction',fontsize=14,fontweight='bold')
axes[0].legend(loc='lower right',fontsize=10)
axes[0].set_ylim([0,100])
axes[0].axhline(y=86.8,color='#2ecc71',linestyle='--',alpha=0.5)
axes[0].text(10.2,86.8,'86.8%',fontsize=9,color='#2ecc71',va='center')

horde_episodes=[50,100,200,500,1000,1500,2000]
horde_crown_long=[30.7,10.6,73.1,11.7,14.1,4.8,19.4]
horde_cql_win=[1.3,2.4,3.6,5.3,4.9,4.0,6.0]
horde_awr=[0.87,0.5,0.86,0.99,0.5,0.6,0.7]
ax1b=axes[1].twinx()
axes[1].plot(horde_episodes,horde_crown_long,'-s',color='#9b59b6',label='Crown (long γ)',linewidth=2,markersize=6)
axes[1].plot(horde_episodes,horde_cql_win,'-^',color='#e67e22',label='CQL Win',linewidth=2,markersize=6)
ax1b.plot(horde_episodes,horde_awr,'-d',color='#1abc9c',label='AWR Policy',linewidth=2,markersize=6)
axes[1].set_xlabel('Episode',fontsize=12)
axes[1].set_ylabel('TD Loss',fontsize=12)
ax1b.set_ylabel('AWR Loss',fontsize=12,color='#1abc9c')
axes[1].set_title('HORDE GVF Training (19 Demons)',fontsize=14,fontweight='bold')
lines1,labels1=axes[1].get_legend_handles_labels()
lines2,labels2=ax1b.get_legend_handles_labels()
axes[1].legend(lines1+lines2,labels1+labels2,loc='upper right',fontsize=9)

models=['HORDE\nGVF','CQL\nPlanner','CQL\nReacter','CQL\nBoth']
top1=[0,86.8,34.6,73.4]
top3=[0,96.0,66.3,92.3]
top5=[0,98.7,82.6,97.7]
x=np.arange(len(models))
w=0.25
axes[2].bar(x-w,top1,w,label='Top-1',color='#3498db')
axes[2].bar(x,top3,w,label='Top-3',color='#2ecc71')
axes[2].bar(x+w,top5,w,label='Top-5',color='#9b59b6')
axes[2].set_ylabel('Accuracy (%)',fontsize=12)
axes[2].set_title('Final Model Performance',fontsize=14,fontweight='bold')
axes[2].set_xticks(x)
axes[2].set_xticklabels(models,fontsize=10)
axes[2].legend(loc='upper left',fontsize=10)
axes[2].set_ylim([0,105])
for i,v in enumerate(top1):
    if v>0:axes[2].text(i-w,v+1,f'{v:.1f}',ha='center',fontsize=8)
for i,v in enumerate(top3):
    if v>0:axes[2].text(i,v+1,f'{v:.1f}',ha='center',fontsize=8)
for i,v in enumerate(top5):
    if v>0:axes[2].text(i+w,v+1,f'{v:.1f}',ha='center',fontsize=8)
axes[2].annotate('19 Demons\nTD(λ)+CQL+AWR',(0,50),fontsize=9,ha='center',color='#555')

plt.tight_layout()
plt.savefig('training/training_results.png',dpi=150,bbox_inches='tight')
print('Saved training/training_results.png')
