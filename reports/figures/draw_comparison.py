"""Publication-sized vector comparison figures; original-valid and legacy targets."""
from pathlib import Path
import os
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplconfig'))
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator,FormatStrFormatter

ROOT=HERE.parent
SEEDS=[84,101,1234]
ORDER=['full','lstm','cnn_lstm','cnn_bilstm','transformer','dbfp_net','stgcn','agcrn','stv_pgnn_adapted']
STYLE={
 'full':('DA-PG-SHWATCN (Full)','#d62728','o','-'),
 'lstm':('LSTM','#1f77b4','s','-'),
 'cnn_lstm':('CNN-LSTM','#2ca02c','D','--'),
 'cnn_bilstm':('CNN-BiLSTM','#ff7f0e','^','-.'),
 'transformer':('Transformer','#9467bd','v',':'),
 'dbfp_net':('DBFP-Net','#8c564b','P','--'),
 'stgcn':('STGCN','#c66aa8','X',':'),
 'agcrn':('AGCRN','#707070','*','-.'),
 'stv_pgnn_adapted':('STV-PGNN-adapted','#138a91','h','-')}

plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8.5,
 'axes.titlesize':8.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
 'axes.linewidth':.7,'xtick.major.width':.65,'ytick.major.width':.65,
 'xtick.major.size':3,'ytick.major.size':3,'legend.fontsize':7.6,
 'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
 'savefig.facecolor':'white','axes.unicode_minus':False})

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load_data():
    observed=ROOT/'原始有效标签与缺失分层_84_101_1234_20261122'
    assert json.loads((observed/'completed.json').read_text(encoding='utf-8'))['status']=='passed'
    rows=json.loads((observed/'per_horizon_metrics.json').read_text(encoding='utf-8'))
    obs=pd.DataFrame([r for r in rows if r['scope']=='observed' and r['horizon']>0])
    legacy=[];sources=[observed/'per_horizon_metrics.json',observed/'coverage.json']
    for root in [ROOT/'对比实验_1234_21_42_84_20261119_164746',ROOT/'对比实验_101_20261120_223936']:
        path=root/'all_horizons_per_seed.csv';sources.append(path)
        frame=pd.read_csv(path);frame=frame[frame.seed.isin(SEEDS)&frame.horizon.ne('avg')]
        for _,r in frame.iterrows():
            for level in ['node','farm']:
                legacy.append(dict(model=r['case'],seed=int(r.seed),horizon=int(r.horizon[2:]),level=level,
                                   rmse=float(r[level+'_RMSE']),mae=float(r[level+'_MAE']),r2=float(r[level+'_R2'])))
    return {'observed':obs,'legacy':pd.DataFrame(legacy)},sources

def draw(frame,kind):
    assert len(frame)==9*3*6*2
    assert not frame[['model','seed','horizon','level']].duplicated().any()
    assert set(frame.seed)==set(SEEDS) and set(frame.model)==set(ORDER)
    summary=[]
    for (model,level,h),g in frame.groupby(['model','level','horizon']):
        assert set(g.seed)==set(SEEDS)
        row=dict(model=model,level=level,horizon=int(h),seed_count=3)
        for key in ['rmse','mae','r2']:
            row[key+'_mean']=float(g[key].mean());row[key+'_sd']=float(g[key].std(ddof=1))
        summary.append(row)
    summary=pd.DataFrame(summary)
    fig,axes=plt.subplots(2,3,figsize=(7.20,5.25))
    fig.subplots_adjust(left=.095,right=.985,bottom=.125,top=.795,wspace=.43,hspace=.47)
    x=np.arange(1,7)
    for row,level in enumerate(['node','farm']):
        for col,key in enumerate(['rmse','mae','r2']):
            ax=axes[row,col]
            all_low=[];all_high=[]
            for model in ORDER[1:]+ORDER[:1]:
                g=summary[summary.model.eq(model)&summary.level.eq(level)].sort_values('horizon')
                mu=g[key+'_mean'].to_numpy();sd=g[key+'_sd'].to_numpy()
                label,color,marker,linestyle=STYLE[model];full=model=='full'
                # Uncertainty on every model, rather than only highlighting Full's variation.
                ax.errorbar(x,mu,yerr=sd,fmt='none',ecolor=color,elinewidth=.55,
                            capsize=1.5,capthick=.55,alpha=.65 if full else .48,zorder=5 if full else 2)
                ax.plot(x,mu,color=color,marker=marker,linestyle=linestyle,
                        lw=1.65 if full else .88,markersize=4.2 if full else (4.8 if marker=='*' else 3.5),
                        markeredgewidth=.5,markeredgecolor=color,
                        markerfacecolor=color if full else 'white',
                        zorder=6 if full else 3)
                all_low.extend(mu-sd);all_high.extend(mu+sd)
            ax.set_xlim(.82,6.18);ax.set_xticks(x,[f't+{h}' for h in x])
            low=min(all_low);high=max(all_high);span=high-low
            if key=='r2':
                ax.set_ylim(max(-1,low-.08*span),min(1,high+.10*span))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                ylabel='R²'
            else:
                ax.set_ylim(max(0,low-.10*span),high+.10*span)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5,min_n_ticks=4))
                ax.ticklabel_format(axis='y',style='plain',useOffset=False)
                ylabel=key.upper()+' (kW)'
            ax.set_ylabel(ylabel,labelpad=4)
            if row==1:ax.set_xlabel('Forecast horizon',labelpad=5)
            ax.grid(axis='y',color='#d9dde1',linewidth=.45,zorder=0)
            ax.grid(axis='x',color='#e5e8eb',linewidth=.35,zorder=0)
            ax.set_axisbelow(True)
            for spine in ax.spines.values():spine.set_color('#333333')
            letter='abcdef'[row*3+col]
            title=('Turbine-level ' if row==0 else 'Farm-level ')+('R²' if key=='r2' else key.upper())
            ax.set_title(title,pad=8,loc='left')
            ax.text(-.02,1.07,f'({letter})',transform=ax.transAxes,ha='right',va='bottom',
                    fontweight='bold',fontsize=8.5)
    handles=[]
    for model in ORDER:
        label,color,marker,ls=STYLE[model]
        handles.append(Line2D([],[],label=label,color=color,marker=marker,ls=ls,
                       lw=1.65 if model=='full' else .95,markersize=4,
                       markerfacecolor=color if model=='full' else 'white',markeredgewidth=.6))
    legend=fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.986),ncol=3,
                  frameon=False,handlelength=2.3,columnspacing=2.2,handletextpad=.65,
                  labelspacing=.7,borderaxespad=0)
    # Statistical meaning is explicit at final print size, without a large decorative title.
    fig.text(.54,.839,'Mean ± sample SD across seeds 84, 101 and 1234',ha='center',fontsize=7.5,color='#454b50')
    tag='Observed power targets only' if kind=='observed' else 'Original scoring protocol'
    fig.text(.54,.023,tag+'  ·  134 turbines  ·  Six-step forecasts  ·  10-min nominal sampling',
             ha='center',fontsize=7.1,color='#454b50')
    stem='Fig_comparison_observed' if kind=='observed' else 'Fig_comparison_original_protocol'
    # Verify that labels, ticks and legend lie inside the requested fixed-size page.
    fig.canvas.draw();renderer=fig.canvas.get_renderer();page=fig.bbox
    for artist in [legend]+[t for ax in axes.flat for t in [ax.title,ax.xaxis.label,ax.yaxis.label,*ax.get_xticklabels(),*ax.get_yticklabels()]]:
        if not artist.get_visible() or (hasattr(artist,'get_text') and not artist.get_text()):continue
        box=artist.get_window_extent(renderer)
        assert box.x0>=-1 and box.y0>=-1 and box.x1<=page.x1+1 and box.y1<=page.y1+1,(stem,str(artist),box)
    for suffix in ['svg','pdf']:
        fig.savefig(HERE/(stem+'.'+suffix))
    fig.savefig(HERE/(stem+'.png'),dpi=600)
    fig.savefig(HERE/(stem+'.tiff'),dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(HERE/(stem+'_preview.png'),dpi=180)
    plt.close(fig)
    # SVG text stays editable; no data glyphs or labels silently rasterized.
    tree=ET.parse(HERE/(stem+'.svg'));texts=tree.findall('.//{http://www.w3.org/2000/svg}text')
    assert len(texts)>60 and all(label in (HERE/(stem+'.svg')).read_text(encoding='utf-8') for label in [v[0] for v in STYLE.values()])
    return summary.to_dict('records')

def main():
    data,sources=load_data();statistics={}
    for kind,frame in data.items():statistics[kind]=draw(frame,kind)
    (HERE/'figure_statistics.json').write_text(json.dumps(statistics,ensure_ascii=False,indent=2),encoding='utf-8')
    caption_en=('Comparison of six-step wind-power forecasts across nine models. Panels (a–c) show turbine-level RMSE, MAE and R²; '
       'panels (d–f) show the corresponding wind-farm metrics. Lines and markers denote means over three training seeds (84, 101 and 1234), '
       'and error bars indicate sample standard deviations, not confidence intervals. Forecast lead times t+1 to t+6 correspond nominally to 10–60 min. '
       'The main figure uses only finite, nonnegative original power observations as test labels; farm-level scores require valid observations '
       'for all 134 turbines. Predictions are from the previously selected checkpoints without retraining. Original retained timestamps include gaps; '
       'the scoring mask changes evaluation only, while training and checkpoint selection retain the original protocol.')
    caption_cn=('9种模型的六步功率预测结果比较。（a–c）为单机RMSE、MAE及R²；（d–f）为全场对应指标。折线表示种子84、101、1234的均值，'
       '误差棒表示样本标准差，非置信区间。t+1至t+6名义对应10–60分钟。主图仅使用原始有限且非负功率标签；全场要求134台标签全部有效。'
       '预测来自原验证准则选出的固定最佳检查点，未重新训练。原保留时间轴含缺口；本次仅改变测试评分，训练与检查点选择保持原协议。')
    notes=['主图：Fig_comparison_observed，原始有效标签评分，84/101/1234。',
           '附图：Fig_comparison_original_protocol，原对比实验评分，含原目标填补处理；不能与主图混用。',
           '风格：参考用户SVG的2×3布局、Arial字体、顶部共享图例、模型配色和红色Full曲线；补入STV-PGNN-adapted与三种子误差棒。',
           '成图尺寸：182.88×133.35 mm；正文刻度7.5 pt、坐标标签8.5 pt。保留九模型，未平滑、拟合或删除不利结果。',
           'SVG保留可编辑文字；PDF为嵌入字体的矢量图；PNG和LZW TIFF均600 dpi。',
           '三列指标沿用参考图。nRMSE/nMAE是容量归一化版本，与RMSE/MAE曲线仅相差常数，因此未在同图重复。',
           '误差棒按各预测步分别计算。每步曲线不能通过简单算术平均还原pooled RMSE或pooled R²；原脚本AVG为pooled。',
           '', '英文图注：',caption_en,'','中文图注：',caption_cn]
    (HERE/'图注与使用说明.txt').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    (HERE/'provenance.json').write_text(json.dumps(dict(seeds=SEEDS,models=ORDER,
         sources=[dict(path=str(p),sha256=sha(p)) for p in sources],
         reference_svg=dict(path='C:/Users/Administrator/Downloads/figure_compare_py.svg',sha256=sha(Path('C:/Users/Administrator/Downloads/figure_compare_py.svg'))),
         builder_sha256=sha(Path(__file__)),errorbars='sample SD; ddof=1; n=3',
         dimensions_mm=[182.88,133.35],raster_dpi=600),ensure_ascii=False,indent=2),encoding='utf-8')
    print('PASS: both protocols; 6 panels x 9 models; exact three seeds; editable SVG/PDF; 600-dpi PNG/TIFF')

if __name__=='__main__':main()
