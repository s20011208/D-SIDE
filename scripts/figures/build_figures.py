#!/usr/bin/env python3
"""Build D-SIDE README figures from a small, immutable saved-results bundle.

Requires only Python, NumPy, and Matplotlib. No model fitting, hidden checkpoint,
original workspace, network access, or SciPy is needed. The hero and workflow
are explicitly schematic; numerical panels use assets/figure_data.json.
"""
import argparse
import ctypes
import ctypes.util
import json
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
NAVY = '#102A43'
TEAL = '#087F8C'
TEAL_LIGHT = '#D9EEF0'
AMBER = '#C68426'
AMBER_LIGHT = '#F8EBD4'
GRAY = '#66788A'
LIGHT = '#E4EBF1'
INK = '#20374D'
BG = '#F6F9FC'


def save_figure(fig, output, name):
    fig.savefig(output/(name+'.svg'), bbox_inches='tight', pad_inches=.18,
                metadata={'Date': None, 'Creator': 'D-SIDE figure builder'})
    svg_path = output/(name+'.svg')
    svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines())+'\n')
    fig.savefig(output/(name+'.png'), dpi=170, bbox_inches='tight', pad_inches=.18)
    plt.close(fig)


def style_axis(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_color(LIGHT)
    ax.spines['bottom'].set_color(LIGHT)
    ax.tick_params(color=LIGHT, labelcolor=GRAY)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=LIGHT, linewidth=.7, alpha=.7)


def svg_begin(width, height, title, desc):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(desc)}</desc>']


def svg_text(x, y, text, size=15, color=INK, weight='400', **attrs):
    extra=' '.join(f'{k.replace("_","-")}="{v}"' for k,v in attrs.items())
    return f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" font-weight="{weight}" font-family="Arial, Helvetica, sans-serif" {extra}>{escape(text)}</text>'


def cubic_point(control_points, parameter):
    """Point on the same cubic Bezier curve drawn in the Pair schematic."""
    u = 1.-parameter
    coefficients = (u**3, 3*u*u*parameter, 3*u*parameter**2, parameter**3)
    return tuple(sum(w*p[k] for w,p in zip(coefficients, control_points)) for k in (0,1))


def equal_state_point(control_points, state):
    """Locate a shared vertical coordinate on a schematic cubic curve.

    Each displayed curve crosses this state once. Solve its actual Bezier
    ordinate rather than positioning the markers by eye.
    """
    lower, upper = 0., 1.
    lower_value = cubic_point(control_points, lower)[1]-state
    upper_value = cubic_point(control_points, upper)[1]-state
    if lower_value*upper_value > 0:
        raise ValueError('Schematic matching state is not bracketed by the curve')
    for _ in range(64):
        middle = (lower+upper)/2
        middle_value = cubic_point(control_points, middle)[1]-state
        if lower_value*middle_value > 0:
            lower, lower_value = middle, middle_value
        else:
            upper = middle
    parameter = (lower+upper)/2
    return cubic_point(control_points, parameter)


def operator_equation_svg(x, y):
    """Readable Kth-order operator notation, with explicit super/subscripts."""
    font = 'Cambria Math, STIX Two Math, DejaVu Serif, serif'
    pieces = [f'<g aria-label="D to the K with source-specific omega, applied to X, equals f sub g of X">']
    for dx,dy,text,size,italic in [(0,0,'D',23,True),(18,-14,'K',13,True),
                                  (18,9,'ω',13,True),(29,13,'g',9,True),
                                  (45,0,'X',23,True),(68,0,'=',21,False),
                                  (93,0,'f',23,True),(103,7,'g',12,True),
                                  (114,0,'(X)',23,True)]:
        pieces.append(f'<text x="{x+dx}" y="{y+dy}" fill="{NAVY}" '
                      f'font-size="{size}" font-family="{font}" '
                      f'font-style="{"italic" if italic else "normal"}">{escape(text)}</text>')
    pieces.append('</g>')
    return ''.join(pieces)


def hero(output):
    svg=svg_begin(1200,250,'D-SIDE: Distributional Selective Integration for Dynamics Estimation',
        'Research software for selective information sharing across systems with heterogeneous governing dynamics. Decorative trajectories are schematic.')
    svg += ['<defs><linearGradient id="back" x1="0" x2="1"><stop stop-color="#102A43"/><stop offset="1" stop-color="#16475A"/></linearGradient><clipPath id="clip"><rect width="1200" height="250" rx="18"/></clipPath></defs>',
            '<rect width="1200" height="250" rx="18" fill="url(#back)"/>',
            '<g clip-path="url(#clip)">',
            '<circle cx="1150" cy="60" r="210" fill="#FFFFFF" opacity=".025"/>',
            '<circle cx="1150" cy="60" r="150" fill="none" stroke="#FFFFFF" opacity=".04"/>',
            '<path d="M735 172 C785 57,825 42,870 115 S952 212,1010 88 S1128 78,1230 155" fill="none" stroke="#74C9CC" stroke-width="3" opacity=".72"/>',
            '<path d="M735 185 C783 82,829 72,874 132 S952 208,1012 113 S1127 94,1230 159" fill="none" stroke="#74C9CC" stroke-width="3" opacity=".46"/>',
            '<path d="M735 199 C786 111,829 101,878 149 S956 208,1013 139 S1128 110,1230 165" fill="none" stroke="#74C9CC" stroke-width="3" opacity=".27"/>',
            '<path d="M735 166 C793 174,806 91,864 77 S928 167,982 181 S1097 54,1230 83" fill="none" stroke="#DDA750" stroke-width="2.5" opacity=".75"/>',
            '</g>',
            svg_text(38,38,'RESEARCH SOFTWARE',12,'#9AC8D1','600',letter_spacing='2'),
            svg_text(36,102,'D-SIDE',58,'#FFFFFF','700',letter_spacing='1'),
            svg_text(39,148,'Distributional Selective Integration',23,'#F3F8FB','500'),
            svg_text(39,180,'for Dynamics Estimation',23,'#F3F8FB','500'),
            svg_text(39,223,'Scalar evidence. Selective borrowing. Nonparametric dynamics.',15,'#AFCAD6'),
            svg_text(1155,228,'Schematic trajectories',11,'#9AC8D1',text_anchor='end'),'</svg>']
    (output/'hero.svg').write_text('\n'.join(svg)+'\n')


def workflow(output):
    svg=svg_begin(1280,318,'D-SIDE workflow',
        'Algorithm schematic: observe noisy trajectories, form labeled equal-state scalar statistics, fit a reliability-weighted global distribution, cluster sources, and estimate shared nonlinear dynamics with source-specific operators. Excluded sources are fitted independently.')
    svg += ['<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0 L7 3.5 L0 7" fill="#8A9AA9"/></marker></defs>',
            '<rect width="1280" height="318" rx="14" fill="#F6F9FC"/>']
    cards=[('01','Observe','Noisy trajectories','Limited data per source'),
           ('02','Pair','Labeled scalar statistics','Match equal states'),
           ('03','Distribution','Distributional evidence','Weight the global density'),
           ('04','Group','Source clustering','Select one shared mechanism'),
           ('05','Estimate','Nonparametric dynamics','Keep source-specific operators')]
    for i,(num,verb,line1,line2) in enumerate(cards):
        x=18+i*252
        svg += [f'<rect x="{x}" y="20" width="236" height="244" rx="12" fill="#FFFFFF" stroke="#DFE8EF"/>',
                svg_text(x+16,45,num,11,TEAL,'700',letter_spacing='1'),
                svg_text(x+16,74,verb,22,NAVY,'700')]
        if i==0:
            for j in range(3):
                y=120+j*13
                svg.append(f'<path d="M{x+21} {y+25} C{x+63} {y-34},{x+124} {y+34},{x+206} {y-5}" fill="none" stroke="{TEAL if j<2 else AMBER}" stroke-width="2.3" opacity="{.95-j*.17}"/>')
            for xx,yy in [(39,134),(75,118),(112,141),(151,126),(191,129)]:
                svg.append(f'<circle cx="{x+xx}" cy="{yy}" r="3" fill="{TEAL}" opacity=".7"/>')
        elif i==1:
            curves = [([(24.,162.),(65.,110.),(102.,102.),(166.,120.)],NAVY,'navy'),
                      ([(30.,179.),(96.,174.),(126.,97.),(209.,100.)],TEAL,'teal')]
            shared_state = 124.
            endpoints = []
            for control,color,name in curves:
                shifted = [(x+xx,yy) for xx,yy in control]
                p0,p1,p2,p3 = shifted
                path = f'M{p0[0]} {p0[1]} C{p1[0]} {p1[1]},{p2[0]} {p2[1]},{p3[0]} {p3[1]}'
                svg.append(f'<path id="pair-{name}-curve" d="{path}" fill="none" stroke="{color}" stroke-width="2.4"/>')
                endpoints.append((*equal_state_point(shifted,shared_state),color,name))
            svg.append(f'<path id="equal-state-connector" d="M{endpoints[0][0]:.9f} {shared_state} H{endpoints[1][0]:.9f}" '
                       f'fill="none" stroke="{AMBER}" stroke-width="1.4" stroke-dasharray="4 4"/>')
            for xx,yy,color,name in endpoints:
                svg.append(f'<circle id="pair-{name}-point" cx="{xx:.9f}" cy="{yy:.9f}" r="4.3" fill="{color}"/>')
            svg.append(svg_text(x+174,148,'T',20,NAVY,'600'))
        elif i==2:
            for j,h in enumerate([10,17,28,47,63,55,32,20,15,12]):
                svg.append(f'<rect x="{x+21+j*19}" y="{178-h}" width="15" height="{h}" rx="2" fill="{TEAL}" opacity=".38"/>')
            svg.append(f'<path d="M{x+22} 171 C{x+48} 171,{x+66} 99,{x+103} 106 S{x+148} 169,{x+210} 172" fill="none" stroke="{NAVY}" stroke-width="2.3"/>')
        elif i==3:
            pts=[(44,121),(82,104),(100,148),(59,169),(153,118),(188,154)]
            for p,q in [(0,1),(0,2),(0,3),(1,2),(2,3)]:
                svg.append(f'<line x1="{x+pts[p][0]}" y1="{pts[p][1]}" x2="{x+pts[q][0]}" y2="{pts[q][1]}" stroke="#A6D7DB" stroke-width="2"/>')
            for j,(xx,yy) in enumerate(pts):
                svg.append(f'<circle cx="{x+xx}" cy="{yy}" r="10" fill="{TEAL if j<4 else AMBER}"/>')
        else:
            svg += [operator_equation_svg(x+36,135),
                    svg_text(x+116,158,'integer K ≥ 2',11,GRAY,text_anchor='middle'),
                    f'<path d="M{x+23} 190 C{x+69} 164,{x+137} 212,{x+210} 177" fill="none" stroke="{TEAL}" stroke-width="2.8"/>']
        svg += [svg_text(x+16,216,line1,15,NAVY,'600'),svg_text(x+16,242,line2,12,GRAY)]
        if i<4:svg.append(f'<path d="M{x+239} 143 H{x+250}" fill="none" stroke="#8A9AA9" stroke-width="1.6" marker-end="url(#arrow)"/>')
    svg += [svg_text(25,294,'Algorithm schematic  |  Sources outside the selected group retain independent fits.',13,GRAY),'</svg>']
    (output/'workflow.svg').write_text('\n'.join(svg)+'\n')


def source_selection(data, output):
    fig,axes=plt.subplots(1,3,figsize=(14.5,5.0),gridspec_kw={'width_ratios':[1.08,1.05,1.]})
    fig.subplots_adjust(left=.055,right=.98,top=.80,bottom=.30,wspace=.38)
    # Actual same-state pairs for a fixed displayed pair: sources 0 and 5.
    ax=axes[0]
    for g,color in [(0,NAVY),(5,TEAL)]:
        tr=data['training'][g]
        # The stored 301-point curve contains the selector's 61-node grid at
        # every fifth point. Draw that same polyline, so interpolation-based
        # equal-state markers lie exactly on the displayed line segments.
        ax.plot(tr['smooth_time'][::5],tr['selector_position'][::5],
                color=color,lw=2.1,label=f'Source {g}')
        ax.scatter(tr['time'],tr['observed_position'],s=11,color=color,alpha=.25,edgecolors='none')
    rr=[r for r in data['scalar_rows'] if r['g']==0 and r['h']==5]
    show=np.round(np.linspace(3,len(rr)-4,3)).astype(int)
    for i in show:
        r=rr[i];a,b=r['time_p'],r['time_q'];state=r['state']
        ax.plot([a,b],[state,state],color=AMBER,lw=1.6,ls=(0,(3,2)))
        ax.scatter([a,b],[state,state],s=29,c=[NAVY,TEAL],edgecolor='white',lw=.8,zorder=5)
    ax.set(xlabel='Training time (61-node matching grid)',ylabel='Observed position',xlim=(0,3))
    ax.set_title('01  Equal-state pairing',loc='left',fontweight='bold',pad=18)
    ax.legend(loc='upper left',frameon=False,fontsize=10)
    style_axis(ax)
    ax.text(.02,-.34,r'$T_{pq}=(b_q-b_p)/(A_p-A_q)$',transform=ax.transAxes,color=INK,fontsize=12)
    # The stacked density uses actual global weights and diagnostic truth labels.
    ax=axes[1]
    rows=data['scalar_rows'];z=np.array([r['z'] for r in rows]);w=np.array([r['global_weight'] for r in rows])
    good=np.array([r['g']<6 and r['h']<6 for r in rows]);bins=np.linspace(-np.pi/2,np.pi/2,30);dx=bins[1]-bins[0]
    ax.hist([z[good],z[~good]],bins=bins,weights=[w[good]/dx,w[~good]/dx],stacked=True,
        color=[TEAL,AMBER],alpha=.65,edgecolor='white',linewidth=.25,
        label=['Related pairs*','Other pairs*'])
    peak=data['peak'];mu,s,eta=peak['mu'],peak['sigma'],peak['proportion']
    zz=np.linspace(-np.pi/2,np.pi/2,600)
    phi=lambda u:.5*(1+math.erf(u/math.sqrt(2)))
    normalizer=phi((np.pi/2-mu)/s)-phi((-np.pi/2-mu)/s)
    q=np.exp(-.5*((zz-mu)/s)**2)/(s*math.sqrt(2*np.pi)*normalizer)
    ax.plot(zz,eta*q+(1-eta)/np.pi,color=NAVY,lw=2.,label='Fitted density')
    ax.axvspan(*data['center_interval'],color=NAVY,alpha=.045,zorder=0)
    ax.set(xlabel=r'Scalar statistic, $z=\arctan(T)$',ylabel='Weighted density',xlim=(-np.pi/2,np.pi/2))
    ax.set_title('02  Global distribution',loc='left',fontweight='bold',pad=18)
    ax.legend(loc='upper left',frameon=False,fontsize=9.4,handlelength=1.6)
    style_axis(ax)
    ax.text(0,-.34,'Pair reliability weights the density only.',transform=ax.transAxes,color=GRAY,fontsize=10.1)
    # Signed graph evidence plus the actual selected block.
    ax=axes[2];E=np.array(data['evidence'],float);np.fill_diagonal(E,np.nan)
    bound=max(.1,np.nanmax(np.abs(E)))
    cmap=LinearSegmentedColormap.from_list('evidence',[AMBER,'#FBFCFE',TEAL])
    im=ax.imshow(E,vmin=-bound,vmax=bound,cmap=cmap)
    for i in range(8):
        for j in range(8):
            if i!=j:
                ax.text(j,i,f'{E[i,j]:.2f}',ha='center',va='center',fontsize=7.4,color=INK)
    ax.add_patch(Rectangle((-.5,-.5),6,6,fill=False,ec=TEAL,lw=2.0))
    ax.set(xticks=range(8),yticks=range(8),xlabel='Source',ylabel='Source')
    ax.set_title('03  Source clustering',loc='left',fontweight='bold',pad=18)
    ax.tick_params(length=0,labelcolor=GRAY)
    for s in ax.spines.values():s.set_visible(False)
    cbar=fig.colorbar(im,ax=ax,fraction=.046,pad=.035)
    cbar.outline.set_visible(False);cbar.ax.tick_params(labelsize=8,length=0)
    ax.text(0,-.34,'Shared group: 0–5  |  Independent: 6, 7',transform=ax.transAxes,color=TEAL,fontsize=10.1,fontweight='medium')
    fig.text(.055,.945,'From pairwise trajectory contrasts to a selected source group',fontsize=17,fontweight='bold',color=NAVY)
    fig.text(.055,.075,'Actual saved run: seed 2026103000 · new damping, original initial conditions · 1 trajectory and 45 noisy observations per source.',fontsize=9.4,color=GRAY)
    fig.text(.055,.025,'* True mechanism labels are used for diagnostic colors only; they are not supplied to the selector. The shaded interval is not a confidence interval.',fontsize=9.4,color=GRAY)
    save_figure(fig,output,'source_selection')


def prediction(data, output):
    pred=data['prediction'];time=np.array(pred['time']);truth=np.array(pred['truth'])
    fig=plt.figure(figsize=(13.,5.6));gs=fig.add_gridspec(2,2,width_ratios=[1.,1.08],height_ratios=[1.8,1.],
        left=.065,right=.97,bottom=.19,top=.78,wspace=.34,hspace=.23)
    ax=fig.add_subplot(gs[0,0]);err=fig.add_subplot(gs[1,0],sharex=ax);metric=fig.add_subplot(gs[:,1])
    ax.plot(time,truth,color=NAVY,lw=2.3,label='Truth',zorder=3)
    ax.plot(time,pred['independent'],color=GRAY,lw=1.8,ls='--',label='Independent')
    ax.plot(time,pred['dside'],color=TEAL,lw=1.8,ls=(0,(3,1)),label='D-SIDE')
    ax.set(ylabel='Position',title='First held-out trajectory · source 0')
    ax.legend(frameon=False,ncol=3,fontsize=10,loc='upper left')
    ax.tick_params(labelbottom=False);style_axis(ax)
    err.plot(time,np.asarray(pred['independent'])-truth,color=GRAY,lw=1.6,ls='--')
    err.plot(time,np.asarray(pred['dside'])-truth,color=TEAL,lw=1.8)
    err.axhline(0,color=LIGHT,lw=1.)
    err.set(xlabel='Forecast time',ylabel='Error');err.ticklabel_format(axis='y',style='sci',scilimits=(-2,2));style_axis(err)
    scores=data['test_scores']
    values={m:[np.mean([r['mse'] for r in scores if r['method']==m and r['source']==g]) for g in range(8)]
            for m in ['independent','density_only']}
    y=np.arange(8)
    metric.axhspan(-.5,5.5,color=TEAL,alpha=.045)
    for g in range(8):
        metric.plot([values['independent'][g],values['density_only'][g]],[g,g],color='#B6C8D2',lw=1.8,zorder=1)
    metric.scatter(values['independent'],y,s=46,marker='o',facecolor='white',edgecolor=GRAY,lw=1.5,label='Independent',zorder=3)
    metric.scatter(values['density_only'],y,s=43,marker='D',color=TEAL,label='D-SIDE',zorder=4)
    metric.set(xscale='log',yticks=y,yticklabels=[f'Source {g}' for g in range(8)],xlabel='Mean held-out position MSE · log scale',
               title='Average over 10 retrospective seeds')
    metric.set_ylim(7.6,-.6)
    metric.legend(frameon=False,loc='upper right',fontsize=10)
    metric.grid(axis='x',color=LIGHT,lw=.7);metric.grid(axis='y',visible=False)
    metric.tick_params(axis='y',length=0);metric.tick_params(axis='x',color=LIGHT,labelcolor=GRAY)
    for s in metric.spines.values():s.set_visible(False)
    fig.text(.065,.94,'Estimate a mechanism, then simulate from a new initial state',fontsize=17,color=NAVY,fontweight='bold')
    fig.text(.065,.86,'Fixed illustrative test: first source and first trial. Right: 10 new trajectories per source per seed.',fontsize=11,color=GRAY)
    fig.text(.065,.09,'All methods receive the same test initial position and velocity. Forecasts use no future observations or true future states.',fontsize=10,color=GRAY)
    fig.text(.065,.035,'Single trajectory at left; aggregate results at right. Selective integration does not improve every held-out trajectory.',fontsize=10,color=GRAY)
    save_figure(fig,output,'prediction')


def svg_previews(output, names=('hero', 'workflow')):
    """Optional PNG previews via native librsvg; SVG is the primary format.

    This uses standard-library ctypes, not an added Python package. On systems
    without librsvg, all primary SVG and numerical PNG figures still rebuild.
    """
    rsvg_name = ctypes.util.find_library('rsvg-2')
    cairo_name = ctypes.util.find_library('cairo')
    object_name = ctypes.util.find_library('gobject-2.0')
    if not all([rsvg_name, cairo_name, object_name]):
        return False
    from xml.etree import ElementTree
    c = ctypes
    rsvg, cairo, obj = c.CDLL(rsvg_name), c.CDLL(cairo_name), c.CDLL(object_name)
    rsvg.rsvg_handle_new_from_file.argtypes = [c.c_char_p, c.POINTER(c.c_void_p)]
    rsvg.rsvg_handle_new_from_file.restype = c.c_void_p
    rsvg.rsvg_handle_render_cairo.argtypes = [c.c_void_p, c.c_void_p]
    rsvg.rsvg_handle_render_cairo.restype = c.c_int
    cairo.cairo_image_surface_create.argtypes = [c.c_int, c.c_int, c.c_int]
    cairo.cairo_image_surface_create.restype = c.c_void_p
    cairo.cairo_create.argtypes = [c.c_void_p]
    cairo.cairo_create.restype = c.c_void_p
    cairo.cairo_surface_write_to_png.argtypes = [c.c_void_p, c.c_char_p]
    cairo.cairo_surface_write_to_png.restype = c.c_int
    cairo.cairo_destroy.argtypes = [c.c_void_p]
    cairo.cairo_surface_destroy.argtypes = [c.c_void_p]
    obj.g_object_unref.argtypes = [c.c_void_p]
    for name in names:
        path = output/(name+'.svg')
        node = ElementTree.parse(path).getroot()
        width, height = int(node.get('width')), int(node.get('height'))
        error = c.c_void_p()
        handle = rsvg.rsvg_handle_new_from_file(str(path).encode(), c.byref(error))
        if not handle:
            raise RuntimeError('SVG preview failed to load: '+str(path))
        surface = cairo.cairo_image_surface_create(0, width, height)
        context = cairo.cairo_create(surface)
        try:
            if not rsvg.rsvg_handle_render_cairo(handle, context):
                raise RuntimeError('SVG preview failed to render: '+str(path))
            status = cairo.cairo_surface_write_to_png(surface, str(path.with_suffix('.png')).encode())
            if status:
                raise RuntimeError('SVG preview failed to save: '+str(path))
        finally:
            cairo.cairo_destroy(context)
            cairo.cairo_surface_destroy(surface)
            obj.g_object_unref(handle)
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'assets/figure_data.json')
    p.add_argument('--output',type=Path,default=ROOT/'assets')
    p.add_argument('--figures',nargs='+',choices=['hero','workflow','source_selection','prediction'],
                   default=['hero','workflow','source_selection','prediction'])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data=json.loads(a.input.read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10.5,'text.color':INK,'axes.labelcolor':INK,
        'axes.titlecolor':NAVY,'axes.titlesize':12.5,'axes.titleweight':'medium',
        'svg.fonttype':'none','svg.hashsalt':'D-SIDE','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
    builders={'hero':lambda:hero(a.output),'workflow':lambda:workflow(a.output),
              'source_selection':lambda:source_selection(data,a.output),
              'prediction':lambda:prediction(data,a.output)}
    for name in a.figures:builders[name]()
    svg_previews(a.output,[name for name in a.figures if name in ('hero','workflow')])
    print('Built '+', '.join(a.figures))


if __name__=='__main__':main()
