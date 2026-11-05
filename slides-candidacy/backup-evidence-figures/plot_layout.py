"""Place point labels after layout and check rendered text for collisions."""
import math
import numpy as np
from matplotlib.text import Text
from matplotlib.transforms import Bbox


def point_label(ax,label,xy,fontsize=11):
    requests=getattr(ax,'_point_label_requests',[])
    requests.append({'label':str(label),'xy':tuple(xy),'fontsize':min(fontsize,9)})
    ax._point_label_requests=requests


def outside_legend(fig):
    handles=[];labels=[]
    for ax in fig.axes:
        if ax.get_legend() is None:continue
        hh,ll=ax.get_legend_handles_labels()
        for h,l in zip(hh,ll):
            if l not in labels:handles.append(h);labels.append(l)
        ax.get_legend().remove()
    return handles,labels


def _intersection(a,b,pad=0):
    if pad:
        a=Bbox.from_extents(a.x0-pad,a.y0-pad,a.x1+pad,a.y1+pad)
    c=Bbox.intersection(a,b)
    return c is not None and c.width>.5 and c.height>.5


def place_point_labels(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();scale=fig.dpi/72
    results=[]
    for ax in fig.axes:
        requests=getattr(ax,'_point_label_requests',[])
        if not requests:continue
        static=[t.get_window_extent(renderer) for t in ax.texts if t.get_visible() and t.get_text().strip()]
        point_boxes=[]
        for collection in ax.collections:
            if not hasattr(collection,'get_offsets'):continue
            offsets=np.ma.filled(collection.get_offsets(),np.nan)
            if offsets.ndim!=2 or offsets.shape[1]!=2:continue
            pixels=collection.get_offset_transform().transform(offsets)
            sizes=collection.get_sizes() if hasattr(collection,'get_sizes') else [20]
            radius=(math.sqrt(float(np.max(sizes))) / 2 + 2)*scale if len(sizes) else 3*scale
            for x,y in pixels:
                if np.isfinite(x+y) and ax.bbox.contains(x,y):
                    point_boxes.append([x-radius,y-radius,x+radius,y+radius])
        points=np.array(point_boxes).reshape(-1,4)
        line_paths=[line.get_path().transformed(line.get_transform()) for line in ax.lines
                    if line.get_visible() and line.get_linestyle() not in ['None','none','',None]]
        occupied=list(static)
        for req in sorted(requests,key=lambda x:(-len(x['label']),x['label'])):
            ann=ax.annotate(req['label'],req['xy'],xytext=(8,8),textcoords='offset points',
                            fontsize=req['fontsize'],ha='left',va='bottom',zorder=8,
                            annotation_clip=False,
                            arrowprops={'arrowstyle':'-','color':'#69757D','lw':.65,
                                        'shrinkA':2,'shrinkB':4})
            candidates=[]
            for radius in [9,15,23,33,46,62,82,110,145]:
                for angle in [45,135,-45,-135,90,-90,0,180,25,155,-25,-155]:
                    dx=radius*math.cos(math.radians(angle));dy=radius*math.sin(math.radians(angle))
                    candidates.append((dx,dy))
            selected=None
            safe=Bbox.from_extents(ax.bbox.x0+4*scale,ax.bbox.y0+4*scale,
                                   ax.bbox.x1-4*scale,ax.bbox.y1-4*scale)
            for dx,dy in candidates:
                ann.set_position((dx,dy))
                ann.set_ha('left' if dx>1 else 'right' if dx< -1 else 'center')
                ann.set_va('bottom' if dy>1 else 'top' if dy< -1 else 'center')
                ann.update_positions(renderer)
                box=Text.get_window_extent(ann,renderer)
                if not (safe.contains(box.x0,box.y0) and safe.contains(box.x1,box.y1)):continue
                if any(_intersection(box,other,3*scale) for other in occupied):continue
                line_box=Bbox.from_extents(box.x0-2*scale,box.y0-2*scale,
                                          box.x1+2*scale,box.y1+2*scale)
                if any(path.intersects_bbox(line_box,filled=False) for path in line_paths):continue
                if len(points):
                    pad=2*scale
                    hit=((points[:,0]<box.x1+pad)&(points[:,2]>box.x0-pad)&
                         (points[:,1]<box.y1+pad)&(points[:,3]>box.y0-pad))
                    if hit.any():continue
                selected=(dx,dy,box);break
            if selected is None:
                ann.remove()
                raise ValueError(f'No collision-free placement for {req["label"]}')
            dx,dy,box=selected;occupied.append(box.frozen())
            results.append({'label':req['label'],'anchor':list(req['xy']),
                            'offset_points':[dx,dy],'bbox_pixels':list(box.extents)})
    fig.canvas.draw()
    return results


def text_collisions(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();texts=[];seen=set();hidden_ticks=set()
    # Matplotlib retains Text objects for locator ticks outside the view. The
    # axis draw routine omits those ticks, so they are not rendered collisions.
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]:
            lo,hi=sorted(axis.get_view_interval())
            for tick in [*axis.get_major_ticks(),*axis.get_minor_ticks()]:
                if not ax.axison or tick.get_loc()<lo-1e-10 or tick.get_loc()>hi+1e-10:
                    hidden_ticks.update([id(tick.label1),id(tick.label2)])
    for t in fig.findobj(match=Text):
        if id(t) in seen or id(t) in hidden_ticks or not t.get_visible() or not t.get_text().strip():continue
        seen.add(id(t));b=Text.get_window_extent(t,renderer)
        if b.width>0 and b.height>0:texts.append((t.get_text(),b))
    found=[]
    for i,(a,ra) in enumerate(texts):
        for b,rb in texts[i+1:]:
            if _intersection(ra,rb):found.append({'first':a,'second':b})
    return found
