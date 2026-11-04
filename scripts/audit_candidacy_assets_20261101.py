"""Read-only image audit of the current candidacy PDF and its local assets."""
import os
os.environ['NUMPY_MADVISE_HUGEPAGE']='0'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pymupdf

ROOT=Path(__file__).resolve().parents[1]/'slides-candidacy'
OUT=Path('/tmp/candidacy-cleanup-audit')
OUT.mkdir(exist_ok=True)
PDF=ROOT/'Candidacy slides.pdf'

def white(im):
    rgba=im.convert('RGBA')
    bg=Image.new('RGBA',rgba.size,'white')
    return Image.alpha_composite(bg,rgba).convert('RGB')

def features(im):
    im=white(im)
    return im.size, hashlib.sha256(im.tobytes()).hexdigest(), np.asarray(im.resize((48,48))).astype(float)

def main():
    candidates=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_symlink() or not p.is_file() or 'archive' in p.relative_to(ROOT).parts:
            continue
        if p.suffix.lower() in ('.png','.jpg','.jpeg'):
            try:
                with Image.open(p) as im:
                    size,digest,thumb=features(im)
                candidates.append((str(p.relative_to(ROOT)),size,digest,thumb))
                if len(candidates)%25==0:
                    print('SCANNED',len(candidates),str(p.relative_to(ROOT)),flush=True)
            except Exception as ex:
                print('IMAGE_ERROR',p,repr(ex),flush=True)
    print('CANDIDATES',len(candidates),flush=True)
    doc=pymupdf.open(PDF)
    pages=[]
    for i,page in enumerate(doc):
        images=[]
        seen=set()
        for img in page.get_image_info(xrefs=True):
            xref=img['xref']
            if not xref or xref in seen:
                continue
            seen.add(xref)
            data=doc.extract_image(xref)
            with Image.open(io.BytesIO(data['image'])) as im:
                # Keynote may store the transparency as a separate soft mask.
                if data.get('smask'):
                    smask=doc.extract_image(data['smask'])
                    alpha=Image.open(io.BytesIO(smask['image'])).convert('L')
                    im=im.convert('RGBA'); im.putalpha(alpha)
                size,digest,thumb=features(im)
            exact=[p for p,s,h,t in candidates if h==digest and s==size]
            scored=[]
            ratio=size[0]/size[1]
            for p,s,h,t in candidates:
                if abs(np.log((s[0]/s[1])/ratio)) > .08:
                    continue
                rmse=float(np.sqrt(np.mean((t-thumb)**2)))
                scored.append((rmse,p))
            scored.sort()
            imagepath=OUT/f'embedded_p{i+1:02}_x{xref}.png'
            imagepath.write_bytes(data['image'])
            images.append({'xref':xref,'size':size,'bbox':img['bbox'],'exact':exact,
                           'nearest':scored[:5], 'extracted':str(imagepath)})
        pages.append({'page':i+1,'text':page.get_text(),'images':images})
        print('PAGE',i+1,'images',len(images),'exact',sum(bool(x['exact']) for x in images),flush=True)
    report={'pdf':str(PDF),'pdf_sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
            'pages':pages,'candidate_images':len(candidates)}
    (OUT/'image_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    sheet=Image.new('RGB',(1600,340*8),'#e9edf1')
    draw=ImageDraw.Draw(sheet)
    for i,page in enumerate(doc):
        thumb=Image.open(OUT/f'page_{i+1:02}.png').convert('RGB')
        thumb.thumbnail((390,300))
        x=(i%4)*400+(400-thumb.width)//2
        y=(i//4)*340+25
        sheet.paste(thumb,(x,y))
        draw.text(((i%4)*400+10,(i//4)*340+6),f'PAGE {i+1}',fill='black')
    sheet.save(OUT/'deck_contact_sheet.jpg',quality=90)

if __name__=='__main__':
    main()
