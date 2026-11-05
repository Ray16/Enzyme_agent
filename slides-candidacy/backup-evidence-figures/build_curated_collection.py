"""Rebuild only the 14 retained standalone backup figures."""
from pathlib import Path
import os,subprocess,sys,shutil
import fitz
HERE=Path(__file__).resolve().parent
SLIDES=HERE.parent
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]='1'


def main():
    stem=HERE/'amp_adp_comparison'
    subprocess.run(['pdflatex','-interaction=batchmode','-halt-on-error',
                    f'-output-directory={HERE}',str(stem.with_suffix('.tex'))],
                    check=True,capture_output=True,text=True)
    with fitz.open(stem.with_suffix('.pdf')) as d:
        assert len(d)==1
        d[0].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(stem.with_suffix('.png'))
        svg=d[0].get_svg_image(text_as_path=False)
        stem.with_suffix('.svg').write_text('\n'.join(x.rstrip() for x in svg.splitlines())+'\n')
    for script in [HERE/'build_quantitative_figures.py',HERE/'build_method_figures.py',
                   SLIDES/'framework-figures/build_figures.py',
                   SLIDES/'adp-attack-geometry/compose_figure.py']:
        subprocess.run([sys.executable,str(script)],check=True)
    shutil.copy2(SLIDES/'attack-angle/AS9_attack_geometry.png',HERE/'AS9_attack_geometry.png')
    subprocess.run([sys.executable,str(HERE/'catalog_figures.py')],check=True)


if __name__=='__main__':main()
