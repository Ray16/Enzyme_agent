"""Versioned legacy profile with an explicitly selected generation batch size."""
import os,sys
from pathlib import Path
base=Path('/nfs/lambda_stor_01/homes/rzhu/Tool/RFD4-Proteina-compat')
sys.path.insert(0,str(base))
import legacy_gpu
batch=int(os.environ['RFD4_BATCH_SIZE'])
if batch<1 or batch>16:raise ValueError('Invalid batch size')
legacy_gpu.PROFILE['batch_size']=str(batch)
legacy_gpu.main()
