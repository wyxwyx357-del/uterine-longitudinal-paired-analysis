"""Frequency analysis of saved V1 (A) tracks, never B. No interpolation."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from frequency import segments,fit_frequency

def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(path,out,case,spacing):
    out.mkdir(parents=True,exist_ok=True)
    with np.load(path,allow_pickle=False) as z:
        t=z['time_s']; p=z['corrected_A']; pv=z['pose_valid_A']; valid=z['target_valid']; epoch=z['epoch_A']; axis=z['inward_axis']; j=int(z['target_index']); k=int(z['neighbor_index'])
    if len(t)<2 or not np.isfinite(t).all() or not (np.diff(t)>0).all():raise ValueError('Invalid timestamps')
    if spacing<=0:raise ValueError('Invalid calibration')
    rows=[];curves=[]
    for signal in ['target','neighbor_distance']:
        mask=pv & valid[:,j] & np.isfinite(p[:,j]).all(1)
        if signal=='neighbor_distance':mask &= valid[:,k] & np.isfinite(p[:,k]).all(1)
        parts=segments(mask,epoch,np.zeros(len(t),bool),t)
        primary=max(parts,key=lambda ab:(t[ab[1]-1]-t[ab[0]],-ab[0])) if parts else None
        if not parts:
            rows.append(dict(case=case,signal=signal,segment='',primary=True,start_frame='',end_frame='',duration_s=0,status='not_evaluable',reason='no_continuous_valid_segment',frequency_cpm='',period_s='',range_mm='',fitted_amplitude_mm='',diagnostic_best_cpm='',score='',lag_correlation='',half_lag_correlation=''))
        for sid,(a,b) in enumerate(parts):
            y=((p[a:b,j]-p[a,j])@axis if signal=='target' else np.linalg.norm(p[a:b,j]-p[a:b,k],axis=1)-np.linalg.norm(p[a,j]-p[a,k]))
            # Run the exact frozen estimator in its original pixel units.
            fit,grid,scores,trend,fitted=fit_frequency(t[a:b],y)
            amp=''
            if fit['candidate_cpm'] is not None:
                tt=t[a:b]-t[a:b].mean(); phase=2*np.pi*tt*fit['candidate_cpm']/60
                beta=np.linalg.lstsq(np.column_stack([np.ones(len(tt)),tt,np.sin(phase),np.cos(phase)]),y,rcond=None)[0]
                amp=float(np.hypot(*beta[-2:])*spacing)
            passed=fit['status']=='exploratory_repetition_candidate'
            status='not_evaluable' if fit['reason']=='continuous_segment_too_short_for_two_cycles_in_search_range' else fit['status']
            rows.append(dict(case=case,signal=signal,segment=sid,primary=(a,b)==primary,start_frame=a,end_frame=b-1,duration_s=float(t[b-1]-t[a]),status=status,reason=fit['reason'],frequency_cpm=fit['candidate_cpm'] if passed else '',period_s=fit['candidate_period_s'] if passed else '',range_mm=float(np.ptp(y)*spacing),fitted_amplitude_mm=amp,diagnostic_best_cpm=fit['candidate_cpm'],score=fit['score'],lag_correlation=fit['lag_correlation'],half_lag_correlation=fit['half_lag_correlation']))
            for i in range(a,b):curves.append(dict(frame=i,time_s=float(t[i]),signal=signal,segment=sid,displacement_mm=float(y[i-a]*spacing)))
    write_csv(out/'segments.csv',rows);write_csv(out/'curves_mm.csv',curves)
    primary=[r for r in rows if r['primary']];write_csv(out/'primary_summary.csv',primary)
    (out/'run_metadata.json').write_text(json.dumps(dict(case=case,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),mm_per_processed_pixel=spacing,calibration_source='user-provided verified DICOM conversion, isotropic',method='strict V1 A only',frequency_fit_units='processed_pixel; mm conversion after fit',numpy_version=np.__version__),ensure_ascii=False,indent=2))
    return primary

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--case',required=True);ap.add_argument('--mm-per-pixel',type=float,required=True);a=ap.parse_args()
    print(json.dumps(analyze(a.input,a.out,a.case,a.mm_per_pixel),ensure_ascii=False,indent=2))
