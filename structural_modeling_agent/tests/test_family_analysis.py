"""Comparison plots remain readable as completed homolog ensembles accumulate."""
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from structural_modeling_agent.family_analysis import _short_name,comparison_figure


def test_display_prefers_enzyme_symbols_without_inventing_family_members():
    assert _short_name('Long enzyme description',['xyzB'])=='XyzB'
    assert _short_name('Unreviewed enzyme annotation',literature='Named enzyme')=='Named enzyme'
    assert _short_name('Mixed enzyme family protein')=='Unassigned enzyme'
    assert _short_name('Descriptive protein XyzB, unrelated family annotation')=='XyzB'


def test_long_labels_have_dedicated_rows_and_cannot_collapse_metric_axes():
    rows=[{'accession':'synthetic_'+str(i),'weighted_attack_angle_deg':170-i,
           'harmonic_distance_A':3+i,'numerical_profile':'exact'} for i in range(35)]
    labels={r['accession']:'Named enzyme '+str(i)+' · organism with a long descriptive name'
            for i,r in enumerate(rows)}
    original=copy.deepcopy(rows)
    fig=comparison_figure(rows,labels,1)
    fig.canvas.draw()
    axes=fig.axes
    assert all(axis.get_position().width>.2 for axis in axes)
    tick_labels=[value.get_text() for value in axes[0].get_yticklabels()]
    assert len(tick_labels)==35 and all('Named enzyme' in text for text in tick_labels)
    assert all('synthetic_' not in text for text in tick_labels)
    assert rows==original
    assert axes[1].get_xscale()=='log'
    plt.close(fig)
