"""Generate figures for enzyme_agent_workflow slides."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json, os

FIGS = os.path.join(os.path.dirname(__file__), 'figs')
RUNS = '/nfs/lambda_stor_01/homes/rzhu/Enzyme_agent/enzyme_discovery_agent/runs'

# Color palette matching the beamer theme
BLUE = '#144272'
ACCENT_BLUE = '#2980B9'
ORANGE = '#E67E22'
GREEN = '#27AE60'
PURPLE = '#8E44AD'
GRAY = '#34495E'
LIGHT = '#ECF0F1'

# ── Figure 1: Per-run discovery bar chart ──────────────────
def fig_discovery_bars():
    runs = ['Aliphatic\namide', 'Nylon-66\nhydrolysis', 'PPTA\n(para)', 'PMIA\n(meta)']
    discovered = [422, 298, 832, 250]
    screened   = [49,  54,  76,  52]
    evaluated  = [27,  34,  22,  17]
    colors_d = [ACCENT_BLUE, ORANGE, GREEN, PURPLE]
    colors_e = [BLUE, '#C0392B', '#1E8449', '#6C3483']

    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(runs))
    w = 0.35

    bars1 = ax.bar(x - w/2, discovered, w, color=[c + '40' for c in colors_d],
                   edgecolor=colors_d, linewidth=1.2, label='Discovered')
    bars2 = ax.bar(x + w/2, evaluated, w, color=colors_d,
                   edgecolor=colors_d, linewidth=1.2, label='Evaluated')

    for bar, val in zip(bars1, discovered):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, evaluated):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontsize=10)
    ax.set_ylabel('Candidates', fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, 950)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Candidates Discovered vs. Evaluated per Run', fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'discovery_bars.pdf'), bbox_inches='tight')
    plt.close(fig)


# ── Figure 2: Tool usage breakdown (stacked bar) ──────────
def fig_tool_usage():
    runs = ['R1: Aliphatic', 'R2: Nylon-66', 'R3: PPTA', 'R4: PMIA']
    db   = [62, 48, 40, 86]   # search_uniprot + kegg + interpro + fetch
    lit  = [23, 34, 21, 23]   # pubmed + openalex + deep + extract
    eval = [40, 49, 33, 32]   # evaluate + screen + notebook
    chem = [3,  3,  4,  4]    # analyze + validate + rxn

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(runs))
    w = 0.55

    b1 = ax.bar(x, db, w, label='Database (47%)', color=ACCENT_BLUE)
    b2 = ax.bar(x, eval, w, bottom=db, label='Evaluation (30%)', color=GREEN)
    b3 = ax.bar(x, lit, w, bottom=[d+e for d,e in zip(db,eval)], label='Literature (20%)', color=ORANGE)
    b4 = ax.bar(x, chem, w, bottom=[d+e+l for d,e,l in zip(db,eval,lit)], label='Chemistry (3%)', color=PURPLE)

    for i, total in enumerate([128, 134, 98, 145]):
        ax.text(i, total + 2, str(total), ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(runs, fontsize=9)
    ax.set_ylabel('Tool Calls', fontsize=11)
    ax.legend(fontsize=8.5, loc='upper right', ncol=2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Tool Calls by Category per Run', fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'tool_usage.pdf'), bbox_inches='tight')
    plt.close(fig)


# ── Figure 3: Enzyme family landscape (bubble/dot chart) ──
def fig_family_landscape():
    fig, ax = plt.subplots(figsize=(7, 3.2))

    # Data: (run_index, family_label, n_candidates_evaluated, color)
    data = [
        # Run 1 families
        (0, 'NIS synthetases', 9, ACCENT_BLUE),
        (0, 'Gln syn. fold', 3, ACCENT_BLUE),
        (0, 'ATP-grasp', 5, ACCENT_BLUE),
        (0, 'Murein ligases', 3, ACCENT_BLUE),
        # Run 2 families
        (1, 'Ntn hydrolases', 6, ORANGE),
        (1, 'β-lactamase', 4, ORANGE),
        (1, 'Amidase sig.', 6, ORANGE),
        (1, 'Cutinases', 5, ORANGE),
        (1, 'Subtilisins', 3, ORANGE),
        # Run 3 families
        (2, 'Aryl-AMP ligases', 5, GREEN),
        (2, 'Aminocoumarin', 4, GREEN),
        (2, 'IucA/IucC', 3, GREEN),
        (2, 'Gln syn. fold', 1, GREEN),
        # Run 4 families
        (3, 'Anthranilate-CoA', 2, PURPLE),
        (3, 'Aryl-AMP ligases', 4, PURPLE),
        (3, 'Benzoate-CoA', 4, PURPLE),
        (3, 'Aminocoumarin', 1, PURPLE),
    ]

    y_labels = sorted(set(d[1] for d in data), key=lambda x: -max(d[2] for d in data if d[1]==x))
    y_map = {label: i for i, label in enumerate(y_labels)}

    for run_idx, label, size, color in data:
        ax.scatter(run_idx, y_map[label], s=size*50, c=color, alpha=0.6, edgecolors=color, linewidth=1.2)
        if size >= 4:
            ax.text(run_idx, y_map[label], str(size), ha='center', va='center', fontsize=7, fontweight='bold', color='white')

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Aliphatic', 'Nylon-66', 'PPTA', 'PMIA'], fontsize=9)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlim(-0.5, 3.5)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Enzyme Families Explored per Run', fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'family_landscape.pdf'), bbox_inches='tight')
    plt.close(fig)


# ── Figure 4: Report excerpt (formatted table image) ──────
def fig_report_excerpt():
    """Create a styled table showing a report excerpt from the PPTA run."""
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.axis('off')

    col_labels = ['Rank', 'Gene', 'Organism', 'EC', 'PDB', 'Tier']
    cell_data = [
        ['1', 'DhbE', 'B. subtilis', '6.2.1.71', '1MD9, 1MDB, 1MDF', '1'],
        ['2', 'EntE', 'E. coli K12', '6.3.2.14', '3RG2, 4IZ6, +4 more', '1'],
        ['3', 'NovL', 'S. niveus', '6.3.1.15', 'AlphaFold only', '1'],
        ['4', 'AtdA1', 'Acinetobacter sp.', '6.3.1.18', 'AlphaFold only', '1'],
        ['5', 'AsbC', 'B. anthracis', '6.2.1.62', 'AlphaFold only', '2'],
        ['6', 'MbtA', 'M. tuberculosis', '6.2.1.-', 'AlphaFold only', '2'],
    ]

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor(BLUE)
        cell.set_text_props(color='white', fontweight='bold')

    # Style tier 1 rows
    for i in range(1, 5):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor('#D5F5E3')

    for i in range(5, 7):
        for j in range(len(col_labels)):
            table[i, j].set_facecolor('#EBF5FB')

    ax.set_title('Run 3 (PPTA) — Agent-Generated Candidate Table Excerpt',
                 fontsize=11, fontweight='bold', pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'report_excerpt.pdf'), bbox_inches='tight')
    plt.close(fig)


# ── Figure 5: Cross-run Venn/mapping diagram ──────────────
def fig_cross_run():
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Substrate class boxes (left)
    for i, (label, color, y) in enumerate([
        ('Aliphatic\nligation', ACCENT_BLUE, 5.5),
        ('Aromatic\nligation', ORANGE, 3.5),
        ('Amide\nhydrolysis', GREEN, 1.5),
    ]):
        rect = mpatches.FancyBboxPatch((0.3, y-0.5), 2.0, 1.0,
            boxstyle="round,pad=0.1", facecolor=color+'30', edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(1.3, y, label, ha='center', va='center', fontsize=9, fontweight='bold', color=GRAY)

    # Enzyme family boxes (right)
    families = [
        ('NIS synthetases', 6.0), ('Gln syn. fold', 5.0),
        ('ANL superfamily', 3.5),
        ('Ntn hydrolases', 2.0), ('β-lactamase fold', 1.0),
    ]
    for label, y in families:
        rect = mpatches.FancyBboxPatch((6.5, y-0.35), 3.0, 0.7,
            boxstyle="round,pad=0.1", facecolor=LIGHT, edgecolor=GRAY, linewidth=1)
        ax.add_patch(rect)
        ax.text(8.0, y, label, ha='center', va='center', fontsize=8.5, color=GRAY)

    # Arrows
    arrow_style = dict(arrowstyle='->', color=ACCENT_BLUE, lw=1.8)
    ax.annotate('', xy=(6.5, 6.0), xytext=(2.3, 5.7), arrowprops=arrow_style)
    ax.annotate('', xy=(6.5, 5.0), xytext=(2.3, 5.3), arrowprops=dict(arrowstyle='->', color=ACCENT_BLUE, lw=1.8))
    ax.annotate('', xy=(6.5, 3.5), xytext=(2.3, 3.5), arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.8))
    ax.annotate('', xy=(6.5, 2.0), xytext=(2.3, 1.7), arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.8))
    ax.annotate('', xy=(6.5, 1.0), xytext=(2.3, 1.3), arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.8))

    # Key point
    ax.text(5.0, 0.1, 'Zero family overlap between aliphatic and aromatic runs',
            ha='center', va='center', fontsize=8, fontstyle='italic', color='#C0392B', fontweight='bold')

    ax.set_title('Substrate Class → Enzyme Family (Agent-Discovered)', fontsize=11, fontweight='bold', pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'cross_run_map.pdf'), bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    fig_discovery_bars()
    print('  -> discovery_bars.pdf')
    fig_tool_usage()
    print('  -> tool_usage.pdf')
    fig_family_landscape()
    print('  -> family_landscape.pdf')
    fig_report_excerpt()
    print('  -> report_excerpt.pdf')
    fig_cross_run()
    print('  -> cross_run_map.pdf')
    print('Done.')
