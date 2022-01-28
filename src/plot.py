"""
Functions to organize the data and plot
"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform


plt.rcParams["font.family"] = "monospace"

model_name_transformer = {
    'bonito': 'Bonito',
    'catcaller': 'CATCaller',
    'sacall': 'SACall',
    'urnano': 'URNano',
    'mincall': 'MinCall',
    'halcyon': 'Halcyon',
    'causalcall': 'Causalcall',
    'halcyonmod': 'HalcyonMod',
    'bonitofwd': 'BonitoFwd',
    'bonitorev': 'BonitoRev',
    'catcaller': 'CATCaller',
    'sacall': 'SACall',
    'urnano': 'URNano',
    'lstm1': 'LSTM1',
    'lstm3': 'LSTM3',
    'lstm5': 'LSTM5',
    'ctc': 'CTC',
    'crf': 'CRF',
    's2s': 'Seq2Seq'
}

metric_name_transformer = {
    'match_rate': 'Match rate',
    'mismatch_rate': 'Mismatch rate',
    'deletion_rate': 'Deletion rate',
    'insertion_rate': 'Insertion rate',
    'phred_mean_correct': 'PhredQ Mean Correct',
    'phred_mean_error': 'PhredQ Mean Error',
    'phred_error_test': 'PhredQ Correct vs Error Test (p-value)',
    'phred_mean_correct_error_ratio': 'PhredQ Correct/Error',
    'total_homo_error_rate' : 'Homopolymer Total',
    'A_homo_error_rate' : 'Homopolymer A', 
    'C_homo_error_rate' : 'Homopolymer C', 
    'G_homo_error_rate' : 'Homopolymer G', 
    'T_homo_error_rate' : 'Homopolymer T',
    'total_divergence': 'Total Divergence',
    'A_divergence': 'A Divergence',
    'C_divergence': 'C Divergence',
    'G_divergence': 'G Divergence',
    'T_divergence': 'T Divergence'
}

def transform_name(name):

    name_components = name.split('_')
    if len(name_components) > 1:
        new_name = ''
        new_name += model_name_transformer[name_components[0]]
        new_name += ' - '
        new_name += model_name_transformer[name_components[1]]
    else:
        new_name = model_name_transformer[name_components[0]]

    return new_name

def basecalled_reads_counts(df, output_dir_tables = None, output_dir_plots = None, **kwargs):

    plot_df = {
        'model': list(),
        'counts': list(),
        'reason': list()
    }

    for model in set(df['model']):
        st = df['model'].searchsorted(model, 'left')
        nd = df['model'].searchsorted(model, 'right')

        subdf = df[st:nd]
        counts = subdf["comment"].value_counts(normalize = False)
        for k in counts.keys():
            plot_df['model'].append(transform_name(model))
            plot_df['counts'].append(counts[k])
            plot_df['reason'].append(k)
    plot_df = pd.DataFrame(plot_df)
    pdf = plot_df.sort_values('model', ascending=False)    

    
    pdf_to_write = pdf.pivot(index='model', columns='reason', values='counts')
    pdf_to_write = pdf_to_write.rename_axis('model').reset_index()

    if output_dir_tables is not None:
        pdf_to_write.to_csv(os.path.join(output_dir_tables, 'number_of_reads.csv'), header = True, index = False)


    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, **kwargs)

    ax1.barh(y = pdf.loc[pdf['reason'] == 'pass','model'], width= pdf.loc[pdf['reason'] == 'pass','counts'], color = 'grey', edgecolor = 'black')
    ax1.set_title('Basecalled and mapped')
    ax2.barh(y = pdf.loc[pdf['reason'] == 'failed mapping','model'], width= pdf.loc[pdf['reason'] == 'failed mapping','counts'], color = 'grey', edgecolor = 'black')
    ax2.set_title('Failed mapping')
    ax3.barh(y = pdf.loc[pdf['reason'] == 'no prediction','model'], width= pdf.loc[pdf['reason'] == 'no prediction','counts'], color = 'grey', edgecolor = 'black')
    ax3.set_title('Failed basecalling')

    f.suptitle('Number of reads')
    f.tight_layout()

    if output_dir_plots is not None:
        plot_file = os.path.join(output_dir_plots, 'number_of_reads')
        f.savefig(plot_file + '.pdf', transparent=True, dpi=200, bbox_inches="tight")
        f.savefig(plot_file + '.png', facecolor='white', transparent=False, dpi=200, bbox_inches="tight")

    return f, pdf_to_write

def main_rates(df, normalization_column = 'len_reference', output_dir_tables = None, output_dir_plots = None, **kwargs):

    bases = ['A', 'C', 'G', 'T']
    match_columns = list()
    mismatch_columns = list()
    insertion_columns = list()
    deletion_columns = list()
    for b1 in bases:
        for b2 in bases + ['-']:
            for b3 in bases + ['-']:
                for b4 in bases:
                    if b2 == '-' and b3 == '-':
                        continue
                    if b2 == b3:
                        match_columns.append(b1 + b2 + '>' + b3 + b4)
                    else:
                        if b2 == '-':
                            deletion_columns.append(b1 + b2 + '>' + b3 + b4)
                        elif b3 == '-':
                            insertion_columns.append(b1 + b2 + '>' + b3 + b4)
                        else:
                            mismatch_columns.append(b1 + b2 + '>' + b3 + b4)
    
    df['matches'] = df[match_columns].sum(axis=1)
    df['mismatches'] = df[mismatch_columns].sum(axis=1)
    df['insertions'] = df[insertion_columns].sum(axis=1)
    df['deletions'] = df[deletion_columns].sum(axis=1)
    df['len_alignment'] = df[['matches', 'mismatches', 'insertions', 'deletions' ]].sum(axis=1)
    df['len_alignment'] += 2

    df['match_rate'] = df['matches']/df[normalization_column]
    df['mismatch_rate'] = df['mismatches']/df[normalization_column]
    df['insertion_rate'] = df['insertions']/df[normalization_column]
    df['deletion_rate'] = df['deletions']/df[normalization_column]

    pdf_to_write = {
        "Model":list(),
        "match_rate (mean)":list(),
        "match_rate (median)":list(),
        "match_rate (std)":list(),
        "mismatch_rate (mean)":list(),
        "mismatch_rate (median)":list(),
        "mismatch_rate (std)":list(),
        "insertion_rate (mean)":list(),
        "insertion_rate (median)":list(),
        "insertion_rate (std)":list(),
        "deletion_rate (mean)":list(),
        "deletion_rate (median)":list(),
        "deletion_rate (std)":list(),
    }

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, **kwargs)

    axes = [ax1, ax2, ax3, ax4]
    metrics = ['match_rate', 'mismatch_rate', 'deletion_rate', 'insertion_rate']
    for ax, metric in zip(axes, metrics):

        model_name_sorted = np.unique(df['model'])[::-1]
        medians = list()
        for i, model_name in enumerate(model_name_sorted):
            plot_df = df[df['model'] == model_name]
            x = np.array(plot_df[metric])
            x = x[~np.isnan(x)]
            
            medians.append(np.median(x))
        medians = np.array(medians)

        if metric == 'match_rate':
            top_3_models = model_name_sorted[np.argsort(medians)[-3:]]
            medal_colors = ['#CD7F32', '#C0C0C0', '#FFD700'] # bronze, silver, gold
        else:
            top_3_models = model_name_sorted[np.argsort(medians)[:3]]
            medal_colors = ['#FFD700', '#C0C0C0', '#CD7F32'] # gold, silver, bronze 

        ticks_names = list()
        for i, model_name in enumerate(model_name_sorted):

            plot_df = df[df['model'] == model_name]
            ticks_names.append(transform_name(model_name))

            x = np.array(plot_df[metric])
            x = x[~np.isnan(x)]

            if metric == 'match_rate':
                pdf_to_write['Model'].append(transform_name(model_name))
            pdf_to_write[metric+' (mean)'].append(np.mean(x))
            pdf_to_write[metric+' (median)'].append(np.median(x))
            pdf_to_write[metric+' (std)'].append(np.std(x))

            if model_name in top_3_models:
                color = medal_colors[int(np.where(top_3_models == model_name)[0])]
                ax.axvline(np.median(x), color = color)
            else:
                color = 'white'
            
            bp = ax.boxplot(x, positions = [i + 1], showfliers=False, widths = [0.6], vert = False, patch_artist = True)
            
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color='black',  linewidth=1.5)
            plt.setp(bp['boxes'], facecolor = color)

        ax.set_xlabel(metric_name_transformer[metric])


    ax1.set_yticks(ticks = np.arange(1, len(set(df['model'])) + 1, 1))
    ax1.set_yticklabels(labels = ticks_names)
    plt.tight_layout()
    plt.show()

    if output_dir_plots is not None:
        plot_file = os.path.join(output_dir_plots, 'main_rates')
        f.savefig(plot_file + '.pdf', transparent=True, dpi=200, bbox_inches="tight")
        f.savefig(plot_file + '.png', facecolor='white', transparent=False, dpi=200, bbox_inches="tight")

    pdf_to_write = pd.DataFrame(pdf_to_write)
    if output_dir_tables is not None:
        pdf_to_write.to_csv(os.path.join(output_dir_tables, 'main_rates.csv'), header = True, index = False)

    return f, pdf_to_write

def phredq_distributions(df, output_dir_tables = None, output_dir_plots = None, **kwargs):

    df['phred_mean_correct_error_ratio'] = df['phred_mean_correct'] - df['phred_mean_error']
    pdf_to_write = {
        "Model":list(),
        "phredq_difference (mean)":list(),
        "phredq_difference (median)":list(),
        "phredq_difference (std)":list(),
    }

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, **kwargs)

    axes = [ax1, ax2, ax3]
    metrics = ['phred_mean_correct', 'phred_mean_error', 'phred_mean_correct_error_ratio']
    for ax, metric in zip(axes, metrics):

        model_name_sorted = np.unique(df['model'])[::-1]
        medians = list()
        for i, model_name in enumerate(model_name_sorted):
            plot_df = df[df['model'] == model_name]
            x = np.array(plot_df[metric])
            x = x[~np.isnan(x)]
            
            medians.append(np.median(x))
        medians = np.array(medians)

        if metric == 'phred_mean_correct' or metric == 'phred_mean_correct_error_ratio':
            top_3_models = model_name_sorted[np.argsort(medians)[-3:]]
            medal_colors = ['#CD7F32', '#C0C0C0', '#FFD700'] # bronze, silver, gold
        else:
            top_3_models = model_name_sorted[np.argsort(medians)[:3]]
            medal_colors = ['#FFD700', '#C0C0C0', '#CD7F32'] # gold, silver, bronze 

        ticks_names = list()
        for i, model_name in enumerate(model_name_sorted):

            plot_df = df[df['model'] == model_name]
            ticks_names.append(transform_name(model_name))

            x = np.array(plot_df[metric])
            x = x[~np.isnan(x)]

            if model_name in top_3_models:
                color = medal_colors[int(np.where(top_3_models == model_name)[0])]
                ax.axvline(np.median(x), color = color)
            else:
                color = 'white'

            if metric == 'phred_mean_correct_error_ratio':
                pdf_to_write['Model'].append(transform_name(model_name))
                pdf_to_write['phredq_difference (mean)'].append(np.mean(x))
                pdf_to_write['phredq_difference (median)'].append(np.median(x))
                pdf_to_write['phredq_difference (std)'].append(np.std(x))
            
            bp = ax.boxplot(x, positions = [i + 1], showfliers=False, widths = [0.6], vert = False, patch_artist = True)
            
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color='black',  linewidth=1.5)
            plt.setp(bp['boxes'], facecolor = color)

        ax.set_xlabel(metric_name_transformer[metric])


    ax1.set_yticks(ticks = np.arange(1, len(set(df['model'])) + 1, 1))
    ax1.set_yticklabels(labels = ticks_names)
    plt.tight_layout()
    plt.show()


    if output_dir_plots is not None:
        plot_file = os.path.join(output_dir_plots, 'phredq_differences')
        f.savefig(plot_file + '.pdf', transparent=True, dpi=200, bbox_inches="tight")
        f.savefig(plot_file + '.png', facecolor='white', transparent=False, dpi=200, bbox_inches="tight")

    pdf_to_write = pd.DataFrame(pdf_to_write)
    if output_dir_tables is not None:
        pdf_to_write.to_csv(os.path.join(output_dir_tables, 'phredq_differences.csv'), header = True, index = False)

    return f, pdf_to_write


def calculate_signatures(model_df):

    bases = ["A", "C", "G", "T"]
    types = ["Match", "Missmatch_A", "Missmatch_C", "Missmatch_G", "Missmatch_T", "Missmatch", "Insertion", "Deletion"]
    signature_columns = model_df.columns[model_df.columns.str.contains('>')]

    signatures = list()
    for b in bases:
        for b1 in bases:
            for b2 in bases:
                for t in types:
                    signatures.append({'Base': b, 'Context': b1+b+b2, 'Error': t, "Count": 0, "Rate": 0.0, "k": b+b1+b2+t})
    signatures_df = pd.DataFrame(signatures)

    for s_col in signature_columns:
        k1 = s_col[1] if s_col[1] != '-' else s_col[3]
        k2 = s_col[0] + s_col[-1]
        if s_col[1] == '-':
            k3 = "Deletion"
        elif s_col[3] == '-':
            k3 = "Insertion"
        elif s_col[1] == s_col[3]:
            k3 = "Match"
        else:
            k3 = "Missmatch"
        p = np.where(signatures_df["k"] == k1+k2+k3)[0]
        signatures_df.iloc[p, 3] += np.sum(model_df[s_col])
        if k3 == 'Missmatch':
            k3 = "Missmatch_" + s_col[3]
            p = np.where(signatures_df["k"] == k1+k2+k3)[0]
            signatures_df.iloc[p, 3] += np.sum(model_df[s_col])

    for b in np.unique(signatures_df["Base"]):
        sub_df = signatures_df[signatures_df["Base"] == b]
        for c in np.unique(signatures_df["Context"]):
            subsub_df = sub_df[sub_df["Context"] == c]
            subsub_df = subsub_df[subsub_df["Error"].isin(["Match", "Missmatch_A", "Missmatch_C", "Missmatch_G", "Missmatch_T", "Deletion", 'Insertion'])]
            t = np.sum(subsub_df["Count"])
            signatures_df.loc[subsub_df.index, 'Rate'] = signatures_df.loc[subsub_df.index, 'Count']/t

    d_list = list()
    for b in np.unique(signatures_df["Base"]):
        sub_df = signatures_df[signatures_df["Base"] == b]
        sub_df = sub_df[sub_df["Error"].isin(["Match", "Missmatch_A", "Missmatch_C", "Missmatch_G", "Missmatch_T", "Deletion", 'Insertion'])]
        total_bases = np.sum(sub_df['Count'])
        for c in np.unique(signatures_df["Error"]):
            subsub_df = sub_df[sub_df["Error"] == c]
            total_errors = np.sum(subsub_df['Count'])
            d = {'Base': 'Grouped', 'Context': 'N'+b+'N', 'Error': c, 'Count': total_errors, 'Rate':total_errors/total_bases, 'k': b+c}
            d_list.append(d)

    total_bases = np.sum(signatures_df['Count'])
    for c in np.unique(signatures_df["Error"]):
        sub_df = signatures_df[signatures_df["Error"] == c]
        total_errors = np.sum(sub_df['Count'])
        d = {'Base': 'Grouped', 'Context': 'NNN', 'Error': c, 'Count': total_errors, 'Rate':total_errors/total_bases, 'k': b+c}
        d_list.append(d)

    signatures_df = signatures_df.append(pd.DataFrame(d_list))
    return signatures_df

def signatures_plot(df, model_name, output_dir_plots = None, **kwargs):


    model_name = 'mincall_urnano_ctc_True_2000'
    model_df = df[df['model'] == model_name]

    signatures_df = calculate_signatures(model_df)
    bases = ['A', 'C', 'G', 'T']

    f, axs = plt.subplots(2, 5, sharey=False, gridspec_kw={'width_ratios': [1, 1, 1, 1, 0.3]}, **kwargs)
    for i, b in enumerate(bases + ['Grouped']):
        plot_df = signatures_df[signatures_df["Base"] == b]
        plot_df = plot_df[plot_df["Error"].isin(["Missmatch_A", "Missmatch_C", "Missmatch_G", "Missmatch_T", "Deletion", "Insertion"])]
        plot_df = plot_df.sort_values('Error')
        sns.histplot(x = "Context", hue = "Error", weights='Rate', data = plot_df, 
                    multiple='stack', ax = axs[0, i])
        axs[0, i].set_ylabel('Fraction')
        axs[0, i].set_xlabel(None)
        axs[0, i].set_title(b)
        axs[0, i].set_ylim(0, 0.55)
        for tick in axs[0, i].get_xticklabels():
            tick.set_rotation(90)
        if i != 4:
            axs[0, i].get_legend().remove()
        else:
            legend = axs[0, i].get_legend()
            legend.set_title(None)
            legend.texts[0].set_text("Deletion")
            legend.texts[1].set_text("Insertion")
            legend.texts[2].set_text("Mismatch: A")
            legend.texts[3].set_text("Mismatch: C")
            legend.texts[4].set_text("Mismatch: G")
            legend.texts[5].set_text("Mismatch: T")
            legend.set_bbox_to_anchor((1, 0.75))

    for i, b in enumerate(bases):
        plot_df = signatures_df[signatures_df["Base"] == b]
        plot_df = plot_df[plot_df["Error"].isin(["Missmatch_A", "Missmatch_C", "Missmatch_G", "Missmatch_T", "Deletion", "Insertion"])]
        plot_df = plot_df.sort_values('Error')
        sns.histplot(x = "Context", weights='Count', data = plot_df, ax = axs[1, i], color = 'grey')
        axs[1, i].set_ylabel('Counts')
        axs[1, i].set_xlabel(None)
        axs[1, i].set_title(b)
        for tick in axs[1, i].get_xticklabels():
            tick.set_rotation(90)
            
    f.delaxes(axs[1, 4])
    f.tight_layout()

    if output_dir_plots is not None:
        plot_file = os.path.join(output_dir_plots, 'signatures', model_name)
        f.savefig(plot_file + '.pdf', transparent=True, dpi=200, bbox_inches="tight")
        f.savefig(plot_file + '.png', facecolor='white', transparent=False, dpi=200, bbox_inches="tight")

    return f


def randomness_divergence(df, output_dir_tables = None, output_dir_plots = None, **kwargs):

    randomness_df = {
        'model': list(),
        'divergence': list(),
        'base': list()
    }

    for model_name in np.unique(df['model']):

        signatures_df = calculate_signatures(df[df['model'] == model_name])

        signatures_df = signatures_df.sort_values(['Base', 'Context', 'Error'])
        groups = ['NAN', 'NCN', 'NGN', 'NTN']
        for group in groups:
            subdf = signatures_df[signatures_df['Context'] == group]
            subdf = subdf[subdf['Error'] != 'Missmatch']
            avg_rates = np.array(subdf['Rate'])

            base = list(group)[1]
            subdf = signatures_df[signatures_df['Base'] == base]
            subdf = subdf[subdf['Error'] != 'Missmatch']

            for context in np.unique(subdf['Context']):
                subsubdf = subdf[subdf['Context'] == context]
                rates = np.array(subsubdf['Rate'])

                if len(rates) == 0:
                    continue

                randomness_df['model'].append(model_name)
                randomness_df['base'].append(base + '_divergence')
                randomness_df['divergence'].append(jensenshannon(avg_rates, rates) ** 2)
                
                randomness_df['model'].append(model_name)
                randomness_df['base'].append('total_divergence')
                randomness_df['divergence'].append(jensenshannon(avg_rates, rates) ** 2)
            
    randomness_df = pd.DataFrame(randomness_df)

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, **kwargs)

    axes = [ax1, ax2, ax3, ax4, ax5]
    metrics = ['total_divergence', 'A_divergence', 'C_divergence', 'G_divergence', 'T_divergence']

    pdf_to_write = {'Model': list()}
    for metric in metrics:
        pdf_to_write[metric + ' (mean)'] = list()
        pdf_to_write[metric + ' (median)'] = list()
        pdf_to_write[metric + ' (std)'] = list()

    for ax, metric in zip(axes, metrics):

        model_name_sorted = np.unique(randomness_df['model'])[::-1]
        medians = list()
        for i, model_name in enumerate(model_name_sorted):
            plot_df = randomness_df[randomness_df['model'] == model_name]
            x = np.array(plot_df.loc[plot_df['base'] == metric, 'divergence'])
            x = x[~np.isnan(x)]
            
            medians.append(np.median(x))
        medians = np.array(medians)

        top_3_models = model_name_sorted[np.argsort(medians)[:3]]
        medal_colors = ['#FFD700', '#C0C0C0', '#CD7F32'] # gold, silver, bronze 

        ticks_names = list()
        for i, model_name in enumerate(model_name_sorted):

            plot_df = randomness_df[randomness_df['model'] == model_name]
            x = np.array(plot_df.loc[plot_df['base'] == metric, 'divergence'])
            x = x[~np.isnan(x)]
            ticks_names.append(transform_name(model_name))

            if model_name in top_3_models:
                color = medal_colors[int(np.where(top_3_models == model_name)[0])]
                ax.axvline(np.median(x), color = color)
            else:
                color = 'white'

            if metric == 'total_divergence':
                pdf_to_write['Model'].append(transform_name(model_name))
            pdf_to_write[metric + ' (mean)'].append(np.mean(x))
            pdf_to_write[metric + ' (median)'].append(np.median(x))
            pdf_to_write[metric + ' (std)'].append(np.std(x))
            
            bp = ax.boxplot(x, positions = [i + 1], showfliers=False, widths = [0.6], vert = False, patch_artist = True)
            
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color='black',  linewidth=1.5)
            plt.setp(bp['boxes'], facecolor = color)

        ax.set_xlabel(metric_name_transformer[metric])


    ax1.set_yticks(ticks = np.arange(1, len(set(df['model'])) + 1, 1))
    ax1.set_yticklabels(labels = ticks_names)
    plt.tight_layout()
    plt.show()

    if output_dir_plots is not None:
        plot_file = os.path.join(output_dir_plots, 'randomness')
        f.savefig(plot_file + '.pdf', transparent=True, dpi=200, bbox_inches="tight")
        f.savefig(plot_file + '.png', facecolor='white', transparent=False, dpi=200, bbox_inches="tight")

    pdf_to_write = pd.DataFrame(pdf_to_write)
    if output_dir_tables is not None:
        pdf_to_write.to_csv(os.path.join(output_dir_tables, 'randomness.csv'), header = True, index = False)

    return f, pdf_to_write

def signature_correlation(df, output_dir_plots = None, **kwargs):

    signatures = list()
    for model_name in np.unique(df['model']):

        signatures_df = calculate_signatures(df[df['model'] == model_name])
        signatures_df = signatures_df.sort_values(['Base', 'Context', 'Error'])
        signatures_df = signatures_df[signatures_df["Error"].isin(["Match", "Missmatch_A", "Missmatch_C", "Missmatch_G", "Missmatch_T", "Deletion", 'Insertion'])]
        signatures_df = signatures_df[signatures_df['Base'] != 'Grouped']
        signature = np.array(signatures_df['Rate'])
        signatures.append(signature)
    signatures = np.vstack(signatures)

    model_names = list()
    for mn in np.unique(df['model']):
        model_names.append(transform_name(mn))
    model_names = np.array(model_names)

    D = pairwise_distances(X = signatures, metric = 'cosine')
    condensedD = squareform(D)

    # Compute and plot first dendrogram.
    f = plt.figure(**kwargs)

    ax2 = f.add_axes([0.3,0.71,0.6,0.2])
    Y = sch.linkage(condensedD, method='single')
    link_colors = ["black"] * (2 * len(condensedD) - 1)
    Z2 = sch.dendrogram(Y, link_color_func=lambda k: link_colors[k])
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Plot distance matrix.
    axmatrix = f.add_axes([0.3,0.1,0.6,0.6])
    idx2 = Z2['leaves']
    D = D[:,idx2]
    D = D[idx2,:]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=plt.cm.viridis)

    axmatrix.set_xticks(np.arange(0, len(np.unique(df['model'])), 1))
    axmatrix.set_yticks(np.arange(0, len(np.unique(df['model'])), 1))
    axmatrix.set_yticklabels(model_names[idx2])
    axmatrix.set_xticklabels(model_names[idx2])
    axmatrix.tick_params(labelbottom=True,labeltop=False)
    for tick in axmatrix.get_xticklabels():
        tick.set_rotation(90)

    # Plot colorbar.
    axcolor = f.add_axes([0.91,0.1,0.02,0.6])
    plt.colorbar(im, cax=axcolor)
    f.show()

    if output_dir_plots is not None:
        plot_file = os.path.join(output_dir_plots, 'signature_correlation')
        f.savefig(plot_file + '.pdf', transparent=True, dpi=200, bbox_inches="tight")
        f.savefig(plot_file + '.png', facecolor='white', transparent=False, dpi=200, bbox_inches="tight")

def homopolymer_rates(df, output_dir_tables = None, output_dir_plots = None, **kwargs):

    df['total_homo_counts'] = df['homo_A_counts'] + df['homo_C_counts'] + df['homo_G_counts'] + df['homo_T_counts']
    df['total_homo_errors'] = df['homo_A_errors'] + df['homo_C_errors'] + df['homo_G_errors'] + df['homo_T_errors']
    df['total_homo_error_rate'] = df['total_homo_errors']/df['total_homo_counts']
    df['A_homo_error_rate'] = df['homo_A_errors']/df['homo_A_counts']
    df['C_homo_error_rate'] = df['homo_C_errors']/df['homo_C_counts']
    df['G_homo_error_rate'] = df['homo_G_errors']/df['homo_G_counts']
    df['T_homo_error_rate'] = df['homo_T_errors']/df['homo_T_counts']

    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, sharey=True, **kwargs)

    axes = [ax1, ax2, ax3, ax4, ax5]
    metrics = ['total_homo_error_rate', 'A_homo_error_rate', 'C_homo_error_rate', 'G_homo_error_rate', 'T_homo_error_rate']
    pdf_to_write = {'Model': list()}
    for metric in metrics:
        pdf_to_write[metric + ' (mean)'] = list()
        pdf_to_write[metric + ' (median)'] = list()
        pdf_to_write[metric + ' (std)'] = list()
        
    for ax, metric in zip(axes, metrics):

        model_name_sorted = np.unique(df['model'])[::-1]
        medians = list()
        for i, model_name in enumerate(model_name_sorted):
            plot_df = df[df['model'] == model_name]
            x = np.array(plot_df[metric])
            x = x[~np.isnan(x)]
            
            medians.append(np.median(x))
        medians = np.array(medians)

        top_3_models = model_name_sorted[np.argsort(medians)[:3]]
        medal_colors = ['#FFD700', '#C0C0C0', '#CD7F32'] # gold, silver, bronze 

        ticks_names = list()
        for i, model_name in enumerate(model_name_sorted):

            plot_df = df[df['model'] == model_name]
            ticks_names.append(transform_name(model_name))

            x = np.array(plot_df[metric])
            x = x[~np.isnan(x)]

            if model_name in top_3_models:
                color = medal_colors[int(np.where(top_3_models == model_name)[0])]
                ax.axvline(np.median(x), color = color)
            else:
                color = 'white'

            if metric == metrics[0]:
                pdf_to_write['Model'].append(transform_name(model_name))
            pdf_to_write[metric + ' (mean)'].append(np.mean(x))
            pdf_to_write[metric + ' (median)'].append(np.median(x))
            pdf_to_write[metric + ' (std)'].append(np.std(x))
            
            bp = ax.boxplot(x, positions = [i + 1], showfliers=False, widths = [0.6], vert = False, patch_artist = True)
            
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color='black',  linewidth=1.5)
            plt.setp(bp['boxes'], facecolor = color)

        ax.set_xlabel(metric_name_transformer[metric])


    ax1.set_yticks(ticks = np.arange(1, len(set(df['model'])) + 1, 1))
    ax1.set_yticklabels(labels = ticks_names)
    f.supxlabel('Error rate')
    plt.tight_layout()
    plt.show()

    if output_dir_plots is not None:
        plot_file = os.path.join(output_dir_plots, 'homopolymers')
        f.savefig(plot_file + '.pdf', transparent=True, dpi=200, bbox_inches="tight")
        f.savefig(plot_file + '.png', facecolor='white', transparent=False, dpi=200, bbox_inches="tight")

    pdf_to_write = pd.DataFrame(pdf_to_write)
    if output_dir_tables is not None:
        pdf_to_write.to_csv(os.path.join(output_dir_tables, 'homopolymers.csv'), header = True, index = False)

    return f, pdf_to_write

def integrate(x, y):
   sm = 0
   for i in range(1, len(x)):
       h = x[i] - x[i-1]
       sm += h * (y[i-1] + y[i]) / 2

   return sm

def auc(df, output_dir_plots = None, output_dir_tables = None):

    pdf_to_write = {'Model': list(), 'auc': list()}

    for model_name in np.unique(df['model']):

        f, ax1 = plt.subplots(figsize=(7, 7))

        model_df = df[df['model'] == model_name]
        model_df = model_df.sort_values('phred_mean', ascending=True)
        model_df = model_df[~np.isnan(model_df['phred_mean'])]
        model_df = model_df.reset_index()

        x = list()
        y = list()
        z = list()
        for i in range(0, len(model_df), 50):
            sub_df = model_df.loc[i:, :]
            x.append(len(sub_df)/len(model_df))
            y.append(np.mean(sub_df['match_rate']))
            z.append(np.min(sub_df['phred_mean']))

        auc = -integrate(x, y)

        pdf_to_write['Model'].append(model_name)
        pdf_to_write['auc'].append(auc)

        ax2 = ax1.twinx()
        ax1.plot(x, y, color = 'black')
        ax2.plot(x, z, color = 'red')
        ax1.set_xlabel('Fraction of reads')
        ax1.set_ylabel('Average match rate')
        ax2.set_ylabel('Minimum read-mean PhredQ score')
        ax1.set_title(transform_name(model_name) + '\n' + 'AUC: ' + str(round(auc, 3)))
        
        f.tight_layout()
    
    if output_dir_plots is not None:
        plot_file = os.path.join(output_dir_plots, 'aucs', model_name)
        f.savefig(plot_file + '.pdf', transparent=True, dpi=200, bbox_inches="tight")
        f.savefig(plot_file + '.png', facecolor='white', transparent=False, dpi=200, bbox_inches="tight")
    plt.close()
    
    pdf_to_write = pd.DataFrame(pdf_to_write)
    if output_dir_tables is not None:
        pdf_to_write.to_csv(os.path.join(output_dir_tables, 'auc.csv'), header = True, index = False)

    return f, pdf_to_write
