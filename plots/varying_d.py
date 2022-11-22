import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

algorithms = ['SingleObj', 'Polytope', 'RRMS', 'RRMS-Star', 'HS-RRM']
colors = ['black', 'purple', 'forestgreen', 'blue', 'red']

d_vals = [2, 3, 4, 5, 6, 7]

plt.rcParams.update({
    "font.size": 8,
    "text.usetex": True,
    "font.family": "sans-serif",
    "text.latex.preamble": r"\usepackage{arev}",
    "font.sans-serif": ["Helvetica"]})


def read_csv(fname):
    min_mrr = 1
    max_mrr = 0
    avg = {}
    std = {}
    for alg in algorithms:
        avg[alg] = []
        std[alg] = []
    with open(fname, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        i = 0
        for row in reader:
            i += 1
            if i == 1:
                continue
            alg = row[1]
            avg_i = float(row[4])
            std_i = float(row[5])
            avg[alg].append(avg_i)
            std[alg].append(std_i)
            if avg_i < min_mrr:
                min_mrr = avg_i
            if avg_i > max_mrr:
                max_mrr = avg_i
    return avg, std, min_mrr, max_mrr


def draw_d(dataset, k, avg, std, y_min, y_max):
    figs, ax = plt.subplots()

    ax.tick_params(axis='y', which='both', direction='out')
    ax.set_xlabel(r'd', fontsize=10)
    if k == 10:
        ax.set_title(dataset + ' (k = 10)', fontsize=10)
    else:
        ax.set_title(dataset + ' (k = 25)', fontsize=10)
    ax.set_xlim(1.8, 7.2)
    ax.set_xticks(range(2, 8))
    ax.xaxis.set_minor_locator(MultipleLocator(1))

    ax.set_ylabel(r'Estimated MRR', fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max)

    ax.plot(d_vals, avg['SingleObj'], color=colors[0], linewidth=1)
    ax.plot(d_vals, avg['Polytope'], color=colors[1], linewidth=1)
    ax.errorbar(x=d_vals, y=avg['RRMS'], yerr=std['RRMS'], color=colors[2], linewidth=1, ecolor=colors[2], elinewidth=1,
                capsize=3)
    ax.errorbar(x=d_vals, y=avg['RRMS-Star'], yerr=std['RRMS-Star'], color=colors[3], linewidth=1, ecolor=colors[3],
                elinewidth=1, capsize=3)
    ax.errorbar(x=d_vals, y=avg['HS-RRM'], yerr=std['HS-RRM'], color=colors[4], linewidth=1, ecolor=colors[4],
                elinewidth=1, capsize=3)

    figs.tight_layout(rect=[0.125, 0.125, 1, 0.94])
    figs.set_figheight(1.6)
    figs.set_figwidth(2.4)

    plt.grid(axis='both', color='gray', lw=0.1, ls=':', which='both')

    plt.savefig('./%s_k%s_d.pdf' % (dataset, k), format='pdf')


if __name__ == "__main__":
    mrr, var, min_mrr, max_mrr = read_csv('./results/max_cover_k10.csv')
    print(min_mrr, max_mrr)
    draw_d('email-Eu-core', 10, mrr, var, 0.001, 1)

    mrr, var, min_mrr, max_mrr = read_csv('./results/max_cover_k25.csv')
    print(min_mrr, max_mrr)
    draw_d('email-Eu-core', 25, mrr, var, 0.0001, 0.5)

    mrr, var, min_mrr, max_mrr = read_csv('./results/data_summarization_k10.csv')
    print(min_mrr, max_mrr)
    draw_d('MovieLens', 10, mrr, var, 0.0001, 0.125)

    mrr, var, min_mrr, max_mrr = read_csv('./results/data_summarization_k25.csv')
    print(min_mrr, max_mrr)
    draw_d('MovieLens', 25, mrr, var, 0.0001, 0.125)
