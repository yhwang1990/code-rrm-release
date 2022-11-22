import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

algorithms = ['SingleObj', 'Polytope', 'RRMS', 'RRMS-Star', 'HS-RRM']
colors = ['black', 'purple', 'forestgreen', 'blue', 'red']

k_vals_d2 = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
k_vals_d5 = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

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
            avg_i = float(row[3])
            if len(row) > 4:
                std_i = float(row[4])
            else:
                std_i = 0
            avg[alg].append(avg_i)
            std[alg].append(std_i)
            if avg_i < min_mrr:
                min_mrr = avg_i
            if avg_i > max_mrr:
                max_mrr = avg_i
    return avg, std, min_mrr, max_mrr


def draw_k_mc(dataset, d, avg, std, y_min, y_max):
    figs, ax = plt.subplots()

    ax.tick_params(axis='y', which='both', direction='out')
    ax.set_xlabel(r'k', fontsize=10)
    if d == 2:
        ax.set_title(dataset + ' (d = 2)', fontsize=10)
        ax.set_xlim(0, 2)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    else:
        ax.set_title(dataset + ' (d = 5)', fontsize=10)
        ax.set_xlim(0, 51)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.xaxis.set_minor_locator(MultipleLocator(5))

    ax.set_ylabel(r'Estimated MRR', fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max)

    if d == 2:
        ax.hlines(y=avg['SingleObj'], xmin=0, xmax=20, color=colors[0], linewidth=1)
        ax.plot(k_vals_d2, avg['Polytope'], color=colors[1], linewidth=1)
        ax.plot(k_vals_d2, avg['RRMS'], color=colors[2], linewidth=1)
        ax.plot(k_vals_d2, avg['RRMS-Star'], color=colors[3], linewidth=1)
        ax.plot(k_vals_d2, avg['HS-RRM'], color=colors[4], linewidth=1)
    else:
        ax.hlines(y=avg['SingleObj'], xmin=0, xmax=51, color=colors[0], linewidth=1)
        ax.plot(k_vals_d5, avg['Polytope'], color=colors[1], linewidth=1)
        ax.errorbar(x=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50], y=avg['RRMS'][1:], yerr=std['RRMS'][1:], color=colors[2],
                    linewidth=1, ecolor=colors[2], elinewidth=1, capsize=3)
        ax.errorbar(x=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50], y=avg['RRMS-Star'][1:], yerr=std['RRMS-Star'][1:],
                    color=colors[3], linewidth=1, ecolor=colors[3], elinewidth=1, capsize=3)
        ax.errorbar(x=k_vals_d5, y=avg['HS-RRM'], yerr=std['HS-RRM'], color=colors[4], linewidth=1,
                    ecolor=colors[4], elinewidth=1, capsize=3)

    figs.tight_layout(rect=[0.125, 0.125, 1, 0.94])
    figs.set_figheight(1.6)
    figs.set_figwidth(2.4)

    plt.grid(axis='both', color='gray', lw=0.1, ls=':', which='both')

    plt.savefig('./%s_d%s_k.pdf' % (dataset, d), format='pdf')


def draw_k_dm(dataset, d, avg, std, y_min, y_max):
    figs, ax = plt.subplots()

    ax.tick_params(axis='y', which='both', direction='out')
    ax.set_xlabel(r'k', fontsize=10)
    if d == 2:
        ax.set_title(dataset + ' (d = 2)', fontsize=10)
        ax.set_xlim(0, 10)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    else:
        ax.set_title(dataset + ' (d = 5)', fontsize=10)
        ax.set_xlim(0, 51)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.xaxis.set_minor_locator(MultipleLocator(5))

    ax.set_ylabel(r'Estimated MRR', fontsize=10)
    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max)

    if d == 2:
        ax.hlines(y=avg['SingleObj'], xmin=-1, xmax=11, color=colors[0], linewidth=1)
        ax.plot(range(1, 11), avg['Polytope'], color=colors[1], linewidth=1)
        ax.plot(range(1, 11), avg['RRMS'], color=colors[2], linewidth=1)
        ax.plot(range(1, 11), avg['RRMS-Star'], color=colors[3], linewidth=1)
        ax.plot(range(1, 11), avg['HS-RRM'], color=colors[4], linewidth=1)
    else:
        ax.hlines(y=avg['SingleObj'], xmin=0, xmax=51, color=colors[0], linewidth=1)
        ax.plot(k_vals_d5, avg['Polytope'], color=colors[1], linewidth=1)
        ax.errorbar(x=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50], y=avg['RRMS'][1:], yerr=std['RRMS'][1:], color=colors[2],
                    linewidth=1, ecolor=colors[2], elinewidth=1, capsize=3)
        ax.errorbar(x=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50], y=avg['RRMS-Star'][1:], yerr=std['RRMS-Star'][1:],
                    color=colors[3], linewidth=1, ecolor=colors[3], elinewidth=1, capsize=3)
        ax.errorbar(x=k_vals_d5, y=avg['HS-RRM'], yerr=std['HS-RRM'], color=colors[4], linewidth=1,
                    ecolor=colors[4], elinewidth=1, capsize=3)

    figs.tight_layout(rect=[0.125, 0.125, 1, 0.94])
    figs.set_figheight(1.6)
    figs.set_figwidth(2.4)

    plt.grid(axis='both', color='gray', lw=0.1, ls=':', which='both')

    plt.savefig('./%s_d%s_k.pdf' % (dataset, d), format='pdf')


if __name__ == "__main__":
    mrr, var, min_mrr, max_mrr = read_csv('./results/max_cover_d2.csv')
    print(min_mrr, max_mrr)
    draw_k_mc('email-Eu-core', 2, mrr, var, 0.0001, 0.125)

    mrr, var, min_mrr, max_mrr = read_csv('./results/max_cover_d5.csv')
    print(min_mrr, max_mrr)
    draw_k_mc('email-Eu-core', 5, mrr, var, 0.01, 1)

    mrr, var, min_mrr, max_mrr = read_csv('./results/data_summarization_d2.csv')
    print(min_mrr, max_mrr)
    draw_k_dm('MovieLens', 2, mrr, var, 0.0001, 0.1)

    mrr, var, min_mrr, max_mrr = read_csv('./results/data_summarization_d5.csv')
    print(min_mrr, max_mrr)
    draw_k_dm('MovieLens', 5, mrr, var, 0.0006, 0.125)
