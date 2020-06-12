import numpy as np
import pandas as pd


class bf(object):
    def __init__(self, v=0, f=max):
        self.v = v
        self.f = f

    def ftest(self, x):
        x = x.item()
        if np.abs(self.f(x, self.v) - x) < 1e-8:
            self.v = x
        return '{}'.format(x)

    def freal(self, x):
        x = x.item()
        if np.abs(x - self.v) < 1e-8:
            return '\\bf{{{}}}'.format(x)
        else:
            return '{}'.format(x)


def latex_strip(tex):
    tex = tex[tex.find('midrule') + 8:]
    return tex[:tex.find('\\bottomrule')]


def latex_head(tex):
    return tex[:tex.find('midrule') + 8]


def latex(df, groups, cols, vals, funcs):
    head = latex_head(df.to_latex(escape=False))
    total = head
    for g in groups:
        ff = []
        for c, v, f in zip(cols, vals, funcs):
            ff.append((c, bf(v, f)))
        formatters = dict([(c, f.ftest) for c, f in ff])
        tex = df.loc[g].to_latex(formatters=formatters, escape=False)
        formatters = dict([(c, f.freal) for c, f in ff])
        tex = df.loc[g].to_latex(formatters=formatters, escape=False)
        tex = latex_strip(tex)
        tex = '\n'.join(['{}\t&{}'.format(g, line) for line in tex.split('\n') if line.strip() != ''])
        total = total + tex + '\midrule\n'
    return total + r'\bottomrule\end{tabular}'