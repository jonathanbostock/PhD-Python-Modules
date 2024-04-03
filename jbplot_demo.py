### Demo jbplot

import numpy as np
import pandas as pd
import jbplot

data = pd.read_csv("jbplot_demo.csv",
                   sep="\t")

fig, ax = jbplot.figax()

jbplot.plotdf(ax, data,
              x="Success",
              y="Score",
              split="n_good",
              third_var="size",
              gradient=True,
              assemble_legend=True)

jbplot.save(fig, "Demo Figure",
            file_types=["png"])
