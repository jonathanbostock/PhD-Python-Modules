# Jonathan Bostock

import numpy as np
import jbplot

if __name__ =="__main__":

    real = np.linspace(-2,2,101)
    imag = np.linspace(-2,2,101)

    comp_values = np.zeros((101,101), dtype=np.csingle)
    for x, r in enumerate(real):
        for y, i in enumerate(imag):
            comp_values[x,y] = r + i*1j

    fig, ax = jbplot.figax()

    jbplot.complex_heatmap(ax, comp_values)
    jbplot.save(fig, "Complex Heatmap Test",
                file_types=["png"], keep_box=True)
