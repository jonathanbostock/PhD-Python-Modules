Hello! Welcome to jbplot
Why are you here?

Required modules:
	matplotlib
	numpy
	pandas

Summary:
	This is my module for making my own plotting quicker. I have three philosophies for creating it.
		1: Plotting data should take the minimum possible number of inputs
		2: Plots should look nice by default
		3: This module is for plotting and not statistics

Function Docs:

Code calling order is:
        scatterset      --> scatter
        plotlineset     --> plotline
        scatter         --> plotline (sometimes)
        plotfunset      --> plotfun
        plot2dfun       --> plotfun
        plotfun         --> plotline
        plotdf          --> scatterset and/or plotlineset 
        violinplot      --> ridgelinedf
        rdigelinedf     --> plotline

scatter
	Function that takes an axis as input and mutates the axis by adding a scatter plot. Optionally also calls plotline to add lines to the plot.

	Positional arguments
		axis:		The axis where the scatter plot will be plotted
		x_vect:		Vector of x values
		y_vect:		Vector of y values

	Keyword arguments
		y_sig_vect:	Optional vector of y sigma values
		colorcode:	Optional integer value which determines the color to be plotted
		linecode:	Optional integer value to determine the line style (if you want to plot lines between the points)
		marktype:	Optional value "f" for filled (default) or  "l" for a line-based marker
		annotations:	Optional list of strings to add annotations
		ann_y_offset:	Optional argument which offsets the annotations from the points
		color_override:	Optional argument to override the color with a string in the form "#012345"
		**kwargs:	Get passed to plotline if necessary


plotline
	Funciton that mutates an input axis by adding a line plot to it
	
	Positional arguments:
		axis:		As scatter
		x_vect:		As scatter
		y_vect:		As scatter

	Keyword raguments
		y_sig_vect:	As scatter
		colorcode:	As scatter
		linecode:	As scatter
		color_override:	As scatter
		linecode_override:	As scatter

gradient_handler
	Function used internally to handle gradients. Avoid. No I'm not putting underscores there. Avoid it yourself.

scatterset
	Plots a list of lists, using scatter
	
	Positional arguments:
		axis:		As scatter
		x_vect_set:	A list of lists. Can contain only one list, or can have the same number of lists as y_vect_set
		y_vect_set:	A list of lists

	Keyword arguments:
		y_sig_vect_set:	Optional list of lists
		marktype:	As scatter
		line:		Whether to add lines
		linecode_override:	Overrides the default linecodes (which vary the line by dataset)
		type_start:		Gives a starting value other than zero (default) to linecode and colorcode. Useful if you want to have different sets of styles on different plots
		name_list:		Allows for labelling
		gradient:		Whether or not to use a gradient of colors for the line colors
		gradient_vals:		Manual gradient values (i.e. don't use evenly spaced values for the gradient)
		color_override:		Overrides all the colors

plotlineset
	Plots a list of lists, using plotline

	
