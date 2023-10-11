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

	Positional arguments:
		axis:		As all above
		x_vect_set:	As scatterset
		y_vect_set:	As scatterset
	
	Keyword arguments:
		y_sig_vect_set:	As scatterset
		type_start:	As scatterset
		gradient:	As scatterset
		gradient_vals:	As scatterset
		name_list:	As scatterset
		kwargs:		Passed to plotline

plotfun
	Plots a function onto an axis, using plotline

	Positional arguments:
		axis:		As all above
		function:	given as a function or a lambda expression, must take a single positional input. This is the function that will be plotted onto the axis
	
	Keyword arguments:
		sig_function:	Gives a sigma function, which will fill an area above and below the main function, as in plotline. Otherwise as function
		x_min:		Default 0, the minimum x value
		x_max:		Default 1, the maximum x value
		log:		Default False. Whether to distribute the x sampling logarithmically
		n:		Default 100. The number of x values to sample
		kwargs:		Passed to plotline

plotfunset:
	Plots a list of functions onto an axis. Calls plotfun

	Positional arguments:
		axis:		As above
		function_set:	List of functions, must work like plotfun's function argument
	
	Keyword arguments:
		sig_function_set:	List of sigma functions, passed to plotfun
		kwargs:		passed to plotfun

plot2dfun:
	Plots a function which must take two positional arguments, x and y in that order. the second input is curried according to an input.
	
	Positional arguments:
		axis:		As above
		function:	The function which to be plotted
		second_var:	The second variable over which the function is curried, takes a list

	Keyword arguments:
		sig_function:	2 input sigma function, works like plotfun
		gradient:	Whether to plot colors as a gradient
		kwargs:		passed to plotfun

plotdf:
	Plots a dataframe in the form of a scatter or plotlinbe

	Positional arguments:
		axis:		As above
		df:		The dataframe from which to plot

	Keyword arguments:
		x:		Default "x", gives the column name of x values to plot
		y:		Default "y", gives the column name of the y values to plot. Can take a list to plot multiple y values.
		y_sig:		Default None, gives the column name of the y sigma values
		split:		Default None. Gives a second variable to "split" the values by, if working in long form.
		gradient:	Whether to use a gradient from the "split" values
		sort_x:		Default False. If True, then the x values will be sorted before plotting
		name:		Gives a column for the "name" of each plot. Might be a bit screwy, not tested.
		name_map:	Optional function mapping values from the "name" column to new names
		name_list:	A list of names
		plot_type:	Dfeault "scatter", can also be "line" or "scatterline" to plot a scatter/line plot
		gradient_map: 	Function to apply to "split" values before doing a gradient

heatmap:
	Plots a heatmap from a 2d array

	Positional arguments:
		axis:		You should know what this does by now
		matrix:		The 2d array of data

	Keyword arguments:
		colormap:	Default "lin", chooses which colormap to use
		legend:		Whether to add a legend
		val_range:	The range of values, default None. For if you want to plot the map between a defined set of values rather than the range of the data.

barchart:
	Plots a barchart

	Positional arguments:
		axis:		...
		values:		List or dict of values. If a dict is used, then the keys should be the names of the samples. The items in the list or dict can be tuples (y, y_sigma) or just items.

	Keyword arguments:
		names:		Optional list of names for the values
		sigma_list:	Optional list of sigma values.
		rotate_labels:	Default False. If turned on the labels for the bars will be rotated. This throws an error from matplotlib but IDK

barchart_label:
	Adds a label to an existing barchart. Useful for denoting significance

	Positional arguments:
		axis:		...
		text:		The text for the label
		start:		The start of the horizontal line
		end:		The end of the horizontal line
		height:		The height at which to place the line

	Keyword arguments:
		text_raise:	How far above the line to put the text (default zero)

save:
	Saves a figure

	Positional arguments:
		figure:		The figure to save
		name:		The "stem" for the file names
	
	Keyword arguments:
		file_types:	Default ["svg", "png"], a list of file types to save as
		dpi:		Default 600
		keep_box:	Default False, whether to remove the remove the right and top lines of the axis

remove_box:
	Removes parts of the axis
	
	Positional arguments:
		axis:		...
	
	Keyword arguments:
		lines:		Default ["top", "right"]. Which lines to remove from the axis box

nice_legend:
	Makes the legend on an axis nicer

	Positional arguments:
		axis:		

ridgelinedf:
	Plots a ridgeline plot from a dataframe. Complicated!

	Positional arguments:
		figure:		The figure to plot on. Takes a whole figure, since it must create and assemble multiple axes
		data_frame:	The dataframe from which the data comes

	Keyword arguments:
		x:		Default ["x"]. List of x values to get from the dataframe as plotdf
		y:		Defautl ["y"]. behaves like x, can also just be a single item in the list
		gradient:	Default False, whether to make a gradient
		colorcodes:	List of color codes for the plots
		linecodes:	List of line codes for the plots
		x_scale:	Optional adjustment to the x scale of the axes
		plots_per_axis:	How many lines per axis
		overlap:	Default 0.5, how much the axes overlap each other
		axis_names:	List of the names for the individual axes
		legend_names:	List of names for the different plot styles
		x_axis_name:	The name to use for the x axis variable

violinplot:
	Takes exactly the same axes as ridgeline, plots a violin plot.
