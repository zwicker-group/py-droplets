Examples
^^^^^^^^

We here collect examples for using the package to demonstrate some of its
functionality. 


Basic droplets
""""""""""""""

The basic droplet classes can be used as follows

.. include:: ../examples/define_droplets.rst

We first create two droplets represented by two different classes. The basic
class :class:`~droplets.droplets.SphericalDroplet` represents a droplet by its
position and radius, while the more advanced class 
:class:`~droplets.droplets.DiffuseDroplet` also keeps track of the interface
width.
Finally, we combine the two droplets in an emulsion, which then allows further
analysis.


Plotting emulsions
""""""""""""""""""

To visualize an emulsions, one can simply use the
:meth:`~droplets.emulsions.Emulsion.plot`:

.. include:: ../examples/plot_emulsion.rst

Note that the emulsion class can also keep track of the space in which droplets
are defined, e.g, the boundaries of a simulation grid.
For this, the :class:`~droplets.emulsions.Emulsion` supports the `grid` 
argument, which can for instance be an instance of
:class:`~pde.grids.cartesian.CartesianGrid`.


Analyze images
""""""""""""""

.. figure:: ../examples/resources/emulsion.png
   :figclass: align-right

   An emulsion image
   
   
The package also allows analyzing images of emulsions like the one shown on the
right. The code below loads the image, locates the droplets, and then displays
some of their properties

.. include:: ../examples/analyze_image.rst

Note that the determined positions and sizes of the droplets are only roughly
determined by default.
If more accurate data is desired,
:func:`~droplets.image\_analysis.locate\_droplets` supports the `refine`
arguments, which fits the model image of droplet to the actual image to obtain
more accurate parameter estimates.

