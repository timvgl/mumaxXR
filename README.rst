mumaxXR contains an engine for xarray to read output directories of mumax directly, lazily into xarray.
Therefore, the class OvfEngine needs to be imported from mumaxXR. Simply pass the directory, or an ovf file (e.g. m000000.ovf) as the filename_or_obj argument to the method.
If only the directory is specified, the engine tries to find magnetization ovf files. If no magnetization ovf files are found the engine tries to load displacement ovf files.
If neither magnetization nor displacement ovf files are found the engine just loads the first appearing type of ovf files.
The engine has to be passed to xarray as engine=OvfEngine. Set chunks='auto' in order to parallelize loading of the ovf-files. If you want to set chunks to something else, keep in mind,
that each ovf file has to be accessed multiple times, if the chunks for the dimensions x, y, z and comp are not set the default values.

.. code-block:: python

    dataset = xr.open_dataset(example_mumax_output_dir, chunks='auto', engine=OvfEngine)

If multiple kinds of ovf files have been stored in the same time steps, another argument 'wavetype' can be passed to open_dataset, containing a list of the other short names, e.g.,

.. code-block:: python

    dataset = xr.open_dataset(path_to_example_mumax_output_dir, chunks='auto', engine=OvfEngine, wavetype=['m', 'u'])

Keep in mind that the names passed to wavetype have to be exactly like the prefix of ovf files. 

If multiple simulation directories are supposed to be concatenated, multiple mumax simulation directories can be passed to the engine.
The number of ovf files has to be the same. This is meant for concatenating simulations of parameter sweeps.
For 1D concatenation simply pass the parameter values inside a list to the engine.

.. code-block:: python

    dataset = xr.open_dataset(example_mumax_output_dir_0, chunks='auto', engine=OvfEngine, dirListToConcat=[path_to_example_mumax_output_dir_1, path_to_example_mumax_output_dir_2], sweepName='name_of_changed_parameter', sweepParam=[value_0_of_parameter, value_1_of_parameter, value_2_of_parameter])

For ND concatenation pass a list of names to sweepNames and a list of tuples to sweepParams

.. code-block:: python

    dataset = xr.open_dataset(example_mumax_output_dir_1, chunks='auto', engine=OvfEngine, dirListToConcat=[path_to_example_mumax_output_dir_2, path_to_example_mumax_output_dir_3, path_to_example_mumax_output_dir_4], sweepName=['name_of_changed_parameter_1', 'name_of_changed_parameter_2'], sweepParam=[(value_1_1_of_parameter, value_2_1_of_parameter), (value_1_2_of_parameter, value_2_1_of_parameter), (value_1_1_of_parameter, value_2_2_of_parameter), (value_1_2_of_parameter, value_2_2_of_parameter)])

If you want the ovf-files to be deleted after reading, set removeOvfFiles=True. This only works, if the ovf-files don't have to be accessed multiple times.
If you want to use ovf-files periodically, set useEachNthOvfFile=N. N has to be larger than one. Otherwise it is ignored. Using this flag, only each nth ovf file is used for the dataset.

Of course the wavetype option and the mumax directory concatenation can be used simultaneously.