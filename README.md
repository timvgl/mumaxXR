mumaxXR contains an engine for xarray to read output directories of mumax directly, lazily into xarray.
Therefore, the class OvfEngine needs to be imported from mumaxXR. A single directory containing ovf files for magnetization only can be imported using the built-in library pathlib.
In this example, the directory path is converted into a Path object. Using the glob definition allows the loading of all the m*.ovf files.
After sorting the list (important, because otherwise, the time dimension is going to be messed up), the concat_dim has to be defined. For mumax-sim, it is always 't'. This is because in the background,
each ovf file is opened as a single dataset and each ovf file is representive for a single time step.
All these datasets are then concatenated along the time dimension using open_mfdataset. The combine argument has to be 'nested', and parallel=True is set for the fastest experience.
The engine has to be passed to xarray as engine=OvfEngine.
dataset = xr.open_mfdataset(sorted(list(Path('example_mumax_output_dir').glob('**/m*.ovf'))), concat_dim="t", chunks='auto', combine='nested', parallel=True, engine=OvfEngine)

If the magnetization has not been saved as an ovf file, then 'm' inside the glob command has to be replaced by the short name of the alternative ovf file (e.g., 'u' for displacement, etc.).
If multiple kinds of ovf files have been stored in the same time steps, another argument 'wavetype' can be passed to open_mfdataset, containing a list of the other short names, e.g.,
dataset = xr.open_mfdataset(sorted(list(Path('path_to_example_mumax_output_dir').glob('**/m*.ovf'))), concat_dim="t", chunks='auto', combine='nested', parallel=True, engine=OvfEngine, wavetype=['m', 'u'])

If multiple simulation directories are supposed to be concatenated, multiple mumax simulation directories can be passed to the engine.
The number of ovf files has to be the same. This is meant for concatenating simulations of parameter sweeps.
So far, only 1D-sweeps are supported. Work is in progress for DD concatenation.
dataset = xr.open_mfdataset(sorted(list(Path('example_mumax_output_dir').glob('**/m*.ovf'))), concat_dim="t", chunks='auto', combine='nested', parallel=True, engine=OvfEngine, dirListToConcat=[Path(path_to_example_mumax_output_dir_1), Path(path_to_example_mumax_output_dir)], sweepName='name_of_changed_parameter', sweepParam=[value_1_of_parameter, value_2_of_parameter])

