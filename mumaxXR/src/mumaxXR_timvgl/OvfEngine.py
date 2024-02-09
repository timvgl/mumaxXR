import numpy as np
import xarray as xr
import dask
import dask.array
from pathlib import Path
import re

_binary = 4
class MumaxMesh:
    def __init__(self, filename, nodes, world_min, world_max, tmax, n_comp, footer_dict, step_times=None):
        self.filename = filename
        self.nodes = nodes
        self.world_min = world_min
        self.world_max = world_max
        self.tmax = tmax
        self.footer_dict = footer_dict
        # Number of cells in the x-y-plane:
        self.number_of_cells = nodes[0]*nodes[1]*nodes[2]
        self.n_comp = n_comp
        self.step_times = step_times
        self.cellsize = [self.get_cellsize(i) for i in range(3)]

    def get_axis(self, i):
        return np.linspace(self.world_min[i], self.world_max[i], self.nodes[i])

    def get_cellsize(self, i):
        return (self.world_max[i] - self.world_min[i]) / self.nodes[i]

class OvfEngine(xr.backends.BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
        dtype=np.float32,
        wavetype=[],
        dirListToConcat=[],
        sweepName='',
        sweepParam=[]
    ):
        filename_or_obj = Path(filename_or_obj)
        coordsParam = None
        if (len(dirListToConcat) > 0):
            dirListToConcat = [Path(dir) for dir in dirListToConcat]
            if (filename_or_obj.parent not in dirListToConcat):
                dirListToConcat = [filename_or_obj.parent] + dirListToConcat
            if (sweepName == ''):
                sweepName = "unknown"
            if (len(sweepParam) == len(dirListToConcat)):
                coordsParam = np.array(sweepParam)
            else:
                coordsParam = np.array(range(len(dirListToConcat)))

        mesh = self.read_mesh_from_ovf(filename_or_obj)
        dims = None
        coords = None
        shape = None
        data = None
        if (len(dirListToConcat) == 0):
            if (len(wavetype) == 0):
                dims = ['t', 'z', 'y', 'x', 'comp']
                coords = [np.array([mesh.tmax]), mesh.get_axis(2), mesh.get_axis(1), mesh.get_axis(0), np.arange(mesh.n_comp)]
                shape = (1, ) + tuple([axis.shape[0] for axis in coords[1:]])
                data = dask.array.from_delayed(self.read_data_from_ovf(filename_or_obj, dtype=dtype, mesh=mesh), shape=shape, dtype=dtype)
            else:
                dims = ['wavetype', 't', 'z', 'y', 'x', 'comp']
                coords = [np.array(wavetype), np.array([mesh.tmax]), mesh.get_axis(2), mesh.get_axis(1), mesh.get_axis(0), np.arange(mesh.n_comp)]
                shape = (1, ) + tuple([axis.shape[0] for axis in coords[2:]])
                for type in wavetype:
                    if (data is None):
                        data = dask.array.expand_dims(dask.array.from_delayed(self.read_data_from_ovf(filename_or_obj, dtype=dtype, mesh=mesh, type=type), shape=shape, dtype=dtype), 0)
                    else:
                        newData = dask.array.from_delayed(self.read_data_from_ovf(filename_or_obj, dtype=dtype, mesh=mesh, type=type), shape=shape, dtype=dtype)
                        data = dask.array.concatenate((data, dask.array.expand_dims(newData, 0)), axis=0)
        else:
            if (len(wavetype) == 0):
                dims = [sweepName, 't', 'z', 'y', 'x', 'comp']
                coords = [coordsParam, np.array([mesh.tmax]), mesh.get_axis(2), mesh.get_axis(1), mesh.get_axis(0), np.arange(mesh.n_comp)]
                shape = (1, ) + tuple([axis.shape[0] for axis in coords[2:]])
                for dir in dirListToConcat:
                    if (data is None):
                        data = dask.array.expand_dims(dask.array.from_delayed(self.read_data_from_ovf(dir.joinpath(filename_or_obj.name), dtype=dtype, mesh=mesh), shape=shape, dtype=dtype), 0)
                    else:
                        newData = dask.array.from_delayed(self.read_data_from_ovf(dir.joinpath(filename_or_obj.name), dtype=dtype, mesh=mesh), shape=shape, dtype=dtype)
                        data = dask.array.concatenate((data, dask.array.expand_dims(newData, 0)), axis=0)
            else:
                dims = [sweepName, 'wavetype', 't', 'z', 'y', 'x', 'comp']
                coords = [coordsParam, np.array(wavetype), np.array([mesh.tmax]), mesh.get_axis(2), mesh.get_axis(1), mesh.get_axis(0), np.arange(mesh.n_comp)]
                shape = (1, ) + tuple([axis.shape[0] for axis in coords[3:]])
                for dir in dirListToConcat:
                    if (data is None):
                        subData = None
                        for type in wavetype:
                            if (subData is None):
                                subData = dask.array.expand_dims(dask.array.from_delayed(self.read_data_from_ovf(dir.joinpath(filename_or_obj.name), dtype=dtype, mesh=mesh, type=type), shape=shape, dtype=dtype), 0)
                            else:
                                newSubData = dask.array.from_delayed(self.read_data_from_ovf(dir.joinpath(filename_or_obj.name), dtype=dtype, mesh=mesh, type=type), shape=shape, dtype=dtype)
                                subData = dask.array.concatenate((subData, dask.array.expand_dims(newSubData, 0)), axis=0)
                        data = dask.array.expand_dims(subData, 0)
                    else:
                        subData = None
                        for type in wavetype:
                            if (subData is None):
                                subData = dask.array.expand_dims(dask.array.from_delayed(self.read_data_from_ovf(dir.joinpath(filename_or_obj.name), dtype=dtype, mesh=mesh, type=type), shape=shape, dtype=dtype), 0)
                            else:
                                newSubData = dask.array.from_delayed(self.read_data_from_ovf(dir.joinpath(filename_or_obj.name), dtype=dtype, mesh=mesh, type=type), shape=shape, dtype=dtype)
                                subData = dask.array.concatenate((subData, dask.array.expand_dims(newSubData, 0)), axis=0)
                        data = dask.array.concatenate((data, dask.array.expand_dims(subData, 0)), axis=0)
        #print(str(mesh.tmax) + filename_or_obj.stem)
        dset = xr.DataArray(data, dims=dims, coords=coords).to_dataset(name="raw")
        dset.attrs["cellsize"] = mesh.cellsize
        dset.attrs["nodes"] = mesh.nodes
        dset.attrs["min_size"] = mesh.world_min
        dset.attrs["max_size"] = mesh.world_max
        dset.attrs["n_comp"] = mesh.n_comp
        dset.attrs["n_cells"] = mesh.number_of_cells
        return dset


    def read_mesh_from_ovf(self, filename):
        file = open(filename, "rb")
        footer = []
        footer_dict = dict()
        # discard headline
        file.readline()
        for i in range(0, 27):
            line = file.readline()
            # clean up
            line = line.replace(b'# ', b'')
            line = line.replace(b'Desc: Total simulation time: ', b'tmax: ')
            line = line.replace(b'Desc: Time (s) : ', b'tmax: ')
            footer.append(line)
            attr, val = line.split(b': ')
            footer_dict[attr] = val.replace(b"\n", b"")

        xnodes = int(footer_dict[b"xnodes"])
        ynodes = int(footer_dict[b"ynodes"])
        znodes = int(footer_dict[b"znodes"])


        nodes = np.array([xnodes, ynodes, znodes])
        xmin = float(footer_dict[b"xmin"])
        ymin = float(footer_dict[b"ymin"])
        zmin = float(footer_dict[b"zmin"])

        world_min = np.array([xmin, ymin, zmin])

        xmax = float(footer_dict[b"xmax"])
        ymax = float(footer_dict[b"ymax"])
        zmax = float(footer_dict[b"zmax"])

        world_max = np.array([xmax, ymax, zmax])
        if b' s' in footer_dict[b"tmax"]:
            tmax_string, _ = footer_dict[b"tmax"].split(b' s')
        else:
            tmax_string = footer_dict[b"tmax"]
        tmax = float(tmax_string)

        if b'valuedim' in footer_dict:
            n_comp = int(footer_dict[b'valuedim'])
        else:
            n_comp = 1
        file.close()
        return MumaxMesh(filename, nodes, world_min, world_max, tmax, n_comp, footer_dict)
    
    @dask.delayed
    def read_data_from_ovf(self, filename, mesh=None, dtype=np.float32, type=''):
        filename = Path(filename)
        if (type != ''):
            filename = filename.parent.joinpath(Path(type + str(filename.name)[re.search(r"\d", filename.stem).start():]))
        file = open(filename, "rb")
        n_comp = mesh.n_comp

        if not _binary == 4:
            raise ValueError('Error: Unknown binary type assigned for reading in .ovf2 data')

        data_start_pos = 0

        for i in range(0, 46):
            line = file.readline()
            if b'# Begin: Data' in line:
                # the first binary number is a control number specified in the ovf format
                file.read(_binary)
                data_start_pos = file.tell()
                break
            if (i == 45):
                raise ValueError("Error: %s has no well formatted data segment." % filename)

        file.seek(data_start_pos, 0)
        size = int(n_comp * mesh.number_of_cells * _binary / 4)
        data = np.fromfile(file, dtype=dtype, count=size)
        data = np.expand_dims(data.reshape(mesh.nodes[2], mesh.nodes[1], mesh.nodes[0], n_comp), 0)
        file.close()
        return data
    

if __name__ == "__main__":
    #arr = xr.open_dataset(r"/home/tvo/Data_Disk/mumax-python-server/mumaxDataRaw/eta_1e+16_script.mod.out/m000000.ovf", engine=OvfEngine)
    dirList = [r"/home/tvo/Data_Disk/mumax-python-server/mumaxDataRaw/eta_1e+17_script.mod.out/"]
    arr = xr.open_mfdataset(sorted(list(Path(r"/home/tvo/Data_Disk/mumax-python-server/mumaxDataRaw/eta_1e+16_script.mod.out/").glob("**/m*.ovf"))), parallel=True, chunks={"t": 1}, engine=OvfEngine, wavetype=["m", "u"], dirListToConcat=dirList, sweepName="eta", sweepParam=[1e+16, 1e+17])
    #load all m*.ovf files
    #run parallel
    #set chunks
    #set engine
    #if u files are as ovf files available concat in wavetype - same amount of u and m ovf files needed, optional
    #paramsweep that is supposed to concatenated? -> pass all dir in lists. same amount of ovf files as in first arg needed - if wavetypes are there, same amount for u and m
    #give name for sweep param (for dim) - if no name is passed, dim is labled as unknown
    #sweep params for coords of param - if no list is passed, generated ints from 0 to length of dirList
    print(arr)
    arr.to_netcdf("testooooooooo.nc")