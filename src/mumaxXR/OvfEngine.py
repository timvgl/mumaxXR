import numpy as np
import xarray as xr
import dask
import dask.array
from pathlib import Path
import re
import sys
import os
from forbiddenfruit import curse
import warnings

class custom_string():
    def replace_substring_at_nth(self, oldSubstring, newSubstring="", n=-1, count=-1):
        beforeN = self[:n]
        if (n == -1):
            afterN = ''
        else:
            afterN = self[n+1:]
        return beforeN + self[n].replace(oldSubstring, newSubstring, count) + afterN
    def check_upper_available(self):
        for char in self:
            if (char.isupper()):
                return True
        return False

    def check_lower_available(self):
        for char in self:
            if (char.islower()):
                return True
        return False
    def cast_toFloatExceptString(self):
        try:
            self = float(self)
        except:
            self = str(self)
        return self

def castableList(inputList):
    try:
        list(inputList)
        return True
    except TypeError:
        return False

class custom_list():
    def isinstance_recursive(self, checkType, maxDepth=-1, currentDepth=0):
        isTypeList = []
        for self_ in self:
            if (castableList(self_) and (maxDepth != -1 and currentDepth != maxDepth or maxDepth == -1)):
                currentDepth += 1
                isTypeList.append(list(self_).isinstance_recursive(checkType, maxDepth=maxDepth, currentDepth=currentDepth))
            else:
                isTypeList.append(isinstance(self_, checkType))
        if (False in isTypeList):
            return False
        else:
            return True
    def is_homogenious(self):
        try:
            np.asarray(self)
            return True
        except ValueError:
            return False
        
                
    
curse(str, "check_upper_available", custom_string.check_upper_available)
curse(str, "check_lower_available", custom_string.check_lower_available)
curse(str, "replace_substring_at_nth", custom_string.replace_substring_at_nth)
curse(str, "cast_toFloatExceptString", custom_string.cast_toFloatExceptString)

curse(list, "isinstance_recursive", custom_list.isinstance_recursive)
curse(list, "is_homogenious", custom_list.is_homogenious)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

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
            if (isinstance(sweepParam, list)):
                if (len(sweepParam) > 0 and isinstance(sweepParam, (float, int))):
                    if (len(sweepParam) == len(dirListToConcat)):
                        coordsParam = np.array(sweepParam)
                    else:
                        coordsParam = np.array(range(len(dirListToConcat)))
                elif (len(sweepParam) > 0 and sweepParam.isinstance_recursive((int, float)) and sweepParam.is_homogenious()):
                    coordsParam = [np.array(param) for param in sweepParam]
                    if (len(sweepName) != len(list(sweepParam[0])) or not isinstance(sweepName, (list, tuple))):
                        if (not isinstance(sweepName, (list, tuple))):
                            sweepName = [sweepName]
                        if len(sweepName) < len(list(sweepParam[0])):
                            warnings.warn('Not enough labels for sweep labeling found. Appending ...')
                            for i in range(len(list(sweepParam[0])) - len(sweepName)):
                                sweepName.append('unknown')
                        else:
                            warnings.warn('Too many labels for sweep labeling found. Trimming ...')
                            sweepName = sweepName[:len(list(coordsParam[0]))]
                    sortIndices = []
                    for row in sorted(sweepParam):
                        sortIndices.append(sweepParam.index(row))
                    dirListToConcat = [dirListToConcat[i] for i in sortIndices]
                else:
                    raise ValueError('Provided coords are not homogenious. Tuples or lists in list must be the same size for all entries.')

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
                if (isinstance(sweepName, str)):
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
                    dims = sweepName + ['t', 'z', 'y', 'x', 'comp']
                    coords = [np.array([mesh.tmax]), mesh.get_axis(2), mesh.get_axis(1), mesh.get_axis(0), np.arange(mesh.n_comp)]
                    shape = (1, ) + tuple([axis.shape[0] for axis in coords])
                    for dir in dirListToConcat:
                        if (data is None):
                            data = dask.array.expand_dims(dask.array.from_delayed(self.read_data_from_ovf(dir.joinpath(filename_or_obj.name), dtype=dtype, mesh=mesh), shape=shape, dtype=dtype), 0)
                        else:
                            newData = dask.array.from_delayed(self.read_data_from_ovf(dir.joinpath(filename_or_obj.name), dtype=dtype, mesh=mesh), shape=shape, dtype=dtype)
                            data = dask.array.concatenate((data, dask.array.expand_dims(newData, 0)), axis=0)
                    shape = None
                    for i in range(np.transpose(np.asarray(coordsParam))[:,0].size):
                        if shape == None:
                            shape = (np.unique(np.transpose(np.asarray(coordsParam))[i,:]).size, )
                        else:
                            shape += (np.unique(np.transpose(np.asarray(coordsParam))[i,:]).size, )
                            
                    shape += tuple([axis.shape[0] for axis in coords[-5:]])
                    data = dask.array.reshape(data, shape=shape)
                    coordsDir = []
                    for i in range(np.transpose(np.asarray(coordsParam))[:,0].size):
                        coordsDir.append(np.unique(np.transpose(np.asarray(coordsParam))[i,:]))
                    coords = coordsDir + coords
            else:
                if (isinstance(sweepName, str)):
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
                else:
                    dims = sweepName + ['wavetype', 't', 'z', 'y', 'x', 'comp']
                    coords = [np.array(wavetype), np.array([mesh.tmax]), mesh.get_axis(2), mesh.get_axis(1), mesh.get_axis(0), np.arange(mesh.n_comp)]
                    shape = (1, ) + tuple([axis.shape[0] for axis in coords[2:]])
                    data = None
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
                    shape = None
                    for i in range(np.transpose(np.asarray(coordsParam))[:,0].size):
                        if shape == None:
                            shape = (np.unique(np.transpose(np.asarray(coordsParam))[i,:]).size, )
                        else:
                            shape += (np.unique(np.transpose(np.asarray(coordsParam))[i,:]).size, )
                            
                    shape += tuple([axis.shape[0] for axis in coords[-6:]])
                    data = dask.array.reshape(data, shape=shape)
                    coordsDir = []
                    for i in range(np.transpose(np.asarray(coordsParam))[:,0].size):
                        coordsDir.append(np.unique(np.transpose(np.asarray(coordsParam))[i,:]))
                    coords = coordsDir + coords

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
    def read_data_from_ovf(self, filename, mesh=None, dtype=np.float32, type='') -> np.array:
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
    
def convert_TableFile_to_dataset(tableTxtFilePath):
    tableArrayVec = None
    tableArraySc = None
    with HiddenPrints():
        numpyData = np.loadtxt(tableTxtFilePath, dtype=np.float32)
        with open(tableTxtFilePath, "r") as tableFile:
            titleLine = tableFile.readline()
    numpyData = numpyData[np.unique(numpyData[:,0], return_index=True)[1]]
    colList = np.asarray([col.replace("\n", "").replace("# ", "").split(" ")[0].replace_substring_at_nth("x").replace_substring_at_nth("y").replace_substring_at_nth("z") for col in titleLine.split("\t")])
    colListUniqueUnsorted, colListUniqueIndex, counts = np.unique(colList, return_index=True, return_counts=True)
    colListUnique = colList[np.sort(colListUniqueIndex)].tolist()
    counts = counts[[colListUniqueUnsorted.tolist().index(col) for col in colListUnique]]
    counts = counts.tolist()
    if (3 in counts[1:]):
        dimsVec = ["t", "comp"]
        coordsVec = {"t": numpyData[:,0], "comp": [0, 1, 2]}
    if (1 in counts[1:]):
        dimsSc = ["t"]
        coordsSc = {"t": numpyData[:,0]}
    if (len(counts) > 1):
        colListUniqueVec = []
        colListUniqueSc = []
        for col in colListUnique[1:]:
            if (counts[colListUnique.index(col)] == 3):
                colListUniqueVec.append(col)
            elif (counts[colListUnique.index(col)] == 1):
                colListUniqueSc.append(col)
        if (3 in counts[1:]):
            dimsVec = ["datatypeVec"] + dimsVec
            coordsVec["datatypeVec"] = colListUniqueVec
        if (1 in counts[1:]):
            dimsSc = ["datatypeSc"] + dimsSc
            coordsSc["datatypeSc"] = colListUniqueSc
    indexList = [1] + [sum(counts[:i+1]) for i in range(len(counts)) if i >= 1]
    lengthTableDimsList = counts[1:]
    if (3 in counts[1:]):
        tableArrayVec = xr.DataArray(np.concatenate(tuple([[numpyData[:,indexList[i-1]:indexList[i]]] for i in range(len(indexList)) if i >= 1 and np.abs(indexList[i-1]-indexList[i]) == 3]), axis=0), dims=dimsVec, coords=coordsVec)
    if (1 in counts[1:]):
        tableArraySc = xr.DataArray(np.concatenate(tuple([[numpyData[:,indexList[i-1]]] for i in range(len(indexList)) if i >= 1 and np.abs(indexList[i-1]-indexList[i]) == 1]), axis=0), dims=dimsSc, coords=coordsSc)
    dataset = None
    if (isinstance(tableArrayVec, xr.DataArray)):
        dataset = tableArrayVec.expand_dims('vec', 0).to_dataset('vec').rename({0: 'vec'})
        if (isinstance(tableArraySc, xr.DataArray)):
            dataset['sc'] = tableArraySc
    else:
        if (isinstance(tableArraySc, xr.DataArray)):
            dataset = tableArraySc.expand_dims('sc', 0).to_dataset('sc').rename({0: 'sc'})
    return dataset