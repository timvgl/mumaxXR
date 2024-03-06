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
from typing import Tuple, Union
from tqdm import tqdm
from itertools import islice

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
    def castable_int(self):
        try:
            int(self)
            return True
        except ValueError:
            return False
        
    def return_index_before_int(self):
        for char in list(self):
            if (custom_string.castable_int(char) == True):
                return list(self).index(char) -1

def castableList(inputList):
    try:
        if (len(list(inputList)) > 1):
            return True
        else:
            return False
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
curse(str, "return_index_before_int", custom_string.return_index_before_int)


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
        sweepParam=[],
    ):
        filename_or_obj = Path(filename_or_obj)
        coordsParam = None
        if (len(dirListToConcat) > 0):
            dirListToConcat = [Path(ovfFile) for ovfFile in dirListToConcat]
            if (filename_or_obj not in dirListToConcat):
                dirListToConcat = [filename_or_obj] + dirListToConcat
            if (sweepName == ''):
                sweepName = "unknown"
            if (isinstance(sweepParam, list)):
                if (len(sweepParam) > 0 and isinstance(sweepParam, (float, int))):
                    if (len(sweepParam) == len(dirListToConcat)):
                        coordsParam = np.array(sweepParam)
                elif (len(sweepParam) == 0):
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
        startingFileName = ''
        if (len(dirListToConcat) > 0):
            if (dirListToConcat[0].suffix != '.ovf'):
                ovfFilesList = [Path(file).name for file in sorted(list(dirListToConcat[0].glob('**/*000000.ovf')))]
                if ('m000000.ovf' in ovfFilesList):
                    startingFileName = 'm000000.ovf'
                elif ('u000000.ovf' in ovfFilesList):
                    startingFileName = 'u000000.ovf'
                else:
                    raise ValueError('Could not find standart m or u ovf file in directory. Please pass wirst ovf file directly into xarray.')
            else:
                startingFileName = dirListToConcat[0].name
            if (filename_or_obj.suffix != '.ovf'):
                filename_or_obj = filename_or_obj.joinpath(Path(startingFileName))
        mesh = self.read_mesh_from_ovf(filename_or_obj)
        #tmaxArray = dask.array.from_delayed(self.read_t_from_ovf(filename_or_obj), shape=(len(sorted(list(Path(filename_or_obj).parent.glob('**/' + Path(filename_or_obj).stem[Path(filename_or_obj).stem.return_index_before_int()] + '*.ovf')))), ), dtype=np.float32)
        dims = None
        coords = None
        shape = None
        data = None
        if (len(dirListToConcat) == 0):
            if (len(wavetype) == 0):
                dims = ['t', 'z', 'y', 'x', 'comp']
                data, tmaxArray = self.read_data_from_ovf(filename_or_obj, dtype=dtype, mesh=mesh, returnTData=True)
                coords = [tmaxArray, mesh.get_axis(2), mesh.get_axis(1), mesh.get_axis(0), np.arange(mesh.n_comp)]
                shape = [value.size for value in coords]
            else:
                dims = ['wavetype', 't', 'z', 'y', 'x', 'comp']
                data, tmaxArray = self.read_data_from_ovf(filename_or_obj, dtype=dtype, mesh=mesh, type=wavetype[0], returnTData=True)
                data = dask.array.expand_dims(data, 0)

                @dask.delayed
                def concat_data(data, wavetype, filename_or_obj, dtype, mesh) -> dask.array.Array:
                    for type in wavetype[1:]:
                        newData = self.read_data_from_ovf(filename_or_obj, dtype=dtype, mesh=mesh, type=type)
                        data = dask.array.concatenate((data, dask.array.expand_dims(newData, 0)), axis=0)
                    return data
                
                coords = [np.array(wavetype), tmaxArray, mesh.get_axis(2), mesh.get_axis(1), mesh.get_axis(0), np.arange(mesh.n_comp)]
                shape = [value.size for value in coords]
                data = dask.array.from_delayed(concat_data(data, wavetype, filename_or_obj, dtype, mesh), shape=tuple([value.size for value in coords]), dtype=dtype)
        else:
            if (len(wavetype) == 0):
                if (isinstance(sweepName, str)):
                    dims = [sweepName, 't', 'z', 'y', 'x', 'comp']
                    if (dirListToConcat[0].suffix != '.ovf'):
                        dirListToConcat[0] = dirListToConcat[0].joinpath(Path(startingFileName))
                    data, tmaxArray = self.read_data_from_ovf(dirListToConcat[0], dtype=dtype, mesh=mesh, returnTData=True)
                    data = dask.array.expand_dims(data, 0)

                    @dask.delayed
                    def concat_data(data, dirListToConcat, startingFileName, dtype, mesh) -> dask.array.Array:
                        for dir in dirListToConcat[1:]:
                            if (dir.suffix != '.ovf'):
                                dir = dir.joinpath(Path(startingFileName))
                            newData = self.read_data_from_ovf(dir, dtype=dtype, mesh=mesh)
                            data = dask.array.concatenate((data, dask.array.expand_dims(newData, 0)), axis=0)
                        return data
                
                    coords = [coordsParam, tmaxArray, mesh.get_axis(2), mesh.get_axis(1), mesh.get_axis(0), np.arange(mesh.n_comp)]
                    shape = [value.size for value in coords]
                    data = dask.array.from_delayed(concat_data(data, dirListToConcat, startingFileName, dtype, mesh), shape=tuple([value.size for value in coords]), dtype=dtype)
                else:
                    dims = sweepName + ['t', 'z', 'y', 'x', 'comp']
                    if (dirListToConcat[0].suffix != '.ovf'):
                        dirListToConcat[0] = dirListToConcat[0].joinpath(Path(startingFileName))                                
                    data, tmaxArray = self.read_data_from_ovf(dirListToConcat[0], mesh=mesh, returnTData=True)
                    data = dask.array.expand_dims(data, 0)

                    @dask.delayed
                    def concat_data(data, dirListToConcat, startingFileName, dtype, mesh) -> dask.array.Array:
                        for dir in dirListToConcat[1:]:
                            if (dir.suffix != '.ovf'):
                                dir = dir.joinpath(Path(startingFileName))
                            newData = self.read_data_from_ovf(dir, dtype=dtype, mesh=mesh)
                            data = dask.array.concatenate((data, dask.array.expand_dims(newData, 0)), axis=0)
                        return data
                    
                    shape = None
                    for i in range(np.transpose(np.asarray(coordsParam))[:,0].size):
                        if shape == None:
                            shape = (np.unique(np.transpose(np.asarray(coordsParam))[i,:]).size, )
                        else:
                            shape += (np.unique(np.transpose(np.asarray(coordsParam))[i,:]).size, )
                            
                    coords = [tmaxArray, mesh.get_axis(2), mesh.get_axis(1), mesh.get_axis(0), np.arange(mesh.n_comp)]
                    shape += tuple([axis.shape[0] for axis in coords])
                    coordsDir = []
                    for i in range(np.transpose(np.asarray(coordsParam))[:,0].size):
                        coordsDir.append(np.unique(np.transpose(np.asarray(coordsParam))[i,:]))
                    coords = coordsDir + coords
                    data = dask.array.from_delayed(concat_data(data, dirListToConcat, startingFileName, dtype, mesh), shape=tuple([value.size for value in coords]), dtype=dtype)
                    data = dask.array.reshape(data, shape=shape, merge_chunks=False).rechunk(tuple(reversed([list(reversed(list(shape)))[i] if i < 4 else 1 for i in range(len(list(shape)))])))
            else:
                if (isinstance(sweepName, str)):
                    dims = [sweepName, 'wavetype', 't', 'z', 'y', 'x', 'comp']
                    if (dirListToConcat[0].suffix != '.ovf'):
                        dirListToConcat[0] = dirListToConcat[0].joinpath(Path(startingFileName))
                    tmaxArray = self.read_data_from_ovf(dirListToConcat[0], dtype=dtype, mesh=mesh, returnTData=True, returnMeshData=False)

                    @dask.delayed
                    def concat_data(wavetype, dirListToConcat, startingFileName, dtype, mesh) -> dask.array.Array:
                        data = None
                        for dir in dirListToConcat:
                            if (data is None):
                                if (dir.suffix != '.ovf'):
                                    dir = dir.joinpath(Path(startingFileName))
                                subData = None
                                for type in wavetype:
                                    if (subData is None):
                                        subData = self.read_data_from_ovf(dir, dtype=dtype, mesh=mesh, type=type)
                                        subData = dask.array.expand_dims(subData, 0)
                                    else:
                                        newSubData = self.read_data_from_ovf(dir, dtype=dtype, mesh=mesh, type=type)
                                        subData = dask.array.concatenate((subData, dask.array.expand_dims(newSubData, 0)), axis=0)
                                data = dask.array.expand_dims(subData, 0)
                            else:
                                if (dir.suffix != '.ovf'):
                                    dir = dir.joinpath(Path(startingFileName))
                                subData = None
                                for type in wavetype:
                                    if (subData is None):
                                        subData = dask.array.expand_dims(self.read_data_from_ovf(dir, dtype=dtype, mesh=mesh, type=type), 0)
                                    else:
                                        newSubData = self.read_data_from_ovf(dir, dtype=dtype, mesh=mesh, type=type)
                                        subData = dask.array.concatenate((subData, dask.array.expand_dims(newSubData, 0)), axis=0)
                                data = dask.array.concatenate((data, dask.array.expand_dims(subData, 0)), axis=0)
                        return data

                    coords = [coordsParam, np.array(wavetype), tmaxArray, mesh.get_axis(2), mesh.get_axis(1), mesh.get_axis(0), np.arange(mesh.n_comp)]
                    shape = [value.size for value in coords]
                    data = dask.array.from_delayed(concat_data(wavetype, dirListToConcat, startingFileName, dtype, mesh), shape=tuple(shape), dtype=dtype)
                else:
                    dims = sweepName + ['wavetype', 't', 'z', 'y', 'x', 'comp']
                    if (dirListToConcat[0].suffix != '.ovf'):
                        dirListToConcat[0] = dirListToConcat[0].joinpath(Path(startingFileName))
                    tmaxArray = self.read_data_from_ovf(dirListToConcat[0], dtype=dtype, mesh=mesh, returnTData=True, returnMeshData=False)
                    @dask.delayed
                    def concat_data(wavetype, dirListToConcat, startingFileName, dtype, mesh) -> dask.array.Array:
                        data = None
                        for dir in dirListToConcat:
                            print(dir)
                            if (data is None):
                                if (dir.suffix != '.ovf'):
                                    dir = dir.joinpath(Path(startingFileName))
                                subData = None
                                for type in wavetype:
                                    if (subData is None):
                                        subData = self.read_data_from_ovf(dir, dtype=dtype, mesh=mesh, type=type)
                                        subData = dask.array.expand_dims(subData, 0)
                                    else:
                                        newSubData = self.read_data_from_ovf(dir, dtype=dtype, mesh=mesh, type=type)
                                        subData = dask.array.concatenate((subData, dask.array.expand_dims(newSubData, 0)), axis=0)
                                data = dask.array.expand_dims(subData, 0)
                            else:
                                if (dir.suffix != '.ovf'):
                                    dir = dir.joinpath(Path(startingFileName))
                                subData = None
                                for type in wavetype:
                                    if (subData is None):
                                        subData = dask.array.expand_dims(self.read_data_from_ovf(dir, dtype=dtype, mesh=mesh, type=type), 0)
                                    else:
                                        newSubData = self.read_data_from_ovf(dir, dtype=dtype, mesh=mesh, type=type)
                                        subData = dask.array.concatenate((subData, dask.array.expand_dims(newSubData, 0)), axis=0)
                                data = dask.array.concatenate((data, dask.array.expand_dims(subData, 0)), axis=0)
                        print('Done')
                        return data
                        
                    shape = None
                    for i in range(np.transpose(np.asarray(coordsParam))[:,0].size):
                        if shape == None:
                            shape = (np.unique(np.transpose(np.asarray(coordsParam))[i,:]).size, )
                        else:
                            shape += (np.unique(np.transpose(np.asarray(coordsParam))[i,:]).size, )

                    coords = [np.array(wavetype), tmaxArray, mesh.get_axis(2), mesh.get_axis(1), mesh.get_axis(0), np.arange(mesh.n_comp)]
                    shape += tuple([axis.shape[0] for axis in coords[-6:]])
                    coordsDir = []
                    for i in range(np.transpose(np.asarray(coordsParam))[:,0].size):
                        coordsDir.append(np.unique(np.transpose(np.asarray(coordsParam))[i,:]))
                    coords = coordsDir + coords
                    data = dask.array.from_delayed(concat_data(wavetype, dirListToConcat, startingFileName, dtype, mesh), shape=tuple([value.size for value in coords]), dtype=dtype)
                    data = dask.array.reshape(data, shape=shape, merge_chunks=False).rechunk(tuple(reversed([list(reversed(list(shape)))[i] if i < 4 else 1 for i in range(len(list(shape)))])))
        #print(str(mesh.tmax) + filename_or_obj.stem)
        dset = xr.DataArray(data, dims=dims, coords=coords).to_dataset(name="raw")
        dset.attrs["cellsize"] = mesh.cellsize
        dset.attrs["nodes"] = mesh.nodes
        dset.attrs["min_size"] = mesh.world_min
        dset.attrs["max_size"] = mesh.world_max
        dset.attrs["n_comp"] = mesh.n_comp
        dset.attrs["n_cells"] = mesh.number_of_cells
        dset = dset.chunk(dict(zip(dims, list(reversed([list(reversed(list(shape)))[i] if i < 4 else 1 for i in range(len(list(shape)))])))))
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
    
    def read_data_from_ovf(self, filename, mesh=None, dtype=np.float32, returnTData=False, type='', returnMeshData=True) -> Union[Tuple[dask.array.Array, dask.array.Array], dask.array.Array]:
        if (type != ''):
            filename = filename.parent.joinpath(Path(type + str(filename.name)[re.search(r"\d", filename.stem).start():]))
        fileList = sorted(list(Path(filename).parent.glob('**/' + Path(filename).stem[Path(filename).stem.return_index_before_int()] + '*.ovf')))

        data = None
        tData = []

        def get_line_index_t_from_ovf(filename) -> int:
            file = open(filename, "rb")
            for i in range(0, 27):
                line = file.readline()
                # clean up
                line = line.replace(b'# ', b'')
                line = line.replace(b'Desc: Total simulation time: ', b'tmax: ')
                line = line.replace(b'Desc: Time (s) : ', b'tmax: ')
                if ('tmax' in line.decode()):
                    return i
            return -1
            
        def read_t_from_ovf(filename, lineIndex) -> float:
            file = open(filename, "rb")
            footer_dict = dict()
            # discard headline
            line = list(islice(file, lineIndex, lineIndex+1))[0]
            line = line.replace(b'# ', b'')
            line = line.replace(b'Desc: Total simulation time: ', b'tmax: ')
            line = line.replace(b'Desc: Time (s) : ', b'tmax: ')
            attr, val = line.split(b': ')
            footer_dict[attr] = val.replace(b"\n", b"")
            if b' s' in footer_dict[b"tmax"]:
                tmax_string, _ = footer_dict[b"tmax"].split(b' s')
            else:
                tmax_string = footer_dict[b"tmax"]
            return float(tmax_string)

        @dask.delayed
        def create_delayedArray(filename, mesh, dtype) -> np.array:
            filename = Path(filename)
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
            data = data.reshape(mesh.nodes[2], mesh.nodes[1], mesh.nodes[0], n_comp)
            file.close()
            return data
        
        if returnTData == True:
            lineIndex = get_line_index_t_from_ovf(fileList[0])
            if (lineIndex == -1):
                raise ValueError('Could not find time in ovf file.')
            for filename in fileList:
                tData.append(read_t_from_ovf(filename, lineIndex))

        @dask.delayed
        def create_concatenatedArray(fileList, mesh, dtype) -> dask.array.Array: 
            data = None
            for filename in fileList:
                tmpData = dask.array.expand_dims(dask.array.from_delayed(create_delayedArray(filename, mesh, dtype), shape=(mesh.nodes[2], mesh.nodes[1], mesh.nodes[0], mesh.n_comp), dtype=np.float32), 0)
                if (data is None):
                    data = tmpData
                else:
                    data = dask.array.concatenate((data, tmpData), axis=0)
            return data
        data = None
        if (returnMeshData == True):
            data = dask.array.from_delayed(create_concatenatedArray(fileList, mesh, dtype), shape=(len(fileList), mesh.nodes[2], mesh.nodes[1], mesh.nodes[0], mesh.n_comp), dtype=dtype)
            data = data.rechunk((1, mesh.nodes[2], mesh.nodes[1], mesh.nodes[0], mesh.n_comp))
        if returnTData == True and returnMeshData == True:
            return data, np.array(tData)
        elif (returnTData == False and returnMeshData == True):
            return data
        elif (returnTData == True and returnMeshData == False):
            return np.array(tData)
        else:
            raise ValueError('returnTData or returnMeshData has to be True')
    
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