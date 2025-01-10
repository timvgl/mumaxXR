import numpy as np
import xarray as xr
import dask
from pathlib import Path
import re
import sys
import os
from forbiddenfruit import curse
import warnings
from typing import Tuple, Union
from itertools import islice
from copy import deepcopy
from packaging.version import Version

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
        
    def check_last_six_chars(self):
        lastSixCharsInts = True
        selfRev = list(reversed(list(self)))
        for i in range(6):
            if (custom_string.castable_int(selfRev[i]) == False):
                lastSixCharsInts = False
        if lastSixCharsInts == False:
            raise ValueError("Last chars are not ints.")
        return len(list(self)) -7

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
curse(str, "check_last_six_chars", custom_string.check_last_six_chars)


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
    def __init__(self, filename, nodes, world_min, world_max, tmax, freq, n_comp, footer_dict, step_times=None, dtype=np.float32, isFFT=False):
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
        self.dtype = dtype
        self.isFFT = isFFT
        self.freq = freq
    def get_axis(self, i):
            return np.linspace(self.world_min[i], self.world_max[i], self.nodes[i])

    def get_cellsize(self, i):
        return (self.world_max[i] - self.world_min[i]) / self.nodes[i]

class OvfBackendArray(xr.backends.BackendArray):
    def __init__(
        self,
        filename_or_obj,
        dtype,
        lock,
        wavetype=[],
        dirListToConcat=[],
        sweepName='',
        sweepParam=[],
        removeOvfFiles=False,
        useEachNthOvfFile=0,
        singleLoad=False,
        t=None,
        sc=False
    ):
        self.filename_or_obj = filename_or_obj
        self.dtype = dtype
        self.lock = lock
        self.wavetype = wavetype
        self.dirListToConcat = dirListToConcat
        self.sweepName = sweepName
        self.sweepParam = sweepParam
        self.removeOvfFiles = removeOvfFiles
        self.useEachNthOvfFile = useEachNthOvfFile
        self.useEachList = []
        self.tmaxArray = t
        self.singleLoad = singleLoad
        self.sc = sc
        self.fileList = self.generate_array_file_link()
    
    def __getitem__(self, key: tuple):
        return xr.core.indexing.explicit_indexing_adapter(
            key,
            self.shape,
            xr.core.indexing.IndexingSupport.BASIC,
            self._raw_indexing_method,
        )

    def generate_array_file_link(self):
        fileList = []
        self.filename_or_obj = Path(self.filename_or_obj)
        coordsParam = None
        if (len(self.dirListToConcat) > 0):
            self.dirListToConcat = [Path(ovfFile) for ovfFile in self.dirListToConcat]
            if (self.filename_or_obj not in self.dirListToConcat):
                self.dirListToConcat = [self.filename_or_obj] + self.dirListToConcat
            if (self.sweepName == ''):
                self.sweepName = "unknown"
            if (isinstance(self.sweepParam, list)):
                if (len(self.sweepParam) > 0 and isinstance(self.sweepParam, (float, int))):
                    if (len(self.sweepParam) == len(self.dirListToConcat)):
                        coordsParam = np.array(self.sweepParam)
                elif (len(self.sweepParam) == 0):
                    coordsParam = np.array(range(len(self.dirListToConcat)))
                elif (len(self.sweepParam) > 0 and self.sweepParam.isinstance_recursive((int, float)) and self.sweepParam.is_homogenious()):
                    coordsParam = [np.array(param) for param in self.sweepParam]
                    if (len(self.sweepName) != len(list(self.sweepParam[0])) or not isinstance(self.sweepName, (list, tuple))):
                        if (not isinstance(self.sweepName, (list, tuple))):
                            self.sweepName = [self.sweepName]
                        if len(self.sweepName) < len(list(self.sweepParam[0])):
                            warnings.warn('Not enough labels for sweep labeling found. Appending ...')
                            for i in range(len(list(self.sweepParam[0])) - len(self.sweepName)):
                                self.sweepName.append('unknown')
                        else:
                            warnings.warn('Too many labels for sweep labeling found. Trimming ...')
                            self.sweepName = self.sweepName[:len(list(coordsParam[0]))]
                    sortIndices = []
                    for row in sorted(self.sweepParam):
                        sortIndices.append(self.sweepParam.index(row))
                    self.dirListToConcat = [self.dirListToConcat[i] for i in sortIndices]
                else:
                    raise ValueError('Provided coords are not homogenious. Tuples or lists in list must be the same size for all entries.')
                
        startingFileName = ''
        if (len(self.dirListToConcat) > 0):
            if (self.dirListToConcat[0].suffix != '.ovf'):
                ovfFilesList = [Path(file).name for file in sorted(list(self.dirListToConcat[0].glob('**/*000000.ovf')))]
                if (len(self.wavetype) > 0):
                    startingFileName = self.wavetype[0] + '000000.ovf'
                elif ('m000000.ovf' in ovfFilesList):
                    startingFileName = 'm000000.ovf'
                elif ('u000000.ovf' in ovfFilesList):
                    startingFileName = 'u000000.ovf'
                else:
                    startingFileName = ovfFilesList[0]
            else:
                startingFileName = self.dirListToConcat[0].name
            if (self.filename_or_obj.suffix != '.ovf'):
                self.filename_or_obj = self.filename_or_obj.joinpath(Path(startingFileName))
        else:
            if (self.filename_or_obj.suffix != '.ovf'):
                ovfFilesList = [Path(file).name for file in sorted(list(self.filename_or_obj.glob('**/*000000.ovf')))]
                if (len(self.wavetype) > 0 and Path(self.wavetype[0] + '000000.ovf').is_file()):
                    startingFileName = self.wavetype[0] + '000000.ovf'
                elif (len(self.wavetype) > 0):
                    ovfFilesList = [Path(file).name for file in sorted(list(self.filename_or_obj.glob('**/' + self.wavetype[0] + '*.ovf')))]
                    startingFileName = ovfFilesList[0]
                elif ('m000000.ovf' in ovfFilesList):
                    startingFileName = 'm000000.ovf'
                elif ('u000000.ovf' in ovfFilesList):
                    startingFileName = 'u000000.ovf'
                elif len(ovfFilesList) > 0:
                    startingFileName = ovfFilesList[0]
                else:
                    raise FileExistsError('Amigous which file to use for starting ...')
                self.filename_or_obj = self.filename_or_obj.joinpath(Path(startingFileName))
        self.mesh = self.read_mesh_from_ovf(self.filename_or_obj)
        if (len(self.wavetype) > 0):
            matchingTypes = []
            for elem in self.wavetype:
                if (elem in self.filename_or_obj.name):
                    matchingTypes.append(elem)
            replaceString = matchingTypes[np.argmax([len(elem) for elem in matchingTypes])]
            removeWT = []
            replaceOrgName = False
            for elem in self.wavetype:
                mesh = self.read_mesh_from_ovf(self.filename_or_obj.parent.joinpath(self.filename_or_obj.name.replace(replaceString, elem)))
                if (self.sc == True):
                    if mesh.n_comp != 1:
                        removeWT.append(elem)
                        if (replaceString in self.filename_or_obj.name):
                            replaceOrgName = True
                else:
                    if mesh.n_comp != 3:
                        removeWT.append(elem)
                        if (replaceString in self.filename_or_obj.name):
                            replaceOrgName = True
            for elem in removeWT:
                self.wavetype.remove(elem)
            if (replaceOrgName == True and len(self.wavetype) > 0):
                self.filename_or_obj = self.filename_or_obj.parent.joinpath(self.filename_or_obj.name.replace(replaceString, self.wavetype[0]))
            if (len(self.wavetype) == 0):
                self.shape = ()
                self.dims = []
                self.coords = []
                return np.asarray([])
        else:
            if self.read_mesh_from_ovf(self.filename_or_obj).n_comp != 1 and self.sc == True or self.read_mesh_from_ovf(self.filename_or_obj).n_comp != 3 and self.sc == False:
                self.shape = ()
                self.dims = []
                self.coords = []
                return np.asarray([])
            
        startingFileName = ''
        if (len(self.dirListToConcat) > 0):
            if (self.dirListToConcat[0].suffix != '.ovf'):
                ovfFilesList = [Path(file).name for file in sorted(list(self.dirListToConcat[0].glob('**/*000000.ovf')))]
                if (len(self.wavetype) > 0):
                    startingFileName = self.wavetype[0] + '000000.ovf'
                elif ('m000000.ovf' in ovfFilesList):
                    startingFileName = 'm000000.ovf'
                elif ('u000000.ovf' in ovfFilesList):
                    startingFileName = 'u000000.ovf'
                else:
                    startingFileName = ovfFilesList[0]
            else:
                startingFileName = self.dirListToConcat[0].name
            if (self.filename_or_obj.suffix != '.ovf'):
                self.filename_or_obj = self.filename_or_obj.joinpath(Path(startingFileName))
        else:
            if (self.filename_or_obj.suffix != '.ovf'):
                ovfFilesList = [Path(file).name for file in sorted(list(self.filename_or_obj.glob('**/*000000.ovf')))]
                if (len(self.wavetype) > 0 and self.filename_or_obj.parent.joinpath(Path(self.wavetype[0] + '000000.ovf')).is_file()):
                    startingFileName = self.wavetype[0] + '000000.ovf'
                elif (len(self.wavetype) > 0):
                    ovfFilesList = [Path(file).name for file in sorted(list(self.filename_or_obj.glob('**/' + self.wavetype[0] + '*.ovf')))]
                    startingFileName = ovfFilesList[0]
                elif ('m000000.ovf' in ovfFilesList):
                    startingFileName = 'm000000.ovf'
                elif ('u000000.ovf' in ovfFilesList):
                    startingFileName = 'u000000.ovf'
                elif len(ovfFilesList) > 0:
                    startingFileName = ovfFilesList[0]
                else:
                    raise FileExistsError('Amigous which file to for starting ...')
                self.filename_or_obj = self.filename_or_obj.joinpath(Path(startingFileName))
        self.mesh = self.read_mesh_from_ovf(self.filename_or_obj)

        dims = None
        coords = None
        shape = None
        if (len(self.dirListToConcat) == 0):
            if (len(self.wavetype) == 0):
                if self.sc == False:
                    dims = ['t', 'z', 'y', 'x', 'comp']
                else:
                    dims = ['t', 'z', 'y', 'x']
                if (self.tmaxArray is None):
                    fileList, self.tmaxArray = self.get_corresponding_files(self.filename_or_obj, returnTData=True)
                else:
                    fileList[0] = self.get_corresponding_files(self.filename_or_obj, returnTData=False)
                if self.sc == False:
                    coords = [self.tmaxArray, self.mesh.get_axis(2), self.mesh.get_axis(1), self.mesh.get_axis(0), np.arange(self.mesh.n_comp)]
                else:
                    coords = [self.tmaxArray, self.mesh.get_axis(2), self.mesh.get_axis(1), self.mesh.get_axis(0)]
                shape = [value.size for value in coords]
                fileList = np.asarray(fileList)
            else:
                if self.sc == False:
                    dims = ['wavetype', 't', 'z', 'y', 'x', 'comp']
                else:
                    dims = ['wavetypeSc', 't', 'z', 'y', 'x']
                fileList = [None]
                if (self.tmaxArray is None):
                    fileList[0], self.tmaxArray = self.get_corresponding_files(self.filename_or_obj, type=self.wavetype[0], returnTData=True)
                else:
                    fileList[0] = self.get_corresponding_files(self.filename_or_obj, returnTData=False)
                for type in self.wavetype[1:]:
                    fileList.append(self.get_corresponding_files(self.filename_or_obj, type=type))
                if self.sc == False:
                    coords = [np.array(self.wavetype), self.tmaxArray, self.mesh.get_axis(2), self.mesh.get_axis(1), self.mesh.get_axis(0), np.arange(self.mesh.n_comp)]
                else:
                    coords = [np.array(self.wavetype), self.tmaxArray, self.mesh.get_axis(2), self.mesh.get_axis(1), self.mesh.get_axis(0)]
                shape = [value.size for value in coords]
                fileList = np.asarray(fileList)
        else:
            if (len(self.wavetype) == 0):
                if (isinstance(self.sweepName, str)):
                    if (self.sc == False):
                        dims = [self.sweepName, 't', 'z', 'y', 'x', 'comp']
                    else:
                        dims = [self.sweepName, 't', 'z', 'y', 'x']
                    if (self.dirListToConcat[0].suffix != '.ovf'):
                        self.dirListToConcat[0] = self.dirListToConcat[0].joinpath(Path(startingFileName))
                    fileList = [None]
                    if (self.tmaxArray is None):
                        fileList[0], self.tmaxArray = self.get_corresponding_files(self.dirListToConcat[0], returnTData=True)
                    else:
                        fileList[0] = self.get_corresponding_files(self.dirListToConcat[0], returnTData=False)

                    for dir in self.dirListToConcat[1:]:
                        if (dir.suffix != '.ovf'):
                            dir = dir.joinpath(Path(startingFileName))
                        fileList.append(self.get_corresponding_files(dir))
                    if (self.sc == False):
                        coords = [coordsParam, self.tmaxArray, self.mesh.get_axis(2), self.mesh.get_axis(1), self.mesh.get_axis(0), np.arange(self.mesh.n_comp)]
                    else:
                        coords = [coordsParam, self.tmaxArray, self.mesh.get_axis(2), self.mesh.get_axis(1), self.mesh.get_axis(0)]
                    shape = [value.size for value in coords]
                    fileList = np.asarray(fileList)
                else:
                    if (self.sc == False):
                        dims = self.sweepName + ['t', 'z', 'y', 'x', 'comp']
                    else:
                        dims = self.sweepName + ['t', 'z', 'y', 'x']
                    if (self.dirListToConcat[0].suffix != '.ovf'):
                        self.dirListToConcat[0] = self.dirListToConcat[0].joinpath(Path(startingFileName))
                    fileList = [None]   
                    if (self.tmaxArray is None):                         
                        fileList[0], self.tmaxArray = self.get_corresponding_files(self.dirListToConcat[0], returnTData=True)
                    else:
                        fileList[0] = self.get_corresponding_files(self.dirListToConcat[0], returnTData=False)

                    for dir in self.dirListToConcat[1:]:
                        if (dir.suffix != '.ovf'):
                            dir = dir.joinpath(Path(startingFileName))
                        fileList.append(self.get_corresponding_files(dir))
                    
                    shape = None
                    for i in range(np.transpose(np.asarray(coordsParam))[:,0].size):
                        if shape == None:
                            shape = (np.unique(np.transpose(np.asarray(coordsParam))[i,:]).size, )
                        else:
                            shape += (np.unique(np.transpose(np.asarray(coordsParam))[i,:]).size, )
                    if (self.sc == False):
                        coords = [self.tmaxArray, self.mesh.get_axis(2), self.mesh.get_axis(1), self.mesh.get_axis(0), np.arange(self.mesh.n_comp)]
                    else:
                        coords = [self.tmaxArray, self.mesh.get_axis(2), self.mesh.get_axis(1), self.mesh.get_axis(0)]
                    shape += tuple([axis.shape[0] for axis in coords])
                    coordsDir = []
                    for i in range(np.transpose(np.asarray(coordsParam))[:,0].size):
                        coordsDir.append(np.unique(np.transpose(np.asarray(coordsParam))[i,:]))
                    coords = coordsDir + coords
                    if (self.sc == False):
                        fileList = np.reshape(fileList, list(shape)[0:-4])
                    else:
                        fileList = np.reshape(fileList, list(shape)[0:-3])
            else:
                if (isinstance(self.sweepName, str)):
                    if (self.sc == False):
                        dims = [self.sweepName, 'wavetype', 't', 'z', 'y', 'x', 'comp']
                    else:
                        dims = [self.sweepName, 'wavetypeSc', 't', 'z', 'y', 'x']
                    if (self.dirListToConcat[0].suffix != '.ovf'):
                        self.dirListToConcat[0] = self.dirListToConcat[0].joinpath(Path(startingFileName))
                    if (self.tmaxArray is None):
                        self.tmaxArray = self.get_corresponding_files(self.dirListToConcat[0], returnTData=True, returnMeshData=False)
                    for dir in self.dirListToConcat:
                        if (dir.suffix != '.ovf'):
                            dir = dir.joinpath(Path(startingFileName))
                        subFileList = []
                        for type in self.wavetype:
                            subFileList.append(self.get_corresponding_files(dir, type=type))
                        fileList.append(subFileList)

                    if (self.sc == False):
                        coords = [coordsParam, np.array(self.wavetype), self.tmaxArray, self.mesh.get_axis(2), self.mesh.get_axis(1), self.mesh.get_axis(0), np.arange(self.mesh.n_comp)]
                    else:
                        coords = [coordsParam, np.array(self.wavetype), self.tmaxArray, self.mesh.get_axis(2), self.mesh.get_axis(1), self.mesh.get_axis(0)]
                    shape = [value.size for value in coords]
                    fileList = np.asarray(fileList)
                else:
                    if (self.sc == False):
                        dims = self.sweepName + ['wavetype', 't', 'z', 'y', 'x', 'comp']
                    else:
                        dims = self.sweepName + ['wavetypeSc', 't', 'z', 'y', 'x']
                    if (self.dirListToConcat[0].suffix != '.ovf'):
                        self.dirListToConcat[0] = self.dirListToConcat[0].joinpath(Path(startingFileName))
                    if (self.tmaxArray is None):
                        self.tmaxArray = self.get_corresponding_files(self.dirListToConcat[0], returnTData=True, returnMeshData=False)

                    for dir in self.dirListToConcat:
                        if (dir.suffix != '.ovf'):
                            dir = dir.joinpath(Path(startingFileName))
                        subFileList = []
                        for type in self.wavetype:
                            subFileList.append(self.get_corresponding_files(dir, type=type))
                        fileList.append(subFileList)
                    shape = None
                    for i in range(np.transpose(np.asarray(coordsParam))[:,0].size):
                        if shape == None:
                            shape = (np.unique(np.transpose(np.asarray(coordsParam))[i,:]).size, )
                        else:
                            shape += (np.unique(np.transpose(np.asarray(coordsParam))[i,:]).size, )

                    if (self.sc == False):
                        coords = [np.array(self.wavetype), self.tmaxArray, self.mesh.get_axis(2), self.mesh.get_axis(1), self.mesh.get_axis(0), np.arange(self.mesh.n_comp)]
                        shape += tuple([axis.shape[0] for axis in coords[-6:]])
                    else:
                        coords = [np.array(self.wavetype), self.tmaxArray, self.mesh.get_axis(2), self.mesh.get_axis(1), self.mesh.get_axis(0)]
                        shape += tuple([axis.shape[0] for axis in coords[-5:]])
                    coordsDir = []
                    for i in range(np.transpose(np.asarray(coordsParam))[:,0].size):
                        coordsDir.append(np.unique(np.transpose(np.asarray(coordsParam))[i,:]))
                    coords = coordsDir + coords
                    try:
                        if (self.sc == False):
                            fileList = np.reshape(fileList, list(shape)[0:-4])
                        else:
                            fileList = np.reshape(fileList, list(shape)[0:-3])
                    except:
                        print(fileList)
                        exit()
        #print(str(mesh.tmax) + filename_or_obj.stem)
        self.dims = dims
        self.coords = coords
        self.shape = shape
        return fileList
    
    def _raw_indexing_method(self, key: tuple) -> np.ndarray:
        if self.sc == False:
            negativeEnd = -4
        else:
            negativeEnd = -3
        filenames = self.fileList[tuple(list(key)[:negativeEnd])]
        if not isinstance(filenames, np.ndarray):
            filenames = np.asarray([filenames])
        data = None
        for filename in filenames.flatten():
            if (not isinstance(filename, dict)):
                if (data is None):
                    data = self.read_data_from_ovf(filename)
                    if (data.shape == data[tuple(list(key)[negativeEnd:])].shape and self.removeOvfFiles == True):
                        filename.unlink()
                    data = data[tuple(list(key)[negativeEnd:])]
                    if (len(list(data.shape)) < len(list(key))):
                        for i in range(len(list(key))-len(list(data.shape))):
                            data = np.expand_dims(data, 0)
                else:
                    newData = self.read_data_from_ovf(filename)
                    if (newData.shape == newData[tuple(list(key)[negativeEnd:])].shape and self.removeOvfFiles == True):
                        filename.unlink()
                    newData = newData[tuple(list(key)[negativeEnd:])]
                    if (len(list(newData.shape)) < len(list(key))):
                        for i in range(len(list(key))-len(list(newData.shape))):
                            newData = np.expand_dims(newData, 0)
                    data = np.vstack((data, newData))
            else:
                for keyInter in filename.keys():
                    if (not isinstance(filename[keyInter], tuple)):
                        if (data is None):
                            data = self.read_data_from_ovf(filename[keyInter])
                            data = data[tuple(list(key)[negativeEnd:])]
                            if (len(list(data.shape)) < len(list(key))):
                                for i in range(len(list(key))-len(list(data.shape))):
                                    data = np.expand_dims(data, 0)
                        else:
                            newData = self.read_data_from_ovf(filename[keyInter])
                            newData = newData[tuple(list(key)[negativeEnd:])]
                            if (len(list(newData.shape)) < len(list(key))):
                                for i in range(len(list(key))-len(list(newData.shape))):
                                    newData = np.expand_dims(newData, 0)
                            data = np.vstack((data, newData))
                    else:
                        if (data is None):
                            dataBefore = self.read_data_from_ovf(filename[keyInter][1])
                            dataAfter = self.read_data_from_ovf(filename[keyInter][3])
                            dataSlope = (dataAfter - dataBefore) / (filename[keyInter][2] - filename[keyInter][0])
                            data = dataBefore + dataSlope * (keyInter[0] - filename[keyInter][0])
                            data = data[tuple(list(key)[negativeEnd:])]
                            if (len(list(data.shape)) < len(list(key))):
                                for i in range(len(list(key))-len(list(data.shape))):
                                    data = np.expand_dims(data, 0)
                        else:
                            dataBefore = self.read_data_from_ovf(filename[keyInter][1])
                            dataAfter = self.read_data_from_ovf(filename[keyInter][3])
                            dataSlope = (dataAfter - dataBefore) / (filename[keyInter][2] - filename[keyInter][0])
                            newData = dataBefore + dataSlope * (keyInter[0] - filename[keyInter][0])
                            newData = newData[tuple(list(key)[negativeEnd:])]
                            if (len(list(newData.shape)) < len(list(key))):
                                for i in range(len(list(key))-len(list(newData.shape))):
                                    newData = np.expand_dims(newData, 0)
                            data = np.vstack((data, newData))


        data = np.reshape(data, filenames.shape + tuple(list(data.shape)[negativeEnd:]))
        return data

    def read_mesh_from_ovf(self, filename):
        with self.lock, open(filename, "rb") as file:
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
                line = line.replace(b'Desc: Frequency:', b'freq: ')
                footer.append(line)
                attr, val = line.split(b': ')
                footer_dict[attr] = val.replace(b"\n", b"")

            xnodes = int(footer_dict[b"xnodes"])
            ynodes = int(footer_dict[b"ynodes"])
            znodes = int(footer_dict[b"znodes"])

            nodes = np.array([xnodes, ynodes, znodes])
            xmin = float(footer_dict[b"xmin"] if b"xmin" in footer_dict else footer_dict[b"k_xmin"])
            ymin = float(footer_dict[b"ymin"] if b"ymin" in footer_dict else footer_dict[b"k_ymin"])
            zmin = float(footer_dict[b"zmin"] if b"zmin" in footer_dict else footer_dict[b"k_zmin"])



            world_min = np.array([xmin, ymin, zmin])

            xmax = float(footer_dict[b"xmax"] if b"xmax" in footer_dict else footer_dict[b"k_xmax"])
            ymax = float(footer_dict[b"ymax"] if b"ymax" in footer_dict else footer_dict[b"k_ymax"])
            zmax = float(footer_dict[b"zmax"] if b"zmax" in footer_dict else footer_dict[b"k_zmax"])



            world_max = np.array([xmax, ymax, zmax])
            tmax, freq = None, None
            if b"tmax" in footer_dict:
                if b' s' in footer_dict[b"tmax"]:
                    tmax_string, _ = footer_dict[b"tmax"].split(b' s')
                else:
                    tmax_string = footer_dict[b"tmax"]
                tmax = float(tmax_string)
            elif b"freq" in footer_dict:
                if b' Hz' in footer_dict[b"freq"]:
                    freq_string, _ = footer_dict[b"freq"].split(b' Hz')
                else:
                    freq_string = footer_dict[b"freq"]
                freq = float(freq_string)

            if b'valuedim' in footer_dict:
                n_comp = int(footer_dict[b'valuedim'])
            else:
                n_comp = 1
            
            
            if footer_dict[b"Begin"] == b"Data Binary 4+4":
                self.dtype = np.complex64
            else:
                self.dtype = np.float32

            isFFT = b"k_xmin" in footer_dict and b"k_xmax" in footer_dict or b"k_ymin" in footer_dict and b"k_ymax" in footer_dict or b"k_zmin" in footer_dict and b"k_zmax" in footer_dict
            
        return MumaxMesh(filename, nodes, world_min, world_max, tmax, freq, n_comp, footer_dict, dtype=self.dtype, isFFT=isFFT)
    
    def get_corresponding_files(self, filename, returnTData=False, type='', returnMeshData=True) -> Union[Tuple[list, np.ndarray], list, np.ndarray]:
        def get_line_index_t_from_ovf(filename, lock) -> int:
            with lock, open(filename, "rb") as file:
                for i in range(0, 27):
                    line = file.readline()
                    # clean up
                    line = line.replace(b'# ', b'')
                    line = line.replace(b'Desc: Total simulation time: ', b'tmax: ')
                    line = line.replace(b'Desc: Time (s) : ', b'tmax: ')
                    line = line.replace(b'Desc: Frequency:', b'freq: ')
                    if ('tmax' in line.decode() or 'freq' in line.decode()):
                        return i
            return -1
        
        def read_t_from_ovf(filename, lineIndex, lock) -> float:
            with lock, open(filename, "rb") as file:
                footer_dict = dict()
                # discard headline
                try:
                    line = list(islice(file, lineIndex, lineIndex+1))[0]
                    line = line.replace(b'# ', b'')
                    line = line.replace(b'Desc: Total simulation time: ', b'tmax: ')
                    line = line.replace(b'Desc: Time (s) : ', b'tmax: ')
                    line = line.replace(b'Desc: Frequency:', b'freq: ')
                    attr, val = line.split(b': ')
                    footer_dict[attr] = val.replace(b"\n", b"")
                    if b'tmax' in footer_dict:
                        if b' s' in footer_dict[b"tmax"]:
                            tmax_string, _ = footer_dict[b"tmax"].split(b' s')
                        else:
                            tmax_string = footer_dict[b"tmax"]
                        return float(tmax_string)
                    elif b'freq' in footer_dict:
                        if b' Hz' in footer_dict[b"freq"]:
                            freq_string, _ = footer_dict[b"freq"].split(b' Hz')
                        else:
                            freq_string = footer_dict[b"freq"]
                        return float(freq_string)
                    else:
                        raise ValueError("Could neither find time nor frequency in ovf file: " + file.name)
                except IndexError:
                    print('Found ovf file of not finished sim. Stopping from: ' + filename.name)
                    return None
        
        tData = []
        if (type != ''):
            try:
                filename = filename.parent.joinpath(Path(type + str(filename.name)[re.search(r"\d", filename.stem).start():]))
            except AttributeError:
                pass
        try:
            if (self.singleLoad == False):
                fileList = sorted(list(Path(filename).parent.glob('**/' + Path(filename).stem[:Path(filename).stem.check_last_six_chars()+1] + '[0-9][0-9][0-9][0-9][0-9][0-9].ovf')))
            else:
                raise TypeError
        except TypeError:
            fileList = [filename]
        if (self.useEachNthOvfFile != 0 and self.useEachNthOvfFile != 1):
            if (len(self.useEachList) == 0):
                useIndices = list(range(0, len(fileList), self.useEachNthOvfFile))
                self.useEachList = [element for element in list(range(len(fileList))) if element not in useIndices]
            fileList = np.delete(fileList, self.useEachList, None).tolist()
        if (self.tmaxArray is not None and self.tmaxArray.size != len(fileList)):
            def find_nearest(timeList: list, value: float) -> tuple:
                array = np.asarray(timeList)
                idx = (np.abs(timeList - value)).argmin()
                smallestDiffValue = timeList[idx]
                if (smallestDiffValue < value and idx < np.array(timeList).size):
                    return (idx, smallestDiffValue, timeList[idx], 'lower')
                elif (smallestDiffValue > value and idx > 0):
                    return (idx, timeList[idx-1], smallestDiffValue, 'higher')
                else:
                    return tuple([idx, smallestDiffValue])
            
            lineIndex = get_line_index_t_from_ovf(fileList[0], self.lock)
            if (lineIndex == -1):
                raise ValueError('Could not find time in ovf file.')  
            
            for filename in fileList:
                tFrame = read_t_from_ovf(filename, lineIndex, self.lock)
                if not tFrame is None:
                    tData.append(tFrame)
                else:
                    break

            if (self.tmaxArray.size < len(fileList)):#.replace(filename.parents[1].__str__(), '')
                print(filename.parents[0].__str__() + ' contains more ovf-files than expected. Interpolating to ' + str(self.tmaxArray.size) + ' timesteps ...')
                interpolatedfileList = []
                for i in range(self.tmaxArray.size):
                    nearest = list(find_nearest(tData, self.tmaxArray[i]))
                    if (len(nearest) == 2):
                        interpolatedfileList.append({(self.tmaxArray[i]): fileList[nearest[0]]})
                    else:
                        if (nearest[3] == 'lower'):
                            interpolatedfileList.append({(self.tmaxArray[i]): (nearest[1], fileList[nearest[0]], nearest[2], fileList[nearest[0]+1])})
                        elif (nearest[3] == 'higher'):
                            interpolatedfileList.append({(self.tmaxArray[i]): (nearest[1], fileList[nearest[0]-1], nearest[2], fileList[nearest[0]])})
            elif (self.tmaxArray.size > len(fileList)):#.replace(filename.parents[1].__str__(), '')
                print(filename.parents[0].__str__() + ' contains less ovf-files than expected. Interpolating to ' + str(self.tmaxArray.size) + ' timesteps ...')
                interpolatedfileList = []
                for i in range(self.tmaxArray.size):
                    nearest = list(find_nearest(tData, self.tmaxArray[i]))
                    if (len(nearest) == 2):
                        interpolatedfileList.append({(self.tmaxArray[i]): fileList[nearest[0]]})
                    else:
                        if (nearest[3] == 'lower'):
                            interpolatedfileList.append({(self.tmaxArray[i]): (nearest[1], fileList[nearest[0]], nearest[2], fileList[nearest[0]+1])})
                        elif (nearest[3] == 'higher'):
                            interpolatedfileList.append({(self.tmaxArray[i]): (nearest[1], fileList[nearest[0]-1], nearest[2], fileList[nearest[0]])})
            fileList = interpolatedfileList

        if returnTData == True:
            lineIndex = get_line_index_t_from_ovf(fileList[0], self.lock)
            if (lineIndex == -1):
                raise ValueError('Could not find time in ovf file.')
            for filename in fileList:
                tData.append(read_t_from_ovf(filename, lineIndex, self.lock))     

        if returnTData == True and returnMeshData == True:
            return fileList, np.array(tData)
        elif (returnTData == False and returnMeshData == True):
            return fileList
        elif (returnTData == True and returnMeshData == False):
            return np.array(tData)
        else:
            raise ValueError('returnTData or returnMeshData has to be True')
        
    def read_data_from_ovf(self, filename) -> np.ndarray:
        filename = Path(filename)
        with self.lock, open(filename, "rb") as file:
            if self.dtype == np.complex64:
                _binary = 8
            else:
                _binary = 4

            data_start_pos = 0

            for i in range(0, 46):
                line = file.readline()
                if b'# Begin: Data' in line:
                    # the first binary number is a control number specified in the ovf format
                    file.read(4)
                    data_start_pos = file.tell()
                    break
                if (i == 45):
                    raise ValueError("Error: %s has no well formatted data segment." % filename)

            file.seek(data_start_pos, 0)
            if _binary == 8:
                size = int(self.mesh.n_comp * self.mesh.number_of_cells * 2)
                data = np.fromfile(file, dtype=np.float32, count=size)
            else:
                size = int(self.mesh.n_comp * self.mesh.number_of_cells)
                data = np.fromfile(file, dtype=self.dtype, count=size)
            if _binary == 8 and self.mesh.isFFT == True:
                try:
                    if (self.sc == False):
                        data = data.reshape(self.mesh.nodes[2], self.mesh.nodes[1], self.mesh.nodes[0], 2, self.mesh.n_comp)
                        data = data[...,0,:] + 1j*data[...,1,:]
                    else:
                        data = data.reshape(self.mesh.nodes[2], self.mesh.nodes[1], self.mesh.nodes[0], 2)
                        data = data[...,0] + 1j*data[...,1]
                except ValueError:
                    print("Couldnot reshape data from file " + Path(filename).name)
                    print(str(data.shape) + ' vs ' + str((self.mesh.nodes[2], self.mesh.nodes[1], self.mesh.nodes[0], self.mesh.n_comp, 2)) + ' or ' + str((self.mesh.nodes[2], self.mesh.nodes[1], self.mesh.nodes[0], 2)))
                    exit()
                
                if data.shape[0] > 1:
                    if  data.shape[0] % 2 == 0:
                        data[int(data.shape[0] / 2) -1] = data[int(data.shape[0] / 2)]
                if data.shape[1] > 1:
                    if  data.shape[1] % 2 == 0:
                        data[:,int(data.shape[1] / 2) -1] = data[:,int(data.shape[1] / 2)]
            elif _binary == 4 and self.mesh.isFFT == True or _binary == 4 and self.mesh.isFFT == False:
                if (self.sc == False):
                    data = data.reshape(self.mesh.nodes[2], self.mesh.nodes[1], self.mesh.nodes[0], self.mesh.n_comp)
                else:
                    data = data.reshape(self.mesh.nodes[2], self.mesh.nodes[1], self.mesh.nodes[0])
            else:
                raise ValueError("Non FFT complex data supported.")

        return data
        

class OvfEngine(xr.backends.BackendEntrypoint):
    open_dataset_parameters = ["filename_or_obj", "drop_variables", "dtype", "wavetype", "dirListToConcat", "sweepName", "sweepParam", "removeOvfFiles", "useEachNthOvfFile", "singleLoad", "t"]      
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
        removeOvfFiles=False,
        useEachNthOvfFile=0,
        singleLoad=False,
        t=None
    ):
    
        sc = False
        backend_array = OvfBackendArray(
            deepcopy(filename_or_obj),
            deepcopy(dtype),
            dask.utils.SerializableLock(),
            deepcopy(wavetype),
            deepcopy(dirListToConcat),
            deepcopy(sweepName),
            deepcopy(sweepParam),
            deepcopy(removeOvfFiles),
            deepcopy(useEachNthOvfFile),
            deepcopy(singleLoad),
            deepcopy(t),
            sc
        )
        data = xr.core.indexing.LazilyIndexedArray(backend_array)

        sc = True

        backend_array_sc = OvfBackendArray(
            deepcopy(filename_or_obj),
            deepcopy(dtype),
            dask.utils.SerializableLock(),
            deepcopy(wavetype),
            deepcopy(dirListToConcat),
            deepcopy(sweepName),
            deepcopy(sweepParam),
            deepcopy(removeOvfFiles),
            deepcopy(useEachNthOvfFile),
            deepcopy(singleLoad),
            deepcopy(t),
            sc
        )


        dataSc = xr.core.indexing.LazilyIndexedArray(backend_array_sc)

        if backend_array.mesh.isFFT == True:
            notOneDims = ['k_x', 'k_y', 'k_z', 'comp']
            defaultChunks = {}
            if (backend_array.shape != ()):
                if backend_array.mesh.tmax is None and backend_array.mesh.freq is not None:
                    defaultChunks['f'] = 1
                    notOneDims.append('f')
                for dim in backend_array.dims:
                    if (dim not in notOneDims):
                        defaultChunks[dim] = 1
                defaultChunks['k_z'] = backend_array.coords[-4].size
                defaultChunks['k_y'] = backend_array.coords[-3].size
                defaultChunks['k_x'] = backend_array.coords[-2].size
                defaultChunks['comp'] = backend_array.coords[-1].size
                

            if (backend_array_sc.shape != ()):
                defaultChunksSc = {}
                if backend_array_sc.mesh.tmax is None and backend_array_sc.mesh.freq is not None:
                    defaultChunks['f'] = 1
                    notOneDims.append('f')
                for dim in backend_array_sc.dims:
                    if (dim not in notOneDims):
                        defaultChunksSc[dim] = 1
                defaultChunksSc['k_z'] = backend_array_sc.coords[-3].size
                defaultChunksSc['k_y'] = backend_array_sc.coords[-2].size
                defaultChunksSc['k_x'] = backend_array_sc.coords[-1].size
            replaceDims = ['x', 'y', 'z']
            if (backend_array.shape != ()):
                for i in range(len(backend_array.dims)):
                    if backend_array.dims[i] in replaceDims:
                        backend_array.dims[i] = 'k_' + backend_array.dims[i]
                    elif backend_array.mesh.freq is not None and backend_array.mesh.tmax is None and backend_array.dims[i] == 't':
                        backend_array.dims[i] = 'f'
                var = xr.Variable(dims=backend_array.dims, data=data)
                var.encoding["preferred_chunks"] = defaultChunks
            if (backend_array_sc.shape != ()):
                for i in range(len(backend_array_sc.dims)):
                    if backend_array_sc.dims[i] in replaceDims:
                        backend_array_sc.dims[i] = 'k_' + backend_array_sc.dims[i]
                    elif backend_array_sc.mesh.freq is not None and backend_array_sc.mesh.tmax is None and backend_array_sc.dims[i] == 't':
                        backend_array_sc.dims[i] = 'f'
                varSc = xr.Variable(dims=backend_array_sc.dims, data=dataSc)
                varSc.encoding["preferred_chunks"] = defaultChunksSc
            if (backend_array_sc.shape != () and backend_array.shape != ()):
                dataset = xr.Dataset({'raw': var, 'rawSc': varSc}, coords=dict(zip(backend_array.dims + ['wavetypeSc'], backend_array.coords + [backend_array_sc.coords[backend_array_sc.dims.index('wavetypeSc')]])))
            elif (backend_array_sc.shape != () and backend_array.shape == ()):
                dataset = xr.Dataset({'rawSc': varSc}, coords=dict(zip(backend_array_sc.dims, backend_array_sc.coords)))
            elif (backend_array_sc.shape == () and backend_array.shape != ()):
                dataset = xr.Dataset({'raw': var}, coords=dict(zip(backend_array.dims, backend_array.coords)))
        else:
            notOneDims = ['x', 'y', 'z', 'comp']
            defaultChunks = {}
            if (backend_array.shape != ()):
                for dim in backend_array.dims:
                    if (dim not in notOneDims):
                        defaultChunks[dim] = 1
                defaultChunks['z'] = backend_array.coords[-4].size
                defaultChunks['y'] = backend_array.coords[-3].size
                defaultChunks['x'] = backend_array.coords[-2].size
                defaultChunks['comp'] = backend_array.coords[-1].size

            if (backend_array_sc.shape != ()):
                defaultChunksSc = {}
                for dim in backend_array_sc.dims:
                    if (dim not in notOneDims):
                        defaultChunksSc[dim] = 1
                defaultChunksSc['z'] = backend_array_sc.coords[-3].size
                defaultChunksSc['y'] = backend_array_sc.coords[-2].size
                defaultChunksSc['x'] = backend_array_sc.coords[-1].size
            if (backend_array.shape != ()):
                var = xr.Variable(dims=backend_array.dims, data=data)
                var.encoding["preferred_chunks"] = defaultChunks
            if (backend_array_sc.shape != ()):
                varSc = xr.Variable(dims=backend_array_sc.dims, data=dataSc)
                varSc.encoding["preferred_chunks"] = defaultChunksSc
            if (backend_array_sc.shape != () and backend_array.shape != ()):
                dataset = xr.Dataset({'raw': var, 'rawSc': varSc}, coords=dict(zip(backend_array.dims + ['wavetypeSc'], backend_array.coords + [backend_array_sc.coords[backend_array_sc.dims.index('wavetypeSc')]])))
            elif (backend_array_sc.shape != () and backend_array.shape == ()):
                dataset = xr.Dataset({'rawSc': varSc}, coords=dict(zip(backend_array_sc.dims, backend_array_sc.coords)))
            elif (backend_array_sc.shape == () and backend_array.shape != ()):
                dataset = xr.Dataset({'raw': var}, coords=dict(zip(backend_array.dims, backend_array.coords)))
        dataset.attrs["cellsize"] = backend_array.mesh.cellsize
        dataset.attrs["nodes"] = backend_array.mesh.nodes
        dataset.attrs["min_size"] = backend_array.mesh.world_min
        dataset.attrs["max_size"] = backend_array.mesh.world_max
        dataset.attrs["n_comp"] = backend_array.mesh.n_comp
        dataset.attrs["n_cells"] = backend_array.mesh.number_of_cells
        return dataset

    def guess_can_open(self, filename_or_obj):
        filename_or_obj = Path(filename_or_obj)
        if (filename_or_obj.suffix != '.ovf'):
            if filename_or_obj.is_dir() == True and len(list(filename_or_obj.glob('**/*000000.ovf'))) == 0 or filename_or_obj.is_dir() == False:
                return False
            else:
                return True
        else:
            return True
            

def convert_TableFile_to_dataset(tableTxtFilePath, prefix=''):
    tableArrayVec = None
    tableArraySc = None
    with HiddenPrints():
        try:
            numpyData = np.loadtxt(tableTxtFilePath, dtype=np.float32)
        except ValueError:
            numpyData = np.loadtxt(tableTxtFilePath, dtype=str)
            if prefix == '':
                prefix = '_'.join(numpyData[0, 0].split('_')[:-1])
            if Version(str(np.__version__)) <= Version('1.26.1'):
                negativeIndex = np.flatnonzero(np.core.defchararray.find(numpyData[:,0],prefix)!=-1).astype(int)
                positiveIndex = np.flatnonzero(np.core.defchararray.find(numpyData[:,0],prefix)==-1).astype(int)
            else:
                negativeIndex = np.where(np.char.find(np.char.lower(numpyData[:,0]), prefix) > -1)[0]
                positiveIndex = np.where(np.char.find(np.char.lower(numpyData[:,0]), prefix) <= -1)[0]
            numpyDataNeg = np.flip(np.hstack((-np.expand_dims(np.char.replace(numpyData[negativeIndex,0], prefix + '_', '').astype(np.float32), axis=-1), numpyData[negativeIndex,1:].astype(np.float32))), axis=0)
            numpyDataPos = np.hstack((np.expand_dims(numpyData[positiveIndex,0].astype(np.float32), axis=-1), numpyData[positiveIndex,1:].astype(np.float32)))
            numpyData = np.vstack((numpyDataNeg, numpyDataPos))
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
