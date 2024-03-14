#!/usr/bin/env python
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import jpype
import numpy as np
import scyjava
import xarray as xr
from aicsimageio import dimensions, exceptions, transforms
from aicsimageio.readers.reader import Reader
from aicsimageio.types import PathLike, PhysicalPixelSizes
from aicsimageio.utils import io_utils
from fsspec import AbstractFileSystem
from fsspec.implementations.local import LocalFileSystem
from resource_backed_dask_array import (
    ResourceBackedDaskArray,
    resource_backed_dask_array,
)

from brainways.utils.image import ImageSizeHW, resize_image
from brainways.utils.qupath import download_qupath, is_qupath_downloaded


class QupathReader(Reader):
    _qupath_version = "0.5.0"
    _qupath_initialized = False
    ImageServerProvider = None
    BufferedImage = None
    redirect = None

    @staticmethod
    def _initialize_qupath():
        if QupathReader._qupath_initialized:
            return

        # TODO: move this code to some init function inside QupathReader
        if not is_qupath_downloaded(QupathReader._qupath_version):
            download_qupath()
        from paquo._logging import redirect
        from paquo.java import BufferedImage, ImageServerProvider, JClass, String
        from paquo.projects import DEFAULT_IMAGE_PROVIDER

        QupathReader.ImageServerProvider = ImageServerProvider
        QupathReader.BufferedImage = BufferedImage
        QupathReader.RegionRequest = JClass("qupath.lib.regions.RegionRequest")
        QupathReader.redirect = redirect
        QupathReader._image_provider = DEFAULT_IMAGE_PROVIDER
        QupathReader.JString = String

        QupathReader._qupath_initialized = True

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        QupathReader._initialize_qupath()
        try:
            if not isinstance(fs, LocalFileSystem):
                return False
            support = QupathReader.ImageServerProvider.getPreferredUriImageSupport(
                QupathReader.BufferedImage.class_, path
            )
            return support is not None
        except Exception:
            return False

    def __init__(self, image: PathLike, *, fs_kwargs: Dict[str, Any] = None):
        super().__init__(image)
        QupathReader._initialize_qupath()
        self._fs, self._path = io_utils.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs or {},
        )
        self._current_level = 0
        self.__current_server = None

        # Catch non-local file system
        if not isinstance(self._fs, LocalFileSystem):
            raise ValueError(
                "Cannot read Qupath from non-local file system. "
                f"Received URI: {self._path}, which points to {type(self._fs)}."
            )

        with QupathReader.redirect(stderr=True, stdout=True):
            try:
                img_uri = QupathReader._image_provider.uri(self._path)
                support = QupathReader.ImageServerProvider.getPreferredUriImageSupport(
                    QupathReader.BufferedImage, QupathReader.JString(str(img_uri))
                )
                if support is None:
                    self._builders = []
                else:
                    self._builders = list(support.getBuilders())
                self._scenes: Tuple[str, ...] = tuple(
                    f"Scene #{i+1}" for i in range(len(self._builders))
                )
            except RuntimeError:
                raise
            except Exception as e:
                raise exceptions.UnsupportedFileFormatError(
                    self.__class__.__name__, self._path
                ) from e

    def get_thumbnail(self, target_size: ImageSizeHW, channel: int):
        original_level = self.current_level
        target_downsample = max(
            self.dims.X / target_size[1], self.dims.Y / target_size[0]
        )
        thumbnail_levels = [
            i for i, d in enumerate(self.downsamples) if d <= target_downsample * 1.2
        ]
        thumbnail_level = (
            thumbnail_levels[-1] if thumbnail_levels else self.n_levels - 1
        )
        if thumbnail_level != original_level:
            self.set_level(thumbnail_level)
        thumbnail = self.get_image_data("YX", C=channel)
        if thumbnail_level != original_level:
            self.set_level(original_level)
        if thumbnail.shape[:2] != target_size:
            thumbnail = resize_image(thumbnail, size=target_size, keep_aspect=True)
        return thumbnail

    @property
    def current_level(self) -> int:
        return self._current_level

    def set_level(self, level: int):
        if level not in range(self.n_levels):
            raise ValueError(
                f"level must be between 0 and {self.n_levels - 1}, got {level}"
            )
        self._reset_self()
        self._current_level = level

    @property
    def downsamples(self) -> List[float]:
        with QupathReader.redirect(stderr=True, stdout=True):
            _md = self._current_server.getMetadata()
            return [float(x) for x in _md.getPreferredDownsamplesArray()]

    @property
    def channel_names(self) -> Optional[List[str]]:
        with QupathReader.redirect(stderr=True, stdout=True):
            return [
                str(self._current_server.getChannel(i).getName())
                for i in range(self.dims.C)
            ]

    @property
    def channel_colors(self) -> Optional[List[Tuple[int, int, int]]]:
        with QupathReader.redirect(stderr=True, stdout=True):
            colors = []
            for color_i in range(self.dims.C):
                color_int = self._current_server.getChannel(color_i).getColor()
                b = color_int & 255
                g = (color_int >> 8) & 255
                r = (color_int >> 16) & 255
                colors.append((r / 255, g / 255, b / 255))
            return colors

    def _reset_self(self) -> None:
        super()._reset_self()
        self.__current_server = None

    @property
    def _current_server(self):
        if self.__current_server is None:
            with QupathReader.redirect(stderr=True, stdout=True):
                self.__current_server = self._builders[
                    self._current_scene_index
                ].build()
        return self.__current_server

    @property
    def scenes(self) -> Tuple[str, ...]:
        return self._scenes

    @property
    def n_levels(self) -> int:
        with QupathReader.redirect(stderr=True, stdout=True):
            return int(self._current_server.nResolutions())

    def _read_delayed(self, level: Optional[int] = None) -> xr.DataArray:
        if level is None:
            level = self._current_level
        return self._to_xarray(delayed=True, level=level)

    def _read_immediate(self, level: Optional[int] = None) -> xr.DataArray:
        if level is None:
            level = self._current_level
        return self._to_xarray(delayed=False, level=level)

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        """
        Returns
        -------
        sizes: PhysicalPixelSizes
            Using available metadata, the floats representing physical pixel sizes for
            dimensions Z, Y, and X.

        Notes
        -----
        We currently do not handle unit attachment to these values. Please see the file
        metadata for unit information.
        """
        with QupathReader.redirect(stderr=True, stdout=True):
            pixel_calibration = self._current_server.getPixelCalibration()
            return PhysicalPixelSizes(
                Z=None,
                Y=float(pixel_calibration.getPixelHeightMicrons()),
                X=float(pixel_calibration.getPixelWidthMicrons()),
            )

    def _to_xarray(self, delayed: bool = True, level: int = 0) -> xr.DataArray:
        if delayed:
            tile_manager = TileManager(self._current_server, self._path, level=level)
            image_data = tile_manager.to_dask()
        else:
            with QupathReader.redirect(stderr=True, stdout=True):
                request = QupathReader.RegionRequest.createInstance(
                    self._path,
                    self.downsamples[level],
                    0,
                    0,
                    self._current_server.getWidth(),
                    self._current_server.getHeight(),
                    0,
                    0,
                )
                buffered_image = self._current_server.readRegion(request)
            image_data = buffered_image_to_numpy_array(buffered_image)  # TCZYX
            image_data = image_data[np.newaxis, :, np.newaxis, :, :]
        return xr.DataArray(
            image_data,
            dims=dimensions.DEFAULT_DIMENSION_ORDER_LIST,
        )

    def get_image_dask_pyramid_data(
        self, dimension_order_out: Optional[str] = None, **kwargs: Any
    ) -> List[da.Array]:
        pyramid_data = [
            self._read_delayed(level=level).data for level in range(self.n_levels)
        ]
        # If no out orientation, simply return current data as dask array
        if dimension_order_out is None:
            return pyramid_data

        # Transform and return
        return [
            transforms.reshape_data(
                data=level_data,
                given_dims=self.dims.order,
                return_dims=dimension_order_out,
                **kwargs,
            )
            for level_data in pyramid_data
        ]

    @staticmethod
    def qupath_version() -> str:
        """The version of the qupath jar being used."""
        raise NotImplementedError()


def _pixtype2javatype(pixeltype: int):
    FT = scyjava.jimport("loci.formats.FormatTools")
    fmt2type: Dict[int, str] = {
        FT.INT8: jpype.JInt,
        FT.UINT8: jpype.JInt,
        FT.INT16: jpype.JInt,
        FT.UINT16: jpype.JInt,
        FT.INT32: jpype.JInt,
        FT.UINT32: jpype.JInt,
        FT.FLOAT: jpype.JFloat,
        FT.DOUBLE: jpype.JDouble,
    }
    return fmt2type[pixeltype]


def _pixtype2dtype(pixeltype: int, little_endian: bool) -> np.dtype:
    """Convert a loci.formats PixelType integer into a numpy dtype."""
    FT = scyjava.jimport("loci.formats.FormatTools")
    fmt2type: Dict[int, str] = {
        FT.INT8: "i1",
        FT.UINT8: "u1",
        FT.INT16: "i2",
        FT.UINT16: "u2",
        FT.INT32: "i4",
        FT.UINT32: "u4",
        FT.FLOAT: "f4",
        FT.DOUBLE: "f8",
    }
    return np.dtype(("<" if little_endian else ">") + fmt2type[pixeltype])


@lru_cache(maxsize=1)
def _hide_memoization_warning() -> None:
    """HACK: this silences a warning about memoization for now

    An illegal reflective access operation has occurred
    https://github.com/ome/bioformats/issues/3659
    """
    import jpype

    System = jpype.JPackage("java").lang.System
    System.err.close()


class TileManager:
    def __init__(self, image_server, path, level: int = 0):
        self.image_server = image_server
        self.path = path
        self.level = level

        with QupathReader.redirect(stderr=True, stdout=True):
            tile_requests = list(
                self.image_server.getTileRequestManager().getTileRequestsForLevel(level)
            )
            tile_w = tile_requests[0].getImageWidth()
            tile_h = tile_requests[0].getImageHeight()
            self._tile_requests = {
                (tr.getImageY() // tile_h, tr.getImageX() // tile_w): tr
                for tr in tile_requests
            }

    def _dask_chunk(self, block_id: Tuple[int, ...]) -> np.ndarray:
        """Retrieve `block_id` from array.

        This function is for map_blocks (called in `to_dask`).
        If someone indexes a 5D dask array as `arr[0, 1, 2]`, then 'block_id'
        will be (0, 1, 2, 0, 0)
        """
        # Our convention is that the final dask array is in the order TCZYX, so
        # block_id will be coming in as (T, C, Z, Y, X).
        t, c, z, y, x, *_ = block_id

        tile_request = self._tile_requests[(y, x)]
        with QupathReader.redirect(stderr=True, stdout=True):
            request = tile_request.getRegionRequest()
            buffered_image = self.image_server.readRegion(request)
            # buffered_image = self.image_server.readTile(tile_request)
        im = buffered_image_to_numpy_array(buffered_image, channel=c)

        return im[np.newaxis, np.newaxis, np.newaxis]

    def to_dask(self) -> ResourceBackedDaskArray:
        """Create dask array for the specified or current series.

        Note: the order of the returned array will *always* be `TCZYX[r]`,
        where `[r]` refers to an optional RGB dimension with size 3 or 4.
        If the image is RGB it will have `ndim==6`, otherwise `ndim` will be 5.

        The returned object is a `ResourceBackedDaskArray`, which is a wrapper on
        a dask array that ensures the file is open when actually reading (computing)
        a chunk.  It has all the methods and behavior of a dask array.
        See: https://github.com/tlambert03/resource-backed-dask-array

        Returns
        -------
        ResourceBackedDaskArray
        """
        with QupathReader.redirect(stderr=True, stdout=True):
            chunks = (
                (1,),
                (1,) * self.image_server.nChannels(),
                (1,),
                self.tile_heights,
                self.tile_widths,
            )
            dtype = _pixtype2dtype(
                self.image_server.getPixelType().ordinal(), little_endian=True
            )
            arr = da.map_blocks(
                self._dask_chunk,
                chunks=chunks,
                dtype=dtype,
            )
            return resource_backed_dask_array(arr, self)

    def to_numpy(self) -> np.ndarray:
        """Create numpy array for the specified or current series.

        Note: the order of the returned array will *always* be `TCZYX[r]`,
        where `[r]` refers to an optional RGB dimension with size 3 or 4.
        If the image is RGB it will have `ndim==6`, otherwise `ndim` will be 5.

        Parameters
        ----------
        series : int, optional
            The series index to retrieve, by default None
        """
        return np.asarray(self.to_dask())

    @property
    def tile_widths(self):
        return tuple(
            t.getTileWidth() for (y, x), t in self._tile_requests.items() if y == 0
        )

    @property
    def tile_heights(self):
        return tuple(
            t.getTileHeight() for (y, x), t in self._tile_requests.items() if x == 0
        )

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def close(self):
        pass

    @property
    def closed(self) -> bool:
        return False


def buffered_image_to_numpy_array(
    buffered_image, channel: Optional[int] = None
) -> np.ndarray:
    w = buffered_image.getWidth()
    h = buffered_image.getHeight()
    if channel is None:
        numpy_image_channels = []
        for channel in range(int(buffered_image.getData().getNumBands())):
            buffer = buffered_image.getData().getSamples(
                0, 0, w, h, channel, jpype.JFloat[w * h]
            )
            numpy_image_channels.append(np.array(buffer).reshape((h, w)))
        numpy_image = np.stack(numpy_image_channels)
    else:
        buffer = buffered_image.getData().getSamples(
            0, 0, w, h, channel, jpype.JFloat[w * h]
        )
        numpy_image = np.array(buffer).reshape((h, w))
    return numpy_image
