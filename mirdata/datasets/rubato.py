"""RUBATO Dataset Loader

.. admonition:: Dataset Info
    :class: dropdown

    Please include the following information at the top level docstring for the dataset's module `dataset.py`:

    1. Describe annotations included in the dataset
    2. Indicate the size of the datasets (e.g. number files and duration, hours)
    3. Mention the origin of the dataset (e.g. creator, institution)
    4. Describe the type of music included in the dataset
    5. Indicate any relevant papers related to the dataset
    6. Include a description about how the data can be accessed and the license it uses (if applicable)

"""

import csv
import json
import os
from typing import BinaryIO, Optional, TextIO, Tuple

# -- import whatever you need here and remove
# -- example imports you won't use
import librosa
import numpy as np
from smart_open import (
    open,
)  # if you use the open function, make sure you include this line!

from mirdata import download_utils, core, annotations

# -- Add any relevant citations here
BIBTEX = """
Missing TISMIR article citation for RUBATO dataset #TODO
@dataset{venohr_rubato_2026,
  author       = {Venohr, Yannik and
                  Schafferhans, Leonie and
                  Ding, Yiwei and
                  Schwär, Simon and
                  Weiß, Christof},
  title        = {RUBATO: A Multi-Version Benchmark for Robust Music
                   Transcription and Analysis
                  },
  month        = oct,
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.23472735},
  url          = {https://doi.org/10.5281/zenodo.23472735},
}
"""


# -- INDEXES specifies different versions of a dataset
# -- "default" and "test" specify which key should be used by default and when running tests
# -- Each index is defined by {"version": core.Index instance}
# -- | filename: index name
# -- | url: Zenodo direct download link of the index (will be available afer the index upload is
# -- accepted to Audio Data Loaders Zenodo community).
# -- | checksum: Checksum of the index hosted at Zenodo.
# -- Direct url for download and checksum can be found in the Zenodo entry of the dataset.
# -- Sample index is a mini-version that makes it easier to test a large datasets.
# -- There must be a local sample index for testing for each remote index.
INDEXES = {
    "default": "1.0",
    "test": "sample",
    "1.0": core.Index(
        filename="rubato_index_1.0.json",
        url="https://zenodo.org/records/14007830/files/rubato_index_1.0.json?download=1",  # TODO -- direct download link for the index hosted at Zenodo
        checksum="6e1276bdab6de05446ddbbc75e6f6cbe",  # TODO: -- add the checksum of the index hosted at Zenodo
    ),
    "sample": core.Index(filename="rubato_index_1.0_sample.json"),
}

REMOTES = {
    "zendodo": download_utils.RemoteFileMetadata(
        filename="rubato.zip",
        url="https://zenodo.org/records/23472735/files/rubato.zip?download=1",
        checksum="00000000000000000000000000000000",  # -- the md5 checksum TODO: -- add the checksum of the zip file hosted at Zenodo
    ),
}

# -- Include any information that should be printed when downloading
# -- remove this variable if you don't need to print anything during download
DOWNLOAD_INFO = """
Include any information you want to be printed when dataset.download() is called.
These can be instructions for how to download the dataset (e.g. request access on zenodo),
caveats about the download, etc
"""

LICENSE_INFO = "Creative Commons Attribution 3.0 Unported License (CC BY 3.0) https://creativecommons.org/licenses/by/3.0/"


class Track(core.Track):
    """RUBATO track class
    # -- YOU CAN AUTOMATICALLY GENERATE THIS DOCSTRING BY CALLING THE SCRIPT:
    # -- `scripts/print_track_docstring.py my_dataset`
    # -- note that you'll first need to have a test track (see "Adding tests to your dataset" below)

    Args:
        track_id (str): track id of the track

    Attributes:
        audio_path (str): path to audio file
        annotation_path (str): path to annotation file
        # -- Add any of the dataset specific attributes here

    Cached Properties:
        annotation (EventData): a description of this annotation

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):

        # -- this sets the following attributes:
        # -- * track_id
        # -- * _dataset_name
        # -- * _data_home
        # -- * _track_paths
        # -- * _track_metadata
        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        # -- add any dataset specific attributes here
        self.audio_path = self.get_path("audio")
        self.annotation_path = self.get_path("annotation")

        # -- if the dataset has an *official* e.g. train/test split, use this
        # -- reserved attribute (can be a property if needed)
        self.split = ...

    # -- If the dataset has metadata that needs to be accessed by Tracks,
    # -- such as a table mapping track ids to composers for the full dataset,
    # -- add them as properties like instead of in the __init__.
    @property
    def composer(self) -> Optional[str]:
        return self._track_metadata.get("composer")

    # -- `annotation` will behave like an attribute, but it will only be loaded
    # -- and saved when someone accesses it. Useful when loading slightly
    # -- bigger files or for bigger datasets. By default, we make any time
    # -- series data loaded from a file a cached property
    @core.cached_property
    def annotation(self) -> Optional[annotations.EventData]:
        return load_annotation(self.annotation_path)

    # -- `audio` will behave like an attribute, but it will only be loaded
    # -- when someone accesses it and it won't be stored. By default, we make
    # -- any memory heavy information (like audio) properties
    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)


# -- if the dataset contains multitracks, you can define a MultiTrack similar to a Track
# -- you can delete the block of code below if the dataset has no multitracks
class MultiTrack(core.MultiTrack):
    """RUBATO multitrack class

    Args:
        mtrack_id (str): multitrack id
        data_home (str): Local path where the dataset is stored.
            If `None`, looks for the data in the default directory, `~/mir_datasets/rubato`

    Attributes:
        mtrack_id (str): track id
        tracks (dict): {track_id: Track}
        track_audio_property (str): the name of the attribute of Track which
            returns the audio to be mixed
        # -- Add any of the dataset specific attributes here

    Cached Properties:
        annotation (EventData): a description of this annotation

    """

    def __init__(
        self, mtrack_id, data_home, dataset_name, index, track_class, metadata
    ):
        # -- this sets the following attributes:
        # -- * mtrack_id
        # -- * _dataset_name
        # -- * _data_home
        # -- * _multitrack_paths
        # -- * _metadata
        # -- * _track_class
        # -- * _index
        # -- * track_ids
        super().__init__(
            mtrack_id=mtrack_id,
            data_home=data_home,
            dataset_name=dataset_name,
            index=index,
            track_class=track_class,
            metadata=metadata,
        )

        # -- optionally add any multitrack specific attributes here
        self.mix_path = ...  # this can be called whatever makes sense for the datasets
        self.annotation_path = ...

        # -- if the dataset has an *official* e.g. train/test split, use this
        # -- reserved attribute (can be a property if needed)
        self.split = ...

    # If you want to support multitrack mixing in this dataset, set this property
    @property
    def track_audio_property(self):
        return "audio"  # the attribute of Track, e.g. Track.audio, which returns the audio to mix

    # -- multitracks can optionally have mix-level cached properties and properties
    @core.cached_property
    def annotation(self) -> Optional[annotations.EventData]:
        """output type: description of output"""
        return load_annotation(self.annotation_path)

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)


# -- this decorator allows this function to take a string or an open bytes file as input
# -- and in either case converts it to an open file handle.
# -- It also checks if the file exists
# -- and, if None is passed, None will be returned
@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a RUBATO audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file

    """
    # -- for example, the code below. This should be dataset specific!
    # -- By default we load to mono
    # -- change this if it doesn't make sense for your dataset.
    return librosa.load(audio_path, sr=None, mono=True)


# -- Write any necessary loader functions for loading the dataset's data


# -- this decorator allows this function to take a string or an open file as input
# -- and in either case converts it to an open file handle.
# -- It also checks if the file exists
# -- and, if None is passed, None will be returned
@io.coerce_to_string_io
def load_annotation(fhandle: TextIO) -> Optional[annotations.EventData]:

    # -- because of the decorator, the file is already open
    reader = csv.reader(fhandle, delimiter=" ")
    intervals = []
    annotation = []
    for line in reader:
        intervals.append([float(line[0]), float(line[1])])
        annotation.append(line[2])

    # there are several annotation types in annotations.py
    # They should be initialized with data, followed by their units
    # see annotations.py for a complete list of types and units.
    annotation_data = annotations.EventData(
        np.array(intervals), "s", np.array(annotation), "open"
    )
    return annotation_data


# -- use this decorator so the docs are complete
@core.docstring_inherit(core.Dataset)
class Dataset(core.Dataset):
    """The RUBATO dataset"""

    def __init__(self, data_home=None, version="default"):
        super().__init__(
            data_home,
            version,
            name="rubato",
            track_class=Track,
            bibtex=BIBTEX,
            indexes=INDEXES,
            remotes=REMOTES,
            download_info=DOWNLOAD_INFO,
            license_info=LICENSE_INFO,
        )

    # -- if your dataset has a top-level metadata file, write a loader for it here
    # -- you do not have to include this function if there is no metadata
    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "metadata_versions.csv")
        metadata_works_path = os.path.join(self.data_home, "metadata_works.csv")

       
        with open(metadata_path, "r") as fhandle:
            metadata_versions = csv.reader(fhandle, delimiter=";")

        with open(metadata_works_path, "r") as fhandle:
            metadata_works = csv.reader(fhandle, delimiter=";")  #TO
        #TODO check whether we can return both metadata tables or only versions possible
        return metadata_versions, metadata_works
