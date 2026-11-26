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

from mirdata import download_utils, core, annotations, io

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
"""  # TODO: -- add the download information for the dataset

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
        metadata_path (str): path to metadata file

    Cached Properties:
        annotation (EventData): a description of this annotation

    """

    def __init__(self, track_id, data_home, dataset_name, index, metadata):

        super().__init__(
            track_id,
            data_home,
            dataset_name=dataset_name,
            index=index,
            metadata=metadata,
        )

        self.audio_path = self.get_path("audio")

    @property
    def audio(self) -> Optional[Tuple[np.ndarray, float]]:
        """The track's audio

        Returns:
            * np.ndarray - audio signal
            * float - sample rate

        """
        return load_audio(self.audio_path)

    @property
    def version_type(self) -> Optional[str]:
        return self._track_metadata.get("version_type")

    @property
    def work_name_semantic(self) -> Optional[str]:
        return self._track_metadata.get("work_name_semantic")

    @property
    def structure(self) -> Optional[str]:
        return self._track_metadata.get("structure")

    @property
    def instrumentation(self) -> Optional[str]:
        return self._track_metadata.get("instrumentation")

    @property
    def lyrics_original(self) -> Optional[str]:
        return self._track_metadata.get("lyrics_original")

    @property
    def voice(self) -> Optional[str]:
        return self._track_metadata.get("voice")

    @property
    def tuning_A4(self):
        return self._track_metadata.get("tuning_A4")

    @property
    def global_key(self):
        return self._track_metadata.get("global_key")

    @property
    def transposition(self):
        return self._track_metadata.get("transposition")

    @property
    def vocal_shift(self):
        return self._track_metadata.get("vocal_shift")

    @property
    def noise_level(self):
        return self._track_metadata.get("noise_level")

    @property
    def length(self):
        return self._track_metadata.get("length")

    @property
    def license(self):
        return self._track_metadata.get("license")

    @property
    def source(self):
        return self._track_metadata.get("source")

    @property
    def work_title(self):
        return self._track_metadata.get("work_title")

    @property
    def global_key_work(self):
        return self._track_metadata.get("global_key_work")

    @property
    def instrumentation_original(self):
        return self._track_metadata.get("instrumentation_original")


@io.coerce_to_bytes_io
def load_audio(fhandle: BinaryIO) -> Tuple[np.ndarray, float]:
    """Load a RUBATO audio file.

    Args:
        fhandle (str or file-like): path or file-like object pointing to an audio file

    Returns:
        * np.ndarray - the audio signal
        * float - The sample rate of the audio file

    """
    return librosa.load(fhandle, sr=None, mono=True)


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


@io.coerce_to_string_io
def load_localKey(fhandle: TextIO) -> Optional[annotations.KeyData]:

    reader = csv.reader(fhandle, delimiter=" ")
    reader.next()  # skip header
    intervals = []
    localKey_label = []
    for line in reader:
        intervals.append([float(line[0]), float(line[1])])
        localKey_label.append(line[2])

    annotation_data = annotations.KeyData(
        np.array(intervals), "s", np.array(localKey_label), "harte"
    )
    return annotation_data


@io.coerce_to_string_io
def load_beats(fhandle: TextIO) -> Optional[annotations.BeatData]:

    reader = csv.reader(fhandle, delimiter=" ")
    reader.next()  # skip header
    time = []
    beats = []
    for line in reader:
        time.append(float(line[0]))
        beats.append(line[1])
    annotation_data = annotations.BeatData(
        np.array(time), "s", np.array(beats), "global-fraction"
    )
    return annotation_data

@io.coerce_to_string_io
def load_chords(fhandle: TextIO) -> Optional[annotations.ChordData]: # TODO shorthand or extended or both? rn shorthand

    reader = csv.reader(fhandle, delimiter=" ")
    reader.next()  # skip header
    intervals = []
    chord_labels = []
    for line in reader:
        intervals.append([float(line[0]), float(line[1])])
        chord_labels.append(line[2])

    annotation_data = annotations.ChordData(
        np.array(intervals), "s", np.array(chord_labels), "open"
    )
    return annotation_data


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

    @core.cached_property
    def _metadata(self):
        metadata_path = os.path.join(self.data_home, "metadata_versions.csv")
        metadata_works_path = os.path.join(self.data_home, "metadata_works.csv")

        with open(metadata_works_path, "r") as fhandle:
            metadata_works = csv.reader(fhandle, delimiter=";")  # TO

        metadata_versions_works = {}
        try:
            with open(metadata_path, "r") as fhandle:
                metadata_versions = csv.reader(fhandle, delimiter=";")
                next(metadata_versions)  # skip header
                with open(metadata_works_path, "r") as fhandle_works:
                    metadata_works = csv.reader(fhandle_works, delimiter=";")
                    next(metadata_works)

                    for row in metadata_versions:
                        key = row[0]  # filename
                        work_id = row[0].split("_")[1]
                        # get the corresponding work metadata for this version
                        work_metadata = next(
                            (w for w in metadata_works if w[0] == work_id), None
                        )
                        if work_metadata:
                            work_title = work_metadata[1]
                            composer = work_metadata[3]
                            global_key_work = work_metadata[4]
                            instrumentation_original = work_metadata[5]
                        metadata_versions_works[key] = {
                            "version_type": row[1],
                            "work_name_semantic": row[3],
                            "structure": row[4],
                            "instrumentation": row[5],
                            "lyrics_original": row[6],
                            "voice": row[7],
                            "tuning_A4": row[8],
                            "global_key": row[9],
                            "transposition": row[10],
                            "vocal_shift": row[11],
                            "noise_level": row[12],
                            "length": row[13],
                            "license": row[14],
                            "source": row[15],
                            "work_title": work_title,
                            "composer": composer,
                            "global_key_work": global_key_work,
                            "instrumentation_original": instrumentation_original,
                        }
        except FileNotFoundError:
            raise FileNotFoundError("Metadata not found. Did you run .download()?")

        return metadata_versions_works
