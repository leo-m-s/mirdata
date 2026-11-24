import argparse
import glob
import json
import os
from mirdata.validate import md5

DATASET_INDEX_PATH = "../mirdata/datasets/indexes/rubato_index_1.0.json"

def get_checksum_or_none(file_path):
    if os.path.exists(file_path):
        return md5(file_path)
    else:
        return None

def make_rubato_index(rubato_data_path, version):
    assert version == "1.0"
    annotation_dir = os.path.join(rubato_data_path, "02_annotations")
    ann_audio_beat_dir = os.path.join(annotation_dir, "ann_audio_beat")
    ann_audio_chord_dir = os.path.join(annotation_dir, "ann_audio_chord")
    ann_audio_localkey_dir = os.path.join(annotation_dir, "ann_audio_localkey")
    ann_audio_measure_dir = os.path.join(annotation_dir, "ann_audio_measure")
    ann_audio_noteEvents_dir = os.path.join(annotation_dir, "ann_audio_noteEvents")
    ann_audio_pitchClass_dir = os.path.join(annotation_dir, "ann_audio_pitchClass")
    ann_audio_startEnd_dir = os.path.join(annotation_dir, "ann_audio_startEnd")
    ann_score_chord_dir = os.path.join(annotation_dir, "ann_score_chord")
    ann_score_localkey_dir = os.path.join(annotation_dir, "ann_score_localkey")
    ann_score_structure_dir = os.path.join(annotation_dir, "ann_score_structure")
    ann_audio_globalkey_dir = annotation_dir
    ann_score_globalkey_dir = annotation_dir
    
    data_dir = os.path.join(rubato_data_path, "01_RawData")
    audio_dir = os.path.join(data_dir, "wav_22050_mono")
    score_midi_dir = os.path.join(data_dir, "score_midi")
    score_musicxml_dir = os.path.join(data_dir, "score_musicxml")
    score_pdf_dir = os.path.join(data_dir, "score_pdf")
    score_musescore_dir = os.path.join(data_dir, "score_musescore")
    
    metadata_path = os.path.join(rubato_data_path, "metadata_versions.csv")
    metadata_works_path = os.path.join(rubato_data_path, "metadata_works.csv")

    track_ids = sorted([os.path.basename(f).split(".")[0] for f in glob.glob(os.path.join(audio_dir, "*.wav"))]) # retrieve track ids from audio files
    score_ids = sorted([os.path.basename(f).split(".")[0] for f in glob.glob(os.path.join(score_midi_dir, "*.mid"))]) # retrieve score ids from midi files

    # top-key level tracks
    index_tracks = {}
    for track_id in track_ids:
        audio_checksum = md5(
            os.path.join(audio_dir, "{}.wav".format(track_id))
        )

        audio_beat_checksum = get_checksum_or_none(
            os.path.join(ann_audio_beat_dir, "{}.csv".format(track_id))
        )

        audio_chord_checksum = get_checksum_or_none(
            os.path.join(ann_audio_chord_dir, "{}.csv".format(track_id))
        )
        audio_localkey_checksum = get_checksum_or_none(
            os.path.join(ann_audio_localkey_dir, "{}.csv".format(track_id))
        )
        audio_measure_checksum = get_checksum_or_none(
            os.path.join(ann_audio_measure_dir, "{}.csv".format(track_id))
        )
        audio_noteEvents_checksum = get_checksum_or_none(
            os.path.join(ann_audio_noteEvents_dir, "{}.csv".format(track_id))
        )
        audio_pitchClass_checksum = get_checksum_or_none(
            os.path.join(ann_audio_pitchClass_dir, "{}.csv".format(track_id))
        )
        audio_startEnd_checksum = get_checksum_or_none(
            os.path.join(ann_audio_startEnd_dir, "{}.csv".format(track_id))
        )

        index_tracks[track_id] = {
            "audio": ("Wavfile/{}.wav".format(track_id), audio_checksum),
            "beat": ("annotation/{}.csv".format(track_id), audio_beat_checksum),
            "chord": ("annotation/{}.csv".format(track_id), audio_chord_checksum),
            "localkey": ("annotation/{}.csv".format(track_id), audio_localkey_checksum),
            "measure": ("annotation/{}.csv".format(track_id), audio_measure_checksum),
            "noteEvents": ("annotation/{}.csv".format(track_id), audio_noteEvents_checksum),
            "pitchClass": ("annotation/{}.csv".format(track_id), audio_pitchClass_checksum),
            "startEnd": ("annotation/{}.csv".format(track_id), audio_startEnd_checksum),
        }
       
    
    index_scores = {}
    for score_id in score_ids:
        midi_checksum = get_checksum_or_none(
            os.path.join(score_midi_dir, "{}.mid".format(score_id))
        )
        musicxml_checksum = get_checksum_or_none(
            os.path.join(score_musicxml_dir, "{}.xml".format(score_id))
        )
        pdf_checksum = get_checksum_or_none(
            os.path.join(score_pdf_dir, "{}.pdf".format(score_id))
        )
        musescore_checksum = get_checksum_or_none(
            os.path.join(score_musescore_dir, "{}.mscz".format(score_id))
        )
        score_chord_checksum = get_checksum_or_none(
            os.path.join(ann_score_chord_dir, "{}.csv".format(score_id))
        )
        score_localkey_checksum = get_checksum_or_none(
            os.path.join(ann_score_localkey_dir, "{}.csv".format(score_id))
        )
        score_structure_checksum = get_checksum_or_none(
            os.path.join(ann_score_structure_dir, "{}.csv".format(score_id))
        )

        index_scores[score_id] = {
            "midi": ("score_midi/{}.mid".format(score_id), midi_checksum),
            "musicxml": ("score_musicxml/{}.musicxml".format(score_id), musicxml_checksum),
            "pdf": ("score_pdf/{}.pdf".format(score_id), pdf_checksum),
            "musescore": ("score_musescore/{}.mscz".format(score_id), musescore_checksum),
            "chord": ("score_chord/{}.csv".format(score_id), score_chord_checksum),
            "localkey": ("score_localkey/{}.csv".format(score_id), score_localkey_checksum),
            "structure": ("score_structure/{}.csv".format(score_id), score_structure_checksum),
        }

    rubato_index = {
            "version": version,
            "tracks": index_tracks,
            "scores": index_scores,
            "metadata": {
                "rubato_metadata_versions": ("metadata_versions.csv", md5(metadata_path)),
                "rubato_metadata_works": ("metadata_works.csv", md5(metadata_works_path)),
            },
        }

    with open(DATASET_INDEX_PATH, "w") as fhandle:
        json.dump(rubato_index, fhandle, indent=2)


def main(args):
    make_rubato_index(args.rubato_data_path, args.version)


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Make dataset index file.")
    PARSER.add_argument(
            "rubato_data_path", type=str, help="Path to Rubato data folder."
        )
    PARSER.add_argument("version", type=str, help="index version")

    main(PARSER.parse_args())