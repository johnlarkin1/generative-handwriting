import os
import xml.etree.ElementTree as ET
from typing import Optional, Set, Tuple

import drawing
import numpy as np
from alphabet import encode_ascii
from constants import (
    MAX_CHAR_LEN,
    MAX_STROKE_LEN,
    STROKE_SCALE_FACTOR,
    STROKE_SPACE_THRESHOLD,
)
from tqdm import tqdm


class HandwritingDataLoader:
    """
    A data loader for the IAM On-Line Handwriting data.

    This data is largely for parsing XML files and extracting the
    stroke data from them. The data is then used to train a
    handwriting model.
    """

    scale_factor: int
    space_threshold: int
    data_dir: str
    train_set: set[str]
    valid1_set: set[str]
    valid2_set: set[str]
    test_set: set[str]

    curr_dir: str

    def __init__(
        self,
        scale_factor: int = STROKE_SCALE_FACTOR,
        space_threshold: int = STROKE_SPACE_THRESHOLD,
        data_dir: str = "data/lineStrokes",
        ascii_dir: str = "data/ascii",
    ) -> None:
        """
        Args:
            scale_factor: Factor to scale the data by
            space_threshold: The maximum allowed space between two points
        """

        self.scale_factor = scale_factor
        self.space_threshold = space_threshold
        self.data_dir = data_dir
        self.ascii_dir = ascii_dir

        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        for name in ["train", "valid1", "valid2", "test"]:
            file_path = os.path.join(self.curr_dir, "data", f"{name}_set.txt")
            with open(file_path, "r") as file:
                setattr(
                    self,
                    f"{name}_set",
                    {line.strip() for line in file.read().splitlines()},
                )

    def load_and_save_data(self) -> None:
        strokes, stroke_lens, trans, trans_lens = self._load_data()
        self._save_data(strokes, stroke_lens, trans, trans_lens)

        # Define dataset names and corresponding filename sets
        dataset_names = ["train", "valid1", "valid2", "test"]
        filename_sets = [
            self.train_set,
            self.valid1_set,
            self.valid2_set,
            self.test_set,
        ]

        for dataset_name, filename_set in zip(dataset_names, filename_sets):
            strokes, stroke_lens, trans, trans_lens = self._load_data(filename_set)
            self._save_data(strokes, stroke_lens, trans, trans_lens, dataset_name)

    def load_individual_stroke_data(self, filename: str) -> Tuple[np.ndarray, int]:
        """
        Load and process a single handwriting sample identified by the filename.

        Args:
            filename: The name of the file to load, expected to be an XML file.

        Returns:
            A tuple containing:
            - A NumPy array of the stroke data (scaled and with applied thresholds).
            - The length of the stroke data array.
        """
        data_file_path = os.path.join(self.curr_dir, self.data_dir, filename)
        stroke_data, stroke_data_length = self._parse_xml(data_file_path)
        # 1 sample, MAX_STROKE_LEN timesteps, 3 features (dx, dy, eos)
        strokes = np.zeros((1, MAX_STROKE_LEN, 3), dtype=np.float32)
        length = min(stroke_data_length, MAX_STROKE_LEN)
        strokes[0, :length, :] = stroke_data[:length]
        return strokes[0], stroke_data_length

    def load_individual_stroke_and_c_data(self, filename: str) -> Tuple[np.ndarray, int, np.ndarray, int]:
        data_file_path = os.path.join(self.curr_dir, self.data_dir, filename)
        return self._load_individual_file_data(data_file_path)

    def _load_data(
        self, filename_set: Optional[Set[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stroke_data_list = []
        stroke_lengths_list = []
        transcription_data_list = []
        transcription_lengths_list = []
        file_paths = []

        # Collect all file paths first to get a total count for the progress bar
        for dirpath, dirnames, files in os.walk(os.path.join(self.curr_dir, self.data_dir)):
            for file_name in filter(lambda f: f.endswith(".xml"), files):
                if "z01-000z" in file_name:
                    continue
                if filename_set:
                    file_name_without_extension = os.path.splitext(file_name)[0]
                    parts = file_name_without_extension.split("-")
                    prefix = "-".join(parts[:2])  # Re-join the first two parts (before the second dash)
                    if prefix in filename_set:
                        file_path = os.path.join(dirpath, file_name)
                        file_paths.append(file_path)
                else:
                    file_path = os.path.join(dirpath, file_name)
                    file_paths.append(file_path)

        # Wrap the file_paths iterable with tqdm to show a progress bar
        for file_path in tqdm(file_paths, desc="Loading Data"):
            (
                strokes,
                stroke_lengths,
                transcriptions,
                transcription_lengths,
            ) = self._load_individual_file_data(file_path)
            stroke_data_list.append(strokes)
            stroke_lengths_list.append(stroke_lengths)
            transcription_data_list.append(transcriptions)
            transcription_lengths_list.append(transcription_lengths)

        strokes = np.zeros((len(stroke_data_list), MAX_STROKE_LEN, 3), dtype=np.float32)
        stroke_lengths = np.array(stroke_lengths_list, dtype=int)
        trans = np.zeros((len(transcription_data_list), MAX_CHAR_LEN), dtype=int)
        trans_len = np.array(transcription_lengths_list, dtype=int)

        # Populate the strokes array
        for i, stroke_data in enumerate(stroke_data_list):
            length = min(len(stroke_data), MAX_STROKE_LEN)
            strokes[i, :length, :] = stroke_data[:length]

        # Populate the transcriptions array
        for i, transcription_data in enumerate(transcription_data_list):
            length = min(len(transcription_data), MAX_CHAR_LEN)
            trans[i, :length] = transcription_data[:length]

        return strokes, stroke_lengths, trans, trans_len

    def _load_individual_file_data(self, file_path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stroke_data, stroke_data_len = self._parse_xml(file_path)
        trans_data, trans_data_len = self._parse_transcription(file_path)
        return stroke_data, stroke_data_len, trans_data, trans_data_len

    def _save_data(
        self,
        strokes: np.ndarray,
        stroke_lengths: np.ndarray,
        trans: np.ndarray,
        trans_lens: np.ndarray,
        dataset_name: Optional[str] = None,
    ) -> None:
        """
        Save the data to a file
        """
        processed_path = os.path.join(self.curr_dir, "data", "processed")
        if not os.path.exists(processed_path):  # Ensure the directory exists
            os.makedirs(processed_path)
        if dataset_name:
            np.save(os.path.join(processed_path, f"{dataset_name}_stroke_data.npy"), strokes)
            np.save(
                os.path.join(processed_path, f"{dataset_name}_stroke_lengths.npy"),
                stroke_lengths,
            )
            np.save(os.path.join(processed_path, f"{dataset_name}_trans.npy"), trans)
            np.save(
                os.path.join(processed_path, f"{dataset_name}_trans_lengths.npy"),
                trans_lens,
            )
        else:
            np.save(os.path.join(processed_path, "stroke_data.npy"), strokes)
            np.save(os.path.join(processed_path, "stroke_lengths.npy"), stroke_lengths)
            np.save(os.path.join(processed_path, "trans.npy"), trans)
            np.save(os.path.join(processed_path, "trans_lengths.npy"), trans_lens)

    def _parse_xml(self, filename: str) -> Tuple[np.ndarray, int]:
        """
        Parse the XML file, extract stroke data, calculate offsets, apply scaling,
        and enforce a maximum distance threshold between points.

        Args:
            filename: The name of the file to parse

        Returns:
            StrokeData containing offsets and end-of-stroke markers, or
            raises if the file cannot be parsed.


        I'm more or less ripping this from here:
        https://github.com/sjvasquez/handwriting-synthesis/blob/master/prepare_data.py
        """
        try:
            tree = ET.parse(filename).getroot()
            strokes = [i for i in tree if i.tag == "StrokeSet"][0]

            coords = []
            for stroke in strokes:
                for i, point in enumerate(stroke):
                    coords.append(
                        [
                            int(point.attrib["x"]),
                            -1 * int(point.attrib["y"]),
                            int(i == len(stroke) - 1),
                        ]
                    )
            coords = np.array(coords)

            # coords = drawing.align(coords)
            coords = drawing.denoise(coords)
            offsets = drawing.coords_to_offsets(coords)
            offsets = offsets[:MAX_STROKE_LEN]
            offsets = drawing.normalize(offsets)
            return offsets, len(offsets)
        except ET.ParseError as e:
            print(f"Error parsing XML file: {filename}")
            raise e

    def _parse_transcription(self, file_path):
        ascii_filename_raw = os.path.splitext(os.path.basename(file_path))[0]
        ascii_filename = ascii_filename_raw[: ascii_filename_raw.rindex("-")]
        transcription_idx = int(ascii_filename_raw[ascii_filename_raw.rindex("-") + 1 :]) - 1
        ascii_filename = ascii_filename + ".txt"
        ascii_filedir = os.path.dirname(file_path).replace("lineStrokes", "ascii")
        ascii_filepath = os.path.join(ascii_filedir, ascii_filename)
        transcriptions = self.get_transcription_sequence(ascii_filepath)
        return transcriptions[transcription_idx], len(transcriptions[transcription_idx])

    def get_transcription_sequence(self, filename: str) -> list[str]:
        sequences = open(filename, "r").read()
        sequences = sequences.replace(r"%%%%%%%%%%%", "\n")
        sequences = [i.strip() for i in sequences.split("\n")]
        lines = sequences[sequences.index("CSR:") + 2 :]
        lines = [line.strip() for line in lines if line.strip()]
        lines = [encode_ascii(line)[:MAX_CHAR_LEN] for line in lines]
        return lines

    def check_and_load_data(self, dataset_name="all"):
        """
        Checks if processed data exists, loads it if so, otherwise processes and saves data.

        Args:
            dataset_name: The name of the dataset to load. Options are 'train', 'valid1', 'valid2', 'test', or 'all'.
                          Defaults to 'all', which processes and loads all datasets.
        """
        processed_path = os.path.join(self.curr_dir, "data", "processed")
        datasets = ["train", "valid1", "valid2", "test"] if dataset_name == "all" else [dataset_name]

        for dataset in datasets:
            stroke_data_path = os.path.join(processed_path, f"{dataset}_stroke_data.npy")
            stroke_lengths_path = os.path.join(processed_path, f"{dataset}_stroke_lengths.npy")
            transcriptions_path = os.path.join(processed_path, f"{dataset}_trans.npy")
            transcription_lengths_path = os.path.join(
                processed_path,
                f"{dataset}_trans_lengths.npy",
            )

            if os.path.exists(stroke_data_path) and os.path.exists(stroke_lengths_path):
                print(f"{dataset} data found. Loading...")
                strokes = np.load(stroke_data_path)
                stroke_lengths = np.load(stroke_lengths_path)
                transcriptions = np.load(transcriptions_path)
                transcription_lengths = np.load(transcription_lengths_path)
                setattr(self, f"{dataset}_strokes", strokes)
                setattr(self, f"{dataset}_trans", transcriptions)
                setattr(self, f"{dataset}_stroke_lengths", stroke_lengths)
                setattr(self, f"{dataset}_trans_lengths", transcription_lengths)
            else:
                print(f"{dataset} data not found. Processing and saving...")
                self.load_and_save_data()

    def get_data(self, dataset_name):
        """
        Returns the requested dataset.

        Args:
            dataset_name: The name of the dataset to return ('train', 'valid1', 'valid2', or 'test').

        Returns:
            A tuple containing the strokes and stroke lengths of the requested dataset.
        """
        if not hasattr(self, f"{dataset_name}_strokes") or not hasattr(self, f"{dataset_name}_stroke_lengths"):
            self.check_and_load_data(dataset_name=dataset_name)

        strokes = getattr(self, f"{dataset_name}_strokes", None)
        stroke_lengths = getattr(self, f"{dataset_name}_stroke_lengths", None)
        trans = getattr(self, f"{dataset_name}_trans", None)
        trans_lengths = getattr(self, f"{dataset_name}_trans_lengths", None)
        return strokes, stroke_lengths, trans, trans_lengths

    def combine_datasets(self, *datasets):
        """
        Combines multiple datasets into one.

        Args:
            *datasets: Variable number of tuples, where each tuple contains (strokes, stroke_lengths)
                       corresponding to a dataset.

        Returns:
            Combined (strokes, stroke_lengths) of all given datasets.
        """
        combined = np.concatenate([data[0] for data in datasets], axis=0)
        combined_lengths = np.concatenate([data[1] for data in datasets], axis=0)
        return combined, combined_lengths

    def prepare_data(self):
        """
        Prepares the data according to the specified strategy: Combines the training set, test set, and the larger
        validation set for training, and uses the smaller validation set for early stopping.
        """
        # Load datasets
        (
            train_strokes,
            train_stroke_lengths,
            train_trans,
            train_trans_lengths,
        ) = self.get_data("train")
        (
            test_strokes,
            test_stroke_lengths,
            test_trans,
            test_trans_lengths,
        ) = self.get_data("test")
        (
            valid2_strokes,
            valid2_stroke_lengths,
            valid2_trans,
            valid2_trans_lengths,
        ) = self.get_data("valid2")
        (
            valid1_strokes,
            valid1_stroke_lengths,
            valid1_trans,
            valid1_trans_lengths,
        ) = self.get_data("valid1")

        # Combine the training set, test set, and valid2 set for training
        (
            self.combined_train_strokes,
            self.combined_train_stroke_lengths,
        ) = self.combine_datasets(
            (train_strokes, train_stroke_lengths),
            (test_strokes, test_stroke_lengths),
            (valid2_strokes, valid2_stroke_lengths),
        )

        (
            self.combined_train_transcriptions,
            self.combined_train_transcription_lengths,
        ) = self.combine_datasets(
            (train_trans, train_trans_lengths),
            (test_trans, test_trans_lengths),
            (valid2_trans, valid2_trans_lengths),
        )

        # Use valid1 set for early stopping (as the validation set)
        self.validation_strokes, self.validation_stroke_lengths = (
            valid1_strokes,
            valid1_stroke_lengths,
        )


if __name__ == "__main__":
    loader = HandwritingDataLoader()
    loader.load_and_save_data()
    loader.prepare_data()
    # loader._parse_transcription(
    #     os.path.join(
    #         os.path.dirname(os.path.realpath(__file__)),
    #         "data/ascii/a01/a01-000/a01-000u-01.txt",
    #     )
    # )
