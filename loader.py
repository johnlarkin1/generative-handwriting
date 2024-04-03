from typing import Optional, Set, Tuple
import drawing
import numpy as np
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm


from constants import MAX_STROKE_LEN, STROKE_SPACE_THRESHOLD, STROKE_SCALE_FACTOR


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
    ) -> None:
        """
        Args:
            scale_factor: Factor to scale the data by
            space_threshold: The maximum allowed space between two points
        """

        self.scale_factor = scale_factor
        self.space_threshold = space_threshold
        self.data_dir = data_dir

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
        strokes, stroke_lengths = self._load_data()
        self._save_data(strokes, stroke_lengths)

        # Define dataset names and corresponding filename sets
        dataset_names = ["train", "valid1", "valid2", "test"]
        filename_sets = [self.train_set, self.valid1_set, self.valid2_set, self.test_set]

        for dataset_name, filename_set in zip(dataset_names, filename_sets):
            strokes, stroke_lengths = self._load_data(filename_set)
            self._save_data(strokes, stroke_lengths, dataset_name)

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
        file_path = os.path.join(self.curr_dir, self.data_dir, filename)
        stroke_data, stroke_data_length = self._parse_xml(file_path)
        # 1 sample, MAX_STROKE_LEN timesteps, 3 features (dx, dy, eos)
        strokes = np.zeros((1, MAX_STROKE_LEN, 3), dtype=np.float32)
        length = min(stroke_data_length, MAX_STROKE_LEN)
        strokes[0, :length, :] = stroke_data[:length]
        return strokes[0], stroke_data_length

    def _load_data(self, filename_set: Optional[Set[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        stroke_data_list = []
        stroke_lengths_list = []
        file_paths = []

        # Collect all file paths first to get a total count for the progress bar
        for dirpath, dirnames, files in os.walk(os.path.join(self.curr_dir, self.data_dir)):
            for file_name in filter(lambda f: f.endswith(".xml"), files):
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
            os.path.splitext(os.path.basename(file_path))[0]
            stroke_data, stroke_data_length = self._parse_xml(file_path)
            stroke_data_list.append(stroke_data)
            stroke_lengths_list.append(stroke_data_length)

        strokes = np.zeros((len(stroke_data_list), MAX_STROKE_LEN, 3), dtype=np.float32)
        stroke_lengths = np.array(stroke_lengths_list, dtype=int)

        # Populate the strokes array
        for i, stroke_data in enumerate(stroke_data_list):
            length = min(len(stroke_data), MAX_STROKE_LEN)
            strokes[i, :length, :] = stroke_data[:length]

        return strokes, stroke_lengths

    def _save_data(
        self,
        strokes: np.ndarray,
        stroke_lengths: np.ndarray,
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
            np.save(os.path.join(processed_path, f"{dataset_name}_stroke_lengths.npy"), stroke_lengths)
        else:
            np.save(os.path.join(processed_path, "stroke_data.npy"), strokes)
            np.save(os.path.join(processed_path, "stroke_lengths.npy"), stroke_lengths)

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
                    coords.append([int(point.attrib["x"]), -1 * int(point.attrib["y"]), int(i == len(stroke) - 1)])
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


if __name__ == "__main__":
    data = HandwritingDataLoader()
