import os
import pandas as pd
from bs4 import BeautifulSoup
import pathlib
from tqdm.auto import tqdm
from multiprocessing import Pool

mapping = {
    "&amp;": "&",  # ampersand (should be tironian note)
    "&AE;": "Æ",  # uppercase ash
    "&D;": "Ð",  # uppercase eth
    "&T;": "Þ",  # uppercase thorn
    "&ae;": "æ",  # lowercase ash
    "&d;": "ð",  # lowercase eth
    "&oe;": "œ",  # lowercase o with stroke
    "&t;": "þ",  # lowercase thorn
    "&egrave;": "è",  # e grace
    "&eacute;": "é",  # e acute
    "&E;": "Ę",  # uppercase e tail
    "&e;": "ę",  # lowercase e tail
    "&auml;": "ä",  # a umlaut
    "&ouml;": "ö",  # o umlaut
    "&uuml;": "ü",  # u umlaut
    "&lbar;": "ł",  # barred l
    "&tbar;": "ꝥ",  # barred thorn
    "&bbar;": "ᴃ",  # barred b
    "&amacron;": "ā",  # a macron
    "&cmacron;": "ċ",  # c macron
    "&emacron;": "ē",  # e macron
    "&imacron;": "ī",  # i macron
    "&nmacron;": "ņ",  # n macron
    "&omacron;": "ō",  # o macron
    "&pmacron;": "ṕ",  # p macron
    "&qmacron;": "ꝗ",  # q macron
    "&rmacron;": "ŗ",  # r macron
    "&Alpha;": "Α",  # uppercase greek alpha
    "&Omega;": "Ω",  # uppercase greek omega
    "&omega;": "ω",  # lowercase greek omega
    "&Lambda;": "Λ",  # uppercase greek lambda
    "&Omicron;": "Ο",  # uppercase greek omicron
    "&Tau;": "Τ",  # uppercase greek tau
    "&Rho;": "Ρ",  # uppercase greek rho
    "&Eta;": "Η",  # uppercase greek eta
    "&Nu;": "Ν",  # uppercase greek nu
}


class DOEC_raw_parser:
    """Parses the SGML files and returns the text content and some metadata."""

    def __init__(self, folder: str):
        """This parser might only work for the SGML files from the DOEC corpus.

        Parameters
        ----------
        folder : str
            The folder where the SGML files are located.
        """

        self.root_folder = pathlib.Path(folder)
        self.file_list = os.listdir(self.root_folder)

    def parse_all(self):
        """Parses all files in the folder and returns a formatted dataframe.

        Returns
        -------
        list
            A list of dictionaries with the parsed data.
        """

        # Content files (these start with "T")
        content_files = [f for f in self.file_list if f.startswith("T")]

        # Create a threadpool with num cpu cores
        pool = Pool(os.cpu_count())

        pbar = tqdm(total=len(content_files))

        def callback(result):
            pbar.update()

        # Parse all files in parallel
        results = []
        with Pool(processes=os.cpu_count()) as pool:
            async_results = [
                pool.apply_async(self._parse_file, args=(x,), callback=callback)
                for x in content_files
            ]
            results = [async_result.get() for async_result in async_results]
        pbar.close()

        # Concatenate all dataframes
        df = pd.concat(results, ignore_index=True)

        return df

    def _parse_file(self, file, progress_bar=None):
        """Parses a single file and returns a dictionary with the parsed data.

        Returns
        -------
        pd.DataFrame
            A dataframe with the parsed data. Columns: "type", "text", "file", "id"
        """

        bs = BeautifulSoup(open(self.root_folder.joinpath(file)), "lxml")

        # First get the metadata
        idno = bs.find("idno")
        if idno:
            idno = idno.text

        title = bs.find("sourcedesc").find("title")
        if not title:
            title = bs.find("title")
        if title:
            title = title.text

        # Get the text
        try:
            texts = bs.find_all("s")
        except AttributeError:
            # Throw error
            raise AttributeError("No text found in file: " + file)

        # Parse all content and attributes from the s tags
        # Normally this should be id and n
        df = pd.DataFrame(
            columns=[
                "idno",
                "title",
                "n",
                "file",
                "id",
                "text",
            ]
        )

        for s in texts:
            # Get the content
            text = s.text

            # Replace all texts entities with the mapping
            for key, value in mapping.items():
                text = text.replace(key, value)

            # Remove <corr></corr> tags
            # this might bite me in the ass at a later point
            text = text.replace("<corr>", "").replace("</corr>", "")

            # Get the attributes
            id = s.get("id", None)
            n = s.get("n", None)

            data = {
                "idno": idno,
                "title": title,
                "n": n,
                "file": file,
                "id": id,
                "text": text,
            }

            # Append to the dataframe
            df = pd.concat(
                [df, pd.DataFrame.from_dict([data])],
                ignore_index=True,
            )

        # Replace all texts with the mapping
        for key, value in mapping.items():
            df["text"] = df["text"].str.replace(key, value)

        return df


def preprocess_data(folder):

    parser = DOEC_raw_parser(folder)
    df = parser.parse_all()

    # A Poetry
    # B Prose
    # C Interlinear Glosses
    # Filter by type idno starts with A,B,C
    masks = [df["idno"].str.startswith("A"),df["idno"].str.startswith("B"),df["idno"].str.startswith("C"),]

    df = pd.concat([df[mask] for mask in masks], ignore_index=True)

    # Assign id to all unique titles
    df["title_id"] = df["title"].astype("category").cat.codes

    return df
