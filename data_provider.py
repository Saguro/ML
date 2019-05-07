from io import BytesIO, StringIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from PIL import Image


class DataProvider:
    def __init__(self, file_name: str, data_zip_dir: str = './data'):
        self._annotations_zip = 'data/annotations/'
        self._videos_zip = 'data/frames/'
        self._path = Path(data_zip_dir)
        self._zip_file_path = self._path / file_name
        self._list_of_videos = None

    @property
    def list_of_videos(self):
        if self._list_of_videos is None:
            self._list_of_videos = [i[len(self._annotations_zip):-4] for i in ZipFile(self._zip_file_path).namelist()
                                    if i.startswith(self._annotations_zip) and i.endswith('csv')]
        return self._list_of_videos

    def get_video(self, video_name: str) -> pd.DataFrame:
        assert video_name in self.list_of_videos,\
            f"Video named '{video_name}' wasn't found in zip file '{self._zip_file_path.name}'"
        zip = ZipFile(self._zip_file_path)
        video_data = pd.read_csv(
            StringIO(str(zip.open(f"{self._annotations_zip}{video_name}.csv").read(), 'utf-8')),
            header=None,
            names='file_name right_hand_x right_hand_y left_hand_x left_hand_y ball_1_x'
                  ' ball_1_y ball_2_x ball_2_y ball_3_x ball_3_y'.split()
        )
        video_data['photo'] = None
        for i, photo_name in video_data['file_name'].iteritems():
            video_data.loc[i, 'photo'] = Image.open(BytesIO(zip.open(self._videos_zip + photo_name).read()))
        return video_data


if __name__ == '__main__':
    DataProvider('ballsandhands.zip').get_video('1-apple-red-room-door')
