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
        self._type = None

    @property
    def list_of_videos(self):
        if self._list_of_videos is None:
            self._list_of_videos = []
            for i in ZipFile(self._zip_file_path).namelist():
                if i.startswith(self._annotations_zip) and i.endswith('csv'):   #szukanie arkuszy
                    for j in range(len(i)-4,3,-1):
                        if i[j] == '/':
                            k=j+1
                            break
                    self._list_of_videos.append(i[k:-4])   # branie tylko nazwy arkusza bez lokalizacji i typu
        return self._list_of_videos

    def get_video(self, video_name: str) -> pd.DataFrame:

        # sprawdza, czy jest taki film - assert = raise error
        assert video_name in self.list_of_videos,\
            f"Video named '{video_name}' wasn't found in zip file '{self._zip_file_path.name}'"

        zip = ZipFile(self._zip_file_path)

        # stworzenie tablicy - odczytanie csv, nazywanie kolumn etc
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


    # możliwe, że jest to jednak zbędne, ale zostawiam na razie
    def data_type(self, video_name):
        zip = ZipFile(self._zip_file_path)
        read_testvideos = zip.open(f'data/testvideos').read().split()
        read_trainvideos = zip.open(f'data/trainvideos').read().split()
        read_validationvideos = zip.open(f'data/validationvideos').read().split()

        video_name = video_name + '.csv'
        for i in range(len(read_testvideos)):
            read_testvideos[i] = str(read_testvideos[i],'utf-8')

        for i in range(len(read_trainvideos)):
            read_trainvideos[i] = str(read_trainvideos[i], 'utf-8')
        for i in range(len(read_validationvideos)):
            read_validationvideos[i] = str(read_validationvideos[i],'utf-8')


        if video_name in read_testvideos:
            self.type = 'test'
        elif video_name in read_trainvideos:
            self.type = 'train'
        elif video_name in read_validationvideos:
            self.type = 'validation'

        return(self.type)

    def get_types(self,type) -> pd.DataFrame:
        zip = ZipFile(self._zip_file_path)

        read_files = zip.open(f'data/{type}videos').read().split()
        video_data = []
        for i in range(len(read_files)):
            video_data.append(self.get_video(str(read_files[i], 'utf-8')[0:-4]))
        pd_data = pd.concat(video_data,ignore_index=True)
        return pd_data







    # def mix(self, video_name):
    #     video_data = self.get_video(video_name)
    #     video_data["type"] = self.data_type(video_name)
    #     return(video_data)




# if __name__ == '__main__':
#     DataProvider('F:/Studia/balls-and-hands-in-videos-of-juggling/ballsandhands.zip').get_video('1-apple-red-room-door')
# # F:\Studia\balls-and-hands-in-videos-of-juggling