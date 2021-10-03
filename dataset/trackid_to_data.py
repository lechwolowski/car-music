import pandas as pd


def create_mapping():
    music_tracks = pd.read_excel("Data_InCarMusic.xlsx", sheet_name="Music Track")
    music_tracks.columns = [x.strip() for x in music_tracks.columns.tolist()]
    mapping = music_tracks[['id', 'artist', 'title']].set_index('id').T.to_dict()
    return lambda track_id: mapping[track_id]

if __name__ == "__main__":
    create_mapping()
    