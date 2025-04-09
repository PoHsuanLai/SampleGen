import os
import yt_dlp

# Folder to save the audio
DIR = "data/artists"
os.makedirs(DIR, exist_ok=True)

# YouTube playlists
YOUTUBE_LINKS = {
    # 'kanye': 'https://www.youtube.com/playlist?list=PLTDmNT4owFz2pivbgvdIIUR3yYTx6LCY0',
    # 'drake': 'https://www.youtube.com/playlist?list=PLikX3X1eunqsk0rrqKUpjjiLwF-3u5H7Z',
    # 'kendrick': 'https://www.youtube.com/playlist?list=PLTDmNT4owFz1bHAzwLiSgku0_Jimde-47',
    # 'future': 'https://www.youtube.com/playlist?list=PLoW2sK576siqaGTHA15Cb2MqrqltvjMYF', 
    # 'denzel_curry': 'https://www.youtube.com/playlist?list=PL1-eQOG75OnT8m-w3iGszJkt70G_htN6A',
    # 'j_cole': 'https://www.youtube.com/playlist?list=PLq3UZa7STrbrWBkMPTzGAHI0WGDdkGuW8', 
    # 'lil_uzi': 'https://www.youtube.com/playlist?list=PL8-o_KZwk58w5Y49dY3lIBPJTliJJBChF', 
    # 'travis': 'https://www.youtube.com/playlist?list=PLMsAGxVktam_nCchp_IzisoKEvNCP1kqR'
    'pop_smoke': 'https://www.youtube.com/playlist?list=PLqA3a-7gfFDCSlHUwxS4RdL5Wnd4dn-vP'
}

def download_audio(artist, playlist_url):
    output_dir = os.path.join(DIR, artist)
    os.makedirs(output_dir, exist_ok=True)

    # This archive file keeps track of downloaded video IDs
    archive_file = os.path.join(output_dir, "downloaded.txt")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(playlist_index)s - %(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'noplaylist': False,  # download entire playlist
        'ignoreerrors': True,  # skip videos that cause errors
        'download_archive': archive_file,  # avoid redownloading
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            print(f"Downloading playlist for: {artist}")
            ydl.download([playlist_url])
        except Exception as e:
            print(f"Error downloading {artist}'s playlist: {e}")

if __name__ == "__main__":
    for artist, url in YOUTUBE_LINKS.items():
        download_audio(artist, url)
