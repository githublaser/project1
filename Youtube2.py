from pytube import YouTube

DOWNLOAD_FOLDER = "C:\\Users\\Kwan_Kim\\Downloads"
url = "https://www.youtube.com/watch?v=zqWOrwBzOjU"
yt = YouTube(url)

print("Available streams:")
for stream in yt.streams:
    print(stream)

stream = yt.streams.get_by_itag(22)  # itag 22는 720p 화질의 mp4 동영상을 나타냅니다.

if stream is None:
    print("Stream with itag=22 is not available for this video.")
else:
    stream.download(DOWNLOAD_FOLDER)