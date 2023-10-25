from youtube_dl import YoutubeDL

ydl_opts = {}
with YoutubeDL(ydl_opts) as ydl:
    ydl.download(['https://www.youtube.com/watch?v=zqWOrwBzOjU'])  # `url`은 다운로드하려는 YouTube 동영상의 URL입니다.