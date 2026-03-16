import subprocess


def embed_soft_subs(video_in: str, srt_file: str, video_out: str):
    """Встраивает субтитры как отдельную дорожку (soft subs)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_in,
        "-i",
        srt_file,
        "-c",
        "copy",  # без перекодировки видео/аудио
        "-c:s",
        "mov_text",  # формат субтитров для MP4
        "-metadata:s:s:0",
        "language=rus",
        video_out,
    ]
    subprocess.run(cmd, check=True)
    print(f"Готово: {video_out}")


if __name__ == "__main__":
    video_path = (
        r"C:\Users\user\Downloads\1 - Introduction\1. Welcome to the Course.mp4"
    )
    sub_path = r"C:\Users\user\Downloads\1 - Introduction\1. Welcome to the Course.srt"
    output_path = (
        r"C:\Users\user\Downloads\1 - Introduction\1. Welcome to the Course.subs.mp4"
    )

    embed_soft_subs(video_path, sub_path, output_path)
