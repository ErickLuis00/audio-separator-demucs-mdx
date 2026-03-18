git clone https://github.com/ErickLuis00/audio-separator-demucs-mdx && cd audio-separator-demucs-mdx && pip install -r requirements.txt && apt update &&  apt install ffmpeg -y && 


cd audio-separator-demucs-mdx/ && python job_worker.py 2>&1 | tee job_worker.log





BETTER ONE IF TERMINALS CLOSES, IT RUN IN BG AND START LOGGING NOW.
python job_worker.py 2>&1 | tee job_worker.log && tail -f job_worker.log





 python -m demucs_separator.run R20240302-013019.WAV

python job_worker.py

cd 