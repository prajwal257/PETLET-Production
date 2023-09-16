# This is the Re-factored Production folder:
- Deahorrhea API is running at: `/predict/diarrhea`
- Jaundice API is running at: `/predict/jaundice/`
- Hybrid Eye Infection API is running at: `/predict/eyeinfection`
- Obesity API running at: `/predict/obesity`
- Constipation API running at: `/predict/constipation`
- Fleas and Ticks Infection API running at: `/predict/fleasinfection`
- Ear Infection API running at: `/predict/earinfection`

## Running Instructions:
- First install all the required dependencies by running the following command: `pip3 install -r requirements.txt`
- This command will ensure that we have the required dependencies with the correct versions installed. Next we directly run the `app.py`.
- Execute the following command: `uvicorn app:app`. With the optional command to choose the approprite port: `--port=8080`, and `--reload` for automatically reloadin it after a failure.
- Total command may look like following: `uvicorn app:app --reload --port=8080`.

## Running as a EC2 Instance:
- Start the EC2 Instance.
- Choose the Ubuntu as OS.
- Ensure the firewall settings.
- Run the following commands:
    - sudo apt-get update
    - sudo apt install python3-pip
    - sudo git clone https://github.com/prajwal257/PETLET-ML-APIs.git
    - cd PETLET-ML-APIs
    - sudo pip install -r requirements.txt --no-cache-dir
    - sudo apt-get install ffmpeg libsm6 libxext6 -y --no-cache-dir
    - sudo python3 -m uvicorn app:app --host 0.0.0.0 --port 8000
- If the container starts at the designated port good, if not then recheck the firewall settings.

#### NOTE
- The free tier AWS EC2 instance only has 8GB of HDD storage so there is a chance that if you install the requirements file 2 times then it will be full and Instance will stop to work.