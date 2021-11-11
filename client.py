
import requests

# server url
URL = "http://127.0.0.1:5000/predict"


# audio file we'd like to send for predicting keyword
FILE_PATH = "./When_I_Was_Your_Man.mp3"


if __name__ == "__main__":

    # open files
    file = open(FILE_PATH, "rb")

    # package stuff to send and perform POST request
    values = {"file": (FILE_PATH, file, "audio/mp3")}
    response = requests.post(URL, files=values)
    
    genre = response.json()

    print("Predicted genre: {}".format(genre["genre"]))

    #print("Predicted genre: {}".format(data["genre"]))
