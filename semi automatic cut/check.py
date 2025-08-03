import os
import json
parsha_names = ["Bereshit", "Noach", "LechLecha", "Vayera", "ChayeiSara", "Toldot", "Vayetzei", "Vayishlach", "Vayeshev", "Miketz", "Vayigash", "Vayechi", "Shemot", "Vaera", "Bo", "Beshalach", "Yitro", "Mishpatim", "Terumah", "Tetzaveh", "KiTisa", "Vayakhel", "Pekudei", "Vayikra", "Tzav", "Shmini", "Tazria", "Metzora", "Acharei Mot", "Kedoshim", "Emor", "Behar", "Bechukotai", "Bamidbar", "Nasso", "Behaalotcha", "Shlach", "Korach", "Chukat", "Balak", "Pinchas", "Matot", "Masei", "Devarim", "Vaethanan", "Eikev", "Reeh", "Shoftim", "KiTeitzei", "KiTavo", "Nitzavim", "Vayeilech", "Haazinu", "VezotHaberakhah"]

dir_path = "C:/Users/avivs/OneDrive - Technion/Project/data/עם טעמים/bar-mitzva.com/sfardi audio/"
json_path = dir_path + "data.json"

with open(json_path, "r", encoding="utf-8") as json_file:
    json_data = json.load(json_file)


def check_files(dir_path, nusach_key="sk"):
    """
    Check for missing audio files
    File naming convention: "<nusach><book>-<parsha><aliyah>.mp3"
    Example: "skb-b000.mp3" where:
    - sk = Sefardi nusach
    - b = book (parent)
    - b00 = parsha value
    - 0 = aliyah number (0-7, where 0 is haftarah)
    """
    for parsha in json_data:
        for i in range(0, 8):
            file_name = nusach_key + parsha["parent"] + "-" + parsha["value"] + str(i) + ".mp3"
            if not os.path.isfile(dir_path + file_name):
                print(file_name)


check_files(dir_path)

