# TODO: Use re to remove nikud and teamim
# import re
# text = re.sub(r'[\u0591-\u05C7]', '', text)
# nikiud: HATAF_SEGOL:"ֱ",HATAF_PATAH:"ֲ",HATAF_QAMATZ:"ֳ",HIRIQ:"ִ",TSERE:"ֵ",SEGOL:"ֶ",PATAH:"ַ",QAMATZ:"ָ",SIN_DOT:"ׂ",SHIN_DOT:"ׁ",HOLAM:"ֹ",DAGESH:"ּ",QUBUTZ:"ֻ",SHEVA:"ְ",QAMATZ_QATAN:"ׇ"


TEAMIM = ['֑', '֒', '֓', '֔', '֕', '֖', '֗', '֘', '֙', '֚', '֛', '֜', '֝', '֞', '֟', '֠', '֡', '֢', '֣', '֤', '֥', '֦', '֧', '֨', '֩', '֪', '֫', '֬', '֭', '֮', 'ֽ']   
BASE_CHAR = "@"

def remove_nikud(text):
    nikud_list = ["ֱ","ֲ","ֳ","ִ","ֵ","ֶ","ַ","ָ","ׂ","ׁ","ֹ","ּ","ֻ","ְ","ׇ"]
    for nikud in nikud_list:
        text = text.replace(nikud, "")
    return text

def just_teamim(text, base_char = BASE_CHAR):
    new_text = ""
    for char in text:
        if char in TEAMIM:
            new_text += base_char
            new_text += char
        elif char == " ":
            new_text += " "
    return new_text

def remove_makav(text):
    """
    Replace makav with space
    """
    makav_list = ["-","־"]
    for makav in makav_list:
        text = text.replace(makav, " ")
    return text
    
# remove nikud and teamim from a string
def remove_nikud_and_teamim(text):
    nikud_and_teamim_list = ["ֱ","ֲ","ֳ","ִ","ֵ","ֶ","ַ","ָ","ׂ","ׁ","ֹ","ּ","ֻ","ְ","ׇ", '֑', '֒', '֓', '֔', '֕', '֖', '֗', '֘', '֙', '֚', '֛', '֜', '֝', '֞', '֟', '֠', '֡', '֢', '֣', '֤', '֥', '֦', '֧', '֨', '֩', '֪', '֫', '֬', '֭', '֮', 'ֽ','׀']
    for nikud_or_teamim in nikud_and_teamim_list:
        text = text.replace(nikud_or_teamim, "")
    return text

def replace_teamim_with_emphasis(text): # 'ֽ' is the teamim for emphasis in a word
    teamim = ['֑', '֒', '֓', '֔', '֕', '֖', '֗', '֘', '֙', '֚', '֛', '֜', '֝', '֞', '֟', '֠', '֡', '֢', '֣', '֤', '֥', '֦', '֧', '֨', '֩', '֪', '֫', '֬', '֭', '֮', 'ֽ']
    for char in teamim:
        text = text.replace(char,'ֽ').replace("׀", "") # 'ֽ' is the teamim for emphasis in a word. '׀' is the teamim that not located in words
    return text

import requests
def remove_nikud_dicta(text, maleify=False):
  """
  Removes nikud from Hebrew text using Dicta's API.

  Args:
    text: Hebrew text with nikud.
    maleify: Boolean indicating whether to add maleify (אימות קריאה) or not.

  Returns:
    Hebrew text without nikud, or an error message if an error occurred.
  """
  api_endpoint = 'https://remove-nikud-2-0.loadbalancer2.dicta.org.il/api'
  payload = {
    "data": text,
    "genre": "rabbinic", 
    "fQQ": True,
    "maleify": ~maleify,  # Add maleify parameter to the payload
    "dasda": True
  }
  
  try:
    response = requests.post(api_endpoint, json=payload)
    response.raise_for_status()
    cleaned_text = response.json()['results'].replace('\u05BD', '') 
    return cleaned_text
  except requests.exceptions.RequestException as e:
    return f"An error occurred: {e}"



# Example usage in remove
if __name__ == "__main__":
  text = "בְנֵי־הָֽאֱלֹהִים֙ אֶת־בְּנ֣וֹת הָֽאָדָ֔ם כִּ֥י טֹבֹ֖ת הֵ֑נָּה וַיִּקְח֤וּ לָהֶם֙ "

  # Remove nikud without maleify
  cleaned_text_no_maleify = remove_nikud_dicta(text)
  print("Without Maleify:", cleaned_text_no_maleify)

  # Remove nikud with maleify
  cleaned_text_with_maleify = remove_nikud_dicta(text, maleify=True)
  print("With Maleify:", cleaned_text_with_maleify)