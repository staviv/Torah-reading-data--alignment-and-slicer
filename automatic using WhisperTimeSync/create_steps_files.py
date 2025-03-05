from nikud_and_teamim import remove_nikud, replace_teamim_with_emphasis, remove_nikud_and_teamim, remove_nikud_dicta
import os

def create_steps_files(text_path, output_dir="temp"):
    """
    Create intermediate text files for synchronization steps.
    
    Args:
        text_path: Path to the original text file
        output_dir: Directory to create step files in
    """
    # get the text
    with open(text_path, "r") as f:
        # first lines
        original_text = f.read()

    # prepare the text
    original_text = original_text.replace("׃ \n", "׃ ").replace("־", "־ ")

    # the first steps are ketiv maleh, for that we use the API of Dicta
    ketiv_maleh = remove_nikud_dicta(original_text) # step 2

    # before the step of ketiv maleh we want another step of text without "׃" or "־".
    cleaned_ketiv_maleh = ketiv_maleh.replace("־", "").replace("׃", ".") # step 1

    # the second step is the ketiv haser without teamim, for that we use our own library
    ketiv_haser_no_teamim = remove_nikud_and_teamim(original_text) # step 3

    # the third step is the ketiv haser with teamim, for that we use our own library
    ketiv_haser_teamim = remove_nikud(original_text) # step 4
    
    # if the directory does not exist, create it
    os.makedirs(output_dir, exist_ok=True)
    # save all the steps in output_dir
    with open(os.path.join(output_dir, "step01.txt"), "w") as f:
        f.write(cleaned_ketiv_maleh)
    with open(os.path.join(output_dir, "step02.txt"), "w") as f:
        f.write(ketiv_maleh)
    with open(os.path.join(output_dir, "step03.txt"), "w") as f:
        f.write(ketiv_haser_no_teamim)
    with open(os.path.join(output_dir, "step04.txt"), "w") as f:
        f.write(ketiv_haser_teamim)
    with open(os.path.join(output_dir, "final_step.txt"), "w") as f:
        f.write(original_text)

    # print("step 1: \n", cleaned_ketiv_maleh)
    # print("step 2: \n", ketiv_maleh)
    # print("step 3: \n", ketiv_haser_no_teamim)
    # print("step 4: \n", ketiv_haser_teamim)
    # print("with nikud: \n", original_text.split("\n")[0])
    
def remove_steps_files(output_dir="temp"):
    """
    Remove temporary step files from the specified directory.
    """
    for file in ["step01.txt", "step01.txt.srt", "step02.txt", "step02.txt.srt", 
                "step03.txt", "step03.txt.srt", "step04.txt", "step04.txt.srt", 
                "final_step.txt", "final_step.txt.srt"]:
        try:
            os.remove(os.path.join(output_dir, file))
        except FileNotFoundError:
            pass