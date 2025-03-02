from nikud_and_teamim import remove_nikud, replace_teamim_with_emphasis, remove_nikud_and_teamim, remove_nikud_dicta
import os

def create_steps_files(original_text_path):
    # get the text
    with open(original_text_path, "r") as f:
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
    
    temp_dir = os.path.join(os.getcwd(), "temp")
    # if the directory does not exist, create it
    os.makedirs(temp_dir, exist_ok=True)
    # save all the steps in temp_dir
    with open(os.path.join(temp_dir, "step01.txt"), "w") as f:
        f.write(cleaned_ketiv_maleh)
    with open(os.path.join(temp_dir, "step02.txt"), "w") as f:
        f.write(ketiv_maleh)
    with open(os.path.join(temp_dir, "step03.txt"), "w") as f:
        f.write(ketiv_haser_no_teamim)
    with open(os.path.join(temp_dir, "step04.txt"), "w") as f:
        f.write(ketiv_haser_teamim)
    with open(os.path.join(temp_dir, "final_step.txt"), "w") as f:
        f.write(original_text)

    # print("step 1: \n", cleaned_ketiv_maleh)
    # print("step 2: \n", ketiv_maleh)
    # print("step 3: \n", ketiv_haser_no_teamim)
    # print("step 4: \n", ketiv_haser_teamim)
    # print("with nikud: \n", original_text.split("\n")[0])
    
def remove_steps_files():
    temp_dir = os.path.join(os.getcwd(), "temp")
    for i in range(1, 5):
        os.remove(os.path.join(temp_dir, f"step0{i}.txt"))
    os.remove(os.path.join(temp_dir, "final_step.txt"))