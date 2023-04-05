import gradio as gr
import pandas as pd
import tensorflow as tf
import numpy as np

def auth(username, password):
    if username == "" and password == "":
        return True
    else:
        return False


def predict(df):
    # TODO:
    df["offansive"] = 1
    df["target"] = None

    # ***************************
    # WRITE YOUR INFERENCE STEPS BELOW
    # from inference import prediction_from_df
    from final_test import test_predict
    offansive_list = []
    target_list = test_predict(df)
    
    for target in target_list:
        if target == 'OTHER':
            offansive_list.append(0)
        else:
            offansive_list.append(1)
            
    df["offansive"] = offansive_list
    df["target"] = target_list
    # *********** END ***********
    return df


def get_file(file):
    output_file = "output_TRT_Data_Warriors.csv"

    # For windows users, replace path seperator
    file_name = file.name.replace("\\", "/")

    df = pd.read_csv(file_name, sep="|")

    predict(df)
    df.to_csv(output_file, index=False, sep="|")
    return (output_file)


# Launch the interface with user password
iface = gr.Interface(get_file, "file", "file")

if __name__ == "__main__":
    iface.launch(share=True, auth=auth, enable_queue=True)