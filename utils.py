import glob 

def list_wav_files(folder_path):
    return glob.glob(f"{folder_path}/*.wav")

def list_xml_files(folder_path):
    return glob.glob(f"{folder_path}/*.xml")
