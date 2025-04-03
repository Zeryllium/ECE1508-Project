import shutil


def zip_adv_data():
    shutil.make_archive(base_name="adversarial_data", format="zip", root_dir="./adversarial")

def unzip_adv_data():
    shutil.unpack_archive(filename="adversarial_data.zip", extract_dir="./adversarial_tmp")

if __name__ == "__main__":
    #zip_adv_data()
    unzip_adv_data()