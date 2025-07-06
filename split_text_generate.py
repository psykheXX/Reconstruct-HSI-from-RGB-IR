import os

# data_path = "D:/potato_hyper_dataset/Train_Spec/"
#
# output_file = './split_txt/train_list.txt'
#
# files = os.listdir(data_path)
#
# with open(output_file, 'w') as f:
#     for file in files:
#         if file.endswith('.mat'):
#             file_name = os.path.splitext(file)[0]
#             f.write(file_name + '\n')
#
# print(f"文件名已写入 {output_file}")

data_dir = "D:/potato_hyper_dataset/Valid_Spec/"

valid_output_file = './split_txt/test_list.txt'

os.makedirs(os.path.dirname(valid_output_file), exist_ok=True)

files = os.listdir(data_dir)

with open(valid_output_file, 'w') as f:
    for file in files:
        if file.endswith('.mat'):
            file_name = os.path.splitext(file)[0]
            f.write(file_name + '\n')

print(f"文件名已写入 {valid_output_file}")