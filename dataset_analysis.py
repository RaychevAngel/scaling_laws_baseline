from datasets import load_from_disk
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

train_dataset0 = load_from_disk("data/value/iteration_0/train").shuffle(seed=42)
#train_dataset1 = load_from_disk("data/value/iteration_1/train")
#train_dataset2 = load_from_disk("data/value/iteration_2/train")

dev_dataset0 = load_from_disk("data/value/iteration_0/dev").shuffle(seed=42)
#dev_dataset1 = load_from_disk("data/value/iteration_1/dev")
#dev_dataset2 = load_from_disk("data/value/iteration_2/dev")



for i in tqdm(range(100)):
    print(train_dataset0[i]["prompt"][0]["content"])
    print(train_dataset0[i]["completion"][0]["content"])

train_count_0_negative = 0
train_count_0_positive = 0

train_count_1_negative = 0
train_count_1_positive = 0

train_count_2_negative = 0
train_count_2_positive = 0

dev_count_0_negative = 0
dev_count_0_positive = 0

dev_count_1_negative = 0
dev_count_1_positive = 0

dev_count_2_negative = 0
dev_count_2_positive = 0


for i in tqdm(range(len(train_dataset0))):
    if train_dataset0[i]["completion"][0]["content"] == "0":
        train_count_0_negative += 1
    elif train_dataset0[i]["completion"][0]["content"] == "1":
        train_count_0_positive += 1

for i in tqdm(range(len(dev_dataset0))):
    if dev_dataset0[i]["completion"][0]["content"] == "0":
        dev_count_0_negative += 1
    elif dev_dataset0[i]["completion"][0]["content"] == "1":
        dev_count_0_positive += 1


#for i in tqdm(range(len(train_dataset1))):
#    if train_dataset1[i]["completion"][0]["content"] == "0":
#        train_count_1_negative += 1
#    elif train_dataset1[i]["completion"][0]["content"] == "1":
#        train_count_1_positive += 1

#for i in tqdm(range(len(dev_dataset1))):
#    if dev_dataset1[i]["completion"][0]["content"] == "0":
#        dev_count_1_negative += 1
#    elif dev_dataset1[i]["completion"][0]["content"] == "1":
#        dev_count_1_positive += 1


#for i in tqdm(range(len(train_dataset2))):
#    if train_dataset2[i]["completion"][0]["content"] == "0":
#        train_count_2_negative += 1
#    elif train_dataset2[i]["completion"][0]["content"] == "1":
#        train_count_2_positive += 1


#for i in tqdm(range(len(dev_dataset2))):
#    if dev_dataset2[i]["completion"][0]["content"] == "0":
#        dev_count_2_negative += 1
#    elif dev_dataset2[i]["completion"][0]["content"] == "1":
#        dev_count_2_positive += 1

print("Train 0: ", train_count_0_negative, train_count_0_positive, train_count_0_positive/(train_count_0_negative+train_count_0_positive))
print("Dev 0: ", dev_count_0_negative, dev_count_0_positive, dev_count_0_positive/(dev_count_0_negative+dev_count_0_positive))
print("Train 1: ", train_count_1_negative, train_count_1_positive, train_count_1_positive/(train_count_1_negative+train_count_1_positive))
print("Dev 1: ", dev_count_1_negative, dev_count_1_positive, dev_count_1_positive/(dev_count_1_negative+dev_count_1_positive))
print("Train 2: ", train_count_2_negative, train_count_2_positive, train_count_2_positive/(train_count_2_negative+train_count_2_positive))
print("Dev 2: ", dev_count_2_negative, dev_count_2_positive, dev_count_2_positive/(dev_count_2_negative+dev_count_2_positive))