filename = "dataset/kitti2012_train_stereo.txt"

with open(filename, 'r') as f:
    l = f.readlines()

    print(len(l))
    print(l[len(l)-1].strip())
    print([l[i].rstrip().split(" ") for i in range(len(l))])

