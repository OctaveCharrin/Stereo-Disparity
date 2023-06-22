
def main(train_filename, test_filename):
    with open(train_filename, 'w') as f:
        for i in range(194):
            line = "Kitti2012/training/colored_0/" + "{:06d}".format(i) + "_10.png" +" "+\
                    "Kitti2012/training/colored_1/" + "{:06d}".format(i) + "_10.png\n"
            f.write(line)

    with open(test_filename,'w') as f:
        for i in range(194):
            line = "Kitti2012/training/disp_noc/" + "{:06d}".format(i) + "_10.png\n"
            f.write(line)

if __name__=="__main__":
    train_file = "kitti2012_training.txt"
    test_file = "kitti2012_testing.txt"
    main(train_filename=train_file, test_filename=test_file)