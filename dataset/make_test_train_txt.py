import random

def main(source_stereo, source_gt, train_stereo, train_gt, test_stereo, test_gt):
    stereo_list = open(source_stereo, 'r').readlines()
    gt_list = open(source_gt, 'r').readlines()

    n = len(stereo_list)
    index = list(range(n))
    random.shuffle(index)

    index_test = index[:int(0.2*n)]
    index_train = index[int(0.2*n):]
    
    with open(train_stereo, 'w') as f:
        for idx in index_train:
            f.write(stereo_list[idx])
    with open(train_gt, 'w') as f:
        for idx in index_train:
            f.write(gt_list[idx])

    with open(test_stereo, 'w') as f:
        for idx in index_test:
            f.write(stereo_list[idx])
    with open(test_gt, 'w') as f:
        for idx in index_test:
            f.write(gt_list[idx])


if __name__=="__main__":
    source_stereo_file = "kitti2012_all_stereo.txt"
    source_gt_file = "kitti2012_all_gtdisp.txt"
    train_stereo_file = "kitti2012_train_stereo.txt"
    train_gtdisp_file = "kitti2012_train_gtdisp.txt"
    test_stereo_file = "kitti2012_test_stereo.txt"
    test_gtdisp_file = "kitti2012_test_gtdisp.txt"
    main(source_stereo=source_stereo_file, source_gt=source_gt_file, train_stereo=train_stereo_file, train_gt=train_gtdisp_file, test_stereo=test_stereo_file, test_gt=test_gtdisp_file)