import numpy as np
from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()

    # forward

    # f0d = diff_helper.load_info("./forward/forward_paddle.npy")
    # f0t = diff_helper.load_info("./forward/forward_torch.npy")
    # diff_helper.compare_info(f0d, f0t)
    # diff_helper.report(
    #     diff_method="mean", diff_threshold=1e-6, path="./forward_diff.txt")

    # forward_D
    f1d = diff_helper.load_info("./forward/forward_D_paddle.npy")
    f1t = diff_helper.load_info("./forward/forward_D_torch.npy")
    diff_helper.compare_info(f1d, f1t)
    diff_helper.report(
        diff_method="mean", diff_threshold=1e-6, path="./forward_D_diff.txt")


    # backward

    # # lr1
    # l0d1 = diff_helper.load_info("./backward/lr1_backward_paddle.npy")
    # l0t1 = diff_helper.load_info("./backward/lr1_backward_torch.npy")
    #
    # diff_helper.compare_info(l0d1, l0t1)
    # diff_helper.report(
    #     diff_method="mean", diff_threshold=1e-5, path="./lr1_backward_diff.txt")

    # # lr2
    # l0d2 = diff_helper.load_info("./backward/lr2_backward_paddle.npy")
    # l0t2 = diff_helper.load_info("./backward/lr2_backward_torch.npy")
    #
    # diff_helper.compare_info(l0d2, l0t2)
    # diff_helper.report(
    #     diff_method="mean", diff_threshold=1e-5, path="./lr2_backward_diff.txt")

    # loss
    # s0d = diff_helper.load_info("./backward/loss_backward_paddle.npy")
    # s0t = diff_helper.load_info("./backward/loss_backward_torch.npy")
    #
    # diff_helper.compare_info(s0d, s0t)
    # diff_helper.report(
    #     diff_method="mean", diff_threshold=1e-5, path="./loss_backward_diff.txt")

    # # metric
    # metricd = diff_helper.load_info("./metric/metric_paddle.npy")
    # metrict = diff_helper.load_info("./metric/metric_torch.npy")
    # diff_helper.compare_info(metricd, metrict)
    # diff_helper.report(
    #     diff_method="mean", diff_threshold=1e-5, path="./metric_diff.txt")
