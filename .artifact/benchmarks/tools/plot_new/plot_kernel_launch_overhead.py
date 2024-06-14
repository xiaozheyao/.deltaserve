# numbers collected from nsys
naive_kernel_time = [11964] # repeatedly called 39 times
naive_total_time = [16710]

ibmm_kernel_time = [338946]
ibmm_total_time = [345989]

def plot(args):
    pass

if __name__ =="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--savepath", type=str, default=".")
    args = parser.parse_args()
    plot(args)