import os

def type_and_shape(d, name=None):
    if name:
        print(name+":")
    print("type: ", type(d))
    print("shape: ", d.shape)
    # print("length: ", d.__len__())

def one_batch_shape(d):
    for batch in d.take(1):
        x, y = batch
        print("one batch sample x shape:", x.numpy().shape)
        print("one batch label y shape:", y.numpy().shape)
        return x.numpy().shape, y.numpy().shape

def one_batch_to_file(d, i, dir):
    j = 0
    for batch in d.__iter__():
        if j == i:
            inputs, targets = batch
            # print the first two data points in the batch
            for k in [0, 1]:
                fname = os.path.join(dir, "batch{}[{}].txt".format(j, k))
                f = open(fname, "w")
                f.write("{}".format(inputs[k]) + "\n")
                f.write("{}".format(targets[k]))
                f.close()
                print("Output to file: ", fname)
            break
        j += 1


