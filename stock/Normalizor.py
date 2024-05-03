class Normalizor:
    def __init__(self, data):
        self.a = data.mean(axis=0)
        self.b = data.std(axis=0)
        # self.a = data.min(axis=0)
        # self.b = data.max(axis=0) - self.a

    def normalize(self, data, indices=None):
        if indices == None:
            return (data - self.a) / self.b
        else:
            # normalize the specific columns only
            a = [self.a[i] for i in indices]
            b = [self.b[i] for i in indices]
            return (data - a) / b

    def denormalize(self, data, indices=None):
        if indices == None:
            return data * self.b + self.a
        else:
            # denormalize the specific columns only
            a = [self.a[i] for i in indices]
            b = [self.b[i] for i in indices]
            return data * b + a