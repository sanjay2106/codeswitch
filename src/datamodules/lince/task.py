class Task:
    def __init__(self, label2id, name, train_path, val_path):
        self.label2id = label2id
        self.name = name
        self.train_path = train_path
        self.val_path = val_path
