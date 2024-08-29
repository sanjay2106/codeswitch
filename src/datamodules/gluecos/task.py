class Task:
    def __init__(self, label2id, name, train_path, val_path,task_no=None):
        self.label2id = label2id
        self.name = name
        self.train_path = train_path
        self.val_path = val_path
        self.task_no = task_no
