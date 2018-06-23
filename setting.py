import os

# Training dataset.............................
data = 'data'
train_data = "data/train"
test_data = "data/test"
save = 'model/save'
teacher_save = 'model/save/teacher'
student_save = 'model/save/student'

# Training setting.............................
batch_size = 100
# student_follow: 
#       After the number of batches teacher trained, student will use teachers output to train
student_follow = 5 # batches
snapshop_default = 4 # batches

class token:
    def __init__(self):
        self.resource = student_follow
        self.teacher = student_follow
        self.student = -1

    def teacher_turn(self):
        if self.teacher == -1: return False
        else:
            self.teacher -= 1
            if self.teacher == 0:
                self.teacher = -1
                self.student = self.resource
            print('teacher turn'.ljust(25, '*'))
            return True

    def student_turn(self):
        if self.student == -1: return False
        else:
            self.student -= 1
            if self.student == 0:
                self.student = -1
                self.teacher = self.resource
            print('stdent turn'.ljust(25, '*'))
            return True
            
class snapshop_token:
    def __init__(self):
        self.threshold = snapshop_default
        self.value = 0

    def check(self, value = 1):
        if self.value < self.threshold:
            self.value += value
            return False
        else:
            self.value = 0
            return True

# Load model..................................
load_model = None


def dir_init():
    """To init dirs in setting.py """
    params = [
        # training path
        data, train_data, test_data, 
        # save path
        save, teacher_save, student_save
    ]
    for path in params:
        if not os.path.exists(path):
            os.mkdir(path)
    pass
